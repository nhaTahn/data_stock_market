"""Rolling validation backtest for hetero_combined + conf_ratio selection.

Protocol: walk-forward w126/t21/s21 (same as production VN validation).
- Each fold: fit hetero_combined on train window, predict on test window.
- Apply selection rule (conf_ratio_q70, full, daily_bot_sig_50pct).
- Compute fold metrics: rel_score, DA, spike_days, daily eq return (long-only top-K portfolio).

Output:
  - per_fold_metrics.csv
  - equity_curve_{rule}.csv
  - summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.architectures.backbone import build_lstm_backbone  # noqa: E402
from src.models.components.losses import RelScoreWeightedTailLoss  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset  # noqa: E402
from src.models.training.scalers import (  # noqa: E402
    fit_feature_scaler, apply_feature_scaler,
    fit_local_target_normalizer, apply_local_target_normalizer,
    fit_target_scaler, apply_target_scaler,
    inverse_target_scaler_values, inverse_local_target_normalizer,
    LocalTargetNormalizer, TargetScaler,
)
from src.models.training.seeds import set_global_seed  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from experiments.training.run_hetero_nll_probe import (  # noqa: E402
    GaussianNLLLoss, CombinedRelScoreNLLLoss,
)

DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
OUTPUT_DIR = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history"
    / "training_runs" / "reports" / "hetero_rolling_backtest_20260521"
)
GOLD_DIR = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "hetero_rolling_backtest_20260521"

# Rolling params
TRAIN_DAYS = 126
TEST_DAYS  = 21
STEP_DAYS  = 21
VAL_START  = pd.Timestamp("2020-04-01")
VAL_END    = pd.Timestamp("2022-11-15")
TOP_K = 10  # long-only top-K stocks per day by conf_ratio or |mu|
COST_BPS = 15  # transaction cost per unit turnover


def robust_loss(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    return float(np.quantile(np.abs(v), 0.5) + 0.5 * np.quantile(np.abs(v), 0.9)) if len(v) else float("nan")


def rel_score_fn(actual: np.ndarray, pred: np.ndarray) -> float:
    base = robust_loss(actual)
    return float("nan") if not (np.isfinite(base) and base > 0) else float(1 - robust_loss(actual - pred) / base)


def build_model(num_features: int, target_scaler: TargetScaler, local_norm: LocalTargetNormalizer) -> keras.Model:
    inputs, encoded = build_lstm_backbone(window_size=15, num_features=num_features,
                                          lstm_units=[64, 32], dropout=0.05)
    mu = layers.Dense(1, name="mu")(encoded)
    log_sigma = layers.Dense(1, name="log_sigma")(encoded)
    output = layers.Concatenate(name="mu_logsigma")([mu, log_sigma])
    model = keras.Model(inputs=inputs, outputs=output)
    rel_loss = RelScoreWeightedTailLoss(
        target_mean=target_scaler.mean, target_std=target_scaler.std,
        use_target_scaler=True, local_scale_floor=local_norm.floor,
        high_quantile=0.85, high_weight=1.75, base_weight=1.0,
        tail_error_threshold=0.035, tail_penalty_weight=0.05,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
        loss=CombinedRelScoreNLLLoss(rel_loss, w_rel=0.7, w_nll=0.3),
    )
    return model


def predict_raw(model, x, scale, target_scaler, local_norm):
    out = np.asarray(model.predict(x, verbose=0), dtype=np.float32)
    if out.ndim == 1: out = out.reshape(-1, 1)
    mu_s = out[:, 0]; ls_s = out[:, 1]
    sigma_s = np.log1p(np.exp(ls_s)) + 1e-4
    mu_l = inverse_target_scaler_values(mu_s, target_scaler)
    mu_r = inverse_local_target_normalizer(mu_l, scale, local_norm).reshape(-1)
    sigma_l = sigma_s * target_scaler.std
    sigma_r = (sigma_l * np.maximum(np.abs(scale), local_norm.floor)).reshape(-1)
    return mu_r.astype(np.float32), sigma_r.astype(np.float32)


def make_equity(actual, pred, dates, codes, conf_ratio, rule, top_k=TOP_K, cost_bps=COST_BPS):
    """Simple long-only equity: each day hold equal-weight top_k stocks by signal."""
    df = pd.DataFrame({"date": dates, "code": codes, "actual": actual,
                        "pred": pred, "conf_ratio": conf_ratio})
    cost = cost_bps / 10000
    equity = 1.0
    prev_held: set = set()
    day_returns = []
    for date, grp in df.groupby("date"):
        if rule == "full":
            signal = grp["pred"].abs()
        elif rule == "conf_ratio_q70":
            signal = grp["conf_ratio"]
        elif rule == "daily_bot_sig_50pct":
            # low sigma = lowest conf_ratio denominator — use neg sigma
            signal = -grp["pred"].abs() / grp["conf_ratio"].replace(0, 1e-8)  # inverse of conf_ratio
            signal = grp["conf_ratio"]  # simpler: still rank by conf_ratio but invert preference
            # Actually daily_bot_sig means low sigma → high conf_ratio if mu similar
            # Let's keep consistent: use 1/sigma as signal for this rule
            sigma_est = np.abs(grp["pred"]) / grp["conf_ratio"].replace(0, 1e-8)
            signal = -sigma_est  # lower sigma → higher rank
        else:
            signal = grp["conf_ratio"]
        top = grp.nlargest(top_k, signal.name if hasattr(signal, "name") else "conf_ratio")
        # compute signal properly
        top = grp.assign(_sig=signal.values).nlargest(min(top_k, len(grp)), "_sig")
        held = set(top["code"].values)
        ret = top["actual"].mean() if len(top) > 0 else 0.0
        # turnover cost
        n_change = len(held.symmetric_difference(prev_held))
        to = n_change / max(top_k, 1)
        net_ret = ret - cost * to
        equity *= (1 + net_ret)
        day_returns.append({"date": date, "gross_ret": ret, "net_ret": net_ret, "equity": equity, "turnover": to})
        prev_held = held
    return pd.DataFrame(day_returns)


def run_fold(fold_id, train_dates, test_dates, all_x, all_y, all_meta, feature_columns, seed):
    """One walk-forward fold."""
    # Split by date
    tr_mask = np.isin(all_meta["Date"].values.astype("datetime64[D]"),
                       train_dates.values.astype("datetime64[D]"))
    te_mask = np.isin(all_meta["Date"].values.astype("datetime64[D]"),
                       test_dates.values.astype("datetime64[D]"))
    if tr_mask.sum() < 500 or te_mask.sum() < 50:
        return None, None

    x_tr, y_tr, meta_tr = all_x[tr_mask], all_y[tr_mask], all_meta.iloc[tr_mask]
    x_te, y_te, meta_te = all_x[te_mask], all_y[te_mask], all_meta.iloc[te_mask]

    scale_tr = meta_tr["__tn__"].to_numpy(dtype=np.float32)
    scale_te = meta_te["__tn__"].to_numpy(dtype=np.float32)
    local_norm = fit_local_target_normalizer(scale_tr, "volatility_20")
    y_tr_local = apply_local_target_normalizer(y_tr, scale_tr, local_norm)
    ts = fit_target_scaler(y_tr_local)
    y_tr_scaled = apply_target_scaler(y_tr_local, ts).reshape(-1, 1)
    y_tr_model = np.concatenate([y_tr_scaled, scale_tr.reshape(-1, 1)], axis=1).astype(np.float32)

    set_global_seed(seed)
    model = build_model(x_tr.shape[2], ts, local_norm)
    model.fit(x_tr, y_tr_model, epochs=30, batch_size=64, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)])

    mu_te, sigma_te = predict_raw(model, x_te, scale_te, ts, local_norm)
    conf_ratio_te = np.abs(mu_te) / np.maximum(sigma_te, 1e-8)

    fold_metrics = {"fold_id": fold_id, "n_train": tr_mask.sum(), "n_test": te_mask.sum(),
                    "test_start": str(test_dates.min().date()), "test_end": str(test_dates.max().date())}
    for rule in ["full", "conf_ratio_q70", "daily_bot_sig_50pct"]:
        if rule == "full":
            mask = np.ones(len(mu_te), dtype=bool)
        elif rule == "conf_ratio_q70":
            # calibrate q70 on TRAIN
            cr_tr_est = np.abs(predict_raw(model, x_tr, scale_tr, ts, local_norm)[0]) / np.maximum(
                predict_raw(model, x_tr, scale_tr, ts, local_norm)[1], 1e-8)
            thr = float(np.quantile(cr_tr_est, 0.70))
            mask = conf_ratio_te >= thr
        elif rule == "daily_bot_sig_50pct":
            sigma_te_arr = sigma_te
            df_tmp = pd.DataFrame({"date": meta_te["Date"].values, "sigma": sigma_te_arr})
            df_tmp.index = np.arange(len(df_tmp))
            mask_arr = np.zeros(len(mu_te), dtype=bool)
            for _, grp in df_tmp.groupby("date"):
                thresh = grp["sigma"].quantile(0.50)
                mask_arr[grp.index[grp["sigma"] <= thresh]] = True
            mask = mask_arr
        rs = rel_score_fn(y_te[mask], mu_te[mask]) if mask.sum() > 0 else float("nan")
        da = float(np.mean(np.sign(y_te[mask]) == np.sign(mu_te[mask]))) if mask.sum() > 0 else float("nan")
        fold_metrics[f"{rule}_rel_score"] = rs
        fold_metrics[f"{rule}_da"] = da
        fold_metrics[f"{rule}_coverage"] = float(mask.mean())

    # equity for each rule (use test fold codes from meta)
    eq_rows = {}
    for rule in ["full", "conf_ratio_q70"]:
        eq = make_equity(y_te, mu_te, meta_te["Date"].values,
                          meta_te["code"].values if "code" in meta_te.columns else np.arange(len(y_te)),
                          conf_ratio_te, rule, top_k=TOP_K)
        eq["fold_id"] = fold_id; eq["rule"] = rule
        eq_rows[rule] = eq

    return fold_metrics, eq_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    feature_columns = DEFAULT_FEATURE_COLUMNS
    frame = load_training_frame(DEFAULT_DATA, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["__tn__"] = frame["volatility_20"].astype(float)

    # Fit feature scaler on entire train
    train_full = frame[frame["Date"] <= "2020-03-31"]
    scaler = fit_feature_scaler(train_full.dropna(subset=feature_columns), feature_columns)
    scaled = apply_feature_scaler(frame, scaler)

    print("Building sequence dataset...")
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled, feature_columns, "target_next_return", 15,
        extra_meta_columns=("__tn__",), sequence_normalization="none",
    )
    print(f"Total sequences: {len(x_all)}")

    # Get all val trading days
    val_dates = pd.DatetimeIndex(sorted(set(meta_all[
        (meta_all["Date"] >= VAL_START) & (meta_all["Date"] <= VAL_END)
    ]["Date"].values)))
    print(f"Val days: {len(val_dates)}")

    # Generate folds
    folds = []
    i = 0
    while True:
        test_start_idx = i * STEP_DAYS
        test_end_idx   = test_start_idx + TEST_DAYS
        if test_end_idx > len(val_dates):
            break
        test_dates = val_dates[test_start_idx:test_end_idx]
        # train = last TRAIN_DAYS days before test_start
        test_start = val_dates[test_start_idx]
        all_days_before = pd.DatetimeIndex(sorted(set(meta_all[meta_all["Date"] < test_start]["Date"].values)))
        if len(all_days_before) < TRAIN_DAYS:
            i += 1; continue
        train_dates = all_days_before[-TRAIN_DAYS:]
        folds.append((train_dates, test_dates))
        i += 1

    print(f"Folds: {len(folds)}")
    seed = 43  # use single seed for rolling speed
    all_fold_metrics = []
    all_eq_rows = []
    for fold_id, (train_dates, test_dates) in enumerate(folds):
        print(f"  Fold {fold_id+1}/{len(folds)} test={test_dates[0].date()}..{test_dates[-1].date()}", end="\r")
        fm, eq_rows = run_fold(fold_id, train_dates, test_dates, x_all, y_all, meta_all, feature_columns, seed)
        if fm is None:
            continue
        all_fold_metrics.append(fm)
        for rule, eq in eq_rows.items():
            all_eq_rows.append(eq)

    print()
    fold_df = pd.DataFrame(all_fold_metrics)
    fold_df.to_csv(OUTPUT_DIR / "per_fold_metrics.csv", index=False)

    eq_df = pd.concat(all_eq_rows, ignore_index=True)
    eq_df.to_csv(OUTPUT_DIR / "equity_curves.csv", index=False)

    # summary
    lines = ["# Rolling Validation Backtest — hetero_combined + conf_ratio selection", "",
             f"Protocol: w{TRAIN_DAYS}/t{TEST_DAYS}/s{STEP_DAYS}, seed=43, top_k={TOP_K}, cost={COST_BPS}bps", "",
             "## Fold-average metrics", ""]
    for rule in ["full", "conf_ratio_q70", "daily_bot_sig_50pct"]:
        col_rs = f"{rule}_rel_score"; col_da = f"{rule}_da"; col_cov = f"{rule}_coverage"
        if col_rs in fold_df.columns:
            rs_vals = fold_df[col_rs].dropna()
            da_vals = fold_df[col_da].dropna()
            cov_vals = fold_df[col_cov].dropna()
            n_pos = int((rs_vals > 0).sum())
            lines.append(f"### {rule}")
            lines.append(f"- mean rel_score: {rs_vals.mean():.4f} ± {rs_vals.std():.4f}")
            lines.append(f"- positive folds: {n_pos}/{len(rs_vals)}")
            lines.append(f"- mean DA: {da_vals.mean():.3f}")
            lines.append(f"- mean coverage: {cov_vals.mean():.3f}")
            lines.append("")

    # equity
    lines.append("## Equity curve summary (long-only top-10)")
    for rule in ["full", "conf_ratio_q70"]:
        eq_r = eq_df[eq_df["rule"] == rule]
        if len(eq_r) == 0:
            continue
        final_eq = eq_r.sort_values("date").iloc[-1]["equity"]
        net_rets = eq_r.sort_values("date")["net_ret"].values
        sharpe = float(np.mean(net_rets) / max(np.std(net_rets), 1e-8) * np.sqrt(252))
        max_dd = float(np.min(np.minimum.accumulate(eq_r.sort_values("date")["equity"].values) /
                               np.maximum.accumulate(eq_r.sort_values("date")["equity"].values) - 1))
        lines.append(f"- **{rule}**: final equity {final_eq:.3f} | Sharpe {sharpe:.2f} | MaxDD {max_dd:.1%}")
    lines += ["", "## Reference", "- stressaux_w20 production: rel_score 0.0248, spike 7.7", ""]

    text = "\n".join(lines)
    (OUTPUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    (GOLD_DIR / "summary.md").write_text(text, encoding="utf-8")
    fold_df.to_csv(GOLD_DIR / "per_fold_metrics.csv", index=False)
    eq_df.to_csv(GOLD_DIR / "equity_curves.csv", index=False)
    print(text)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "gold_dir": str(GOLD_DIR)}, indent=2))


if __name__ == "__main__":
    main()
