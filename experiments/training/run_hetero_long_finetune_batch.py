"""Long fine-tune / validation batch for hetero_combined selection policies.

Designed for a 3–4 hour run. It trains hetero_combined models on rolling folds,
evaluates a compact grid of advisor-track and concentrated policies, and writes
resume-friendly per-fold artifacts.

Scope:
- VN validation only (2020-04-01 -> 2022-11-15). Holdout/test is NOT used.
- Rolling folds use train window 126 days, test window 21 days, step 21 days.
- Optional fold/seed caps support smoke tests.

Main questions:
1. Does conf_ratio selection remain stable across multiple rolling seeds?
2. Does min_positions >= 5/6 still pass after pressure gate?
3. Which rebalance/top_k/config is the best advisor-track candidate?

Outputs:
- per_fold_policy_metrics.csv
- policy_summary.csv
- summary.md
- fold prediction npz files under predictions/

Example:
  venv/bin/python experiments/training/run_hetero_long_finetune_batch.py \
    --seeds 43,52,62 --epochs 30 --patience 5

Smoke test:
  venv/bin/python experiments/training/run_hetero_long_finetune_batch.py \
    --seeds 43 --epochs 1 --max-folds 1 --output-dir /tmp/hetero_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
from src.models.training.datasets import build_sequence_dataset  # noqa: E402
from src.models.training.scalers import (  # noqa: E402
    LocalTargetNormalizer,
    TargetScaler,
    apply_feature_scaler,
    apply_local_target_normalizer,
    apply_target_scaler,
    fit_feature_scaler,
    fit_local_target_normalizer,
    fit_target_scaler,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)
from src.models.training.seeds import set_global_seed  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from experiments.training.run_hetero_nll_probe import CombinedRelScoreNLLLoss  # noqa: E402

DEFAULT_DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data/processed/assets/data_info_vn/history/training_runs/reports"
    / "hetero_long_finetune_batch_20260522"
)
DEFAULT_GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/hetero_long_finetune_batch_20260522"

TRAIN_DAYS = 126
TEST_DAYS = 21
STEP_DAYS = 21
VAL_START = pd.Timestamp("2020-04-01")
VAL_END = pd.Timestamp("2022-11-15")
COST_BPS = 15.0


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    rule: str
    gate: str
    rebalance_every: int
    top_k: int
    min_positions: int


def parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def rel_score_fn(actual: np.ndarray, pred: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float(1.0 - robust_loss(actual - pred) / base)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    return float((equity / peak.replace(0.0, np.nan) - 1.0).min())


def sharpe(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std <= 0.0:
        return float("nan")
    return float(clean.mean() / std * np.sqrt(252.0))


def build_model(
    num_features: int,
    target_scaler: TargetScaler,
    local_norm: LocalTargetNormalizer,
    *,
    window_size: int,
    lstm_units: list[int],
    dropout: float,
    lr: float,
    w_rel: float,
    w_nll: float,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        dropout=dropout,
    )
    mu = layers.Dense(1, name="mu")(encoded)
    log_sigma = layers.Dense(1, name="log_sigma")(encoded)
    output = layers.Concatenate(name="mu_logsigma")([mu, log_sigma])
    rel_loss = RelScoreWeightedTailLoss(
        target_mean=target_scaler.mean,
        target_std=target_scaler.std,
        use_target_scaler=True,
        local_scale_floor=local_norm.floor,
        high_quantile=0.85,
        high_weight=1.75,
        base_weight=1.0,
        tail_error_threshold=0.035,
        tail_penalty_weight=0.05,
    )
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=CombinedRelScoreNLLLoss(rel_loss, w_rel=w_rel, w_nll=w_nll),
    )
    return model


def predict_raw(
    model: keras.Model,
    x: np.ndarray,
    scale: np.ndarray,
    target_scaler: TargetScaler,
    local_norm: LocalTargetNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(model.predict(x, verbose=0), dtype=np.float32)
    if output.ndim == 1:
        output = output.reshape(-1, 1)
    mu_scaled = output[:, 0]
    log_sigma_scaled = output[:, 1]
    sigma_scaled = np.log1p(np.exp(log_sigma_scaled)) + 1e-4
    mu_local = inverse_target_scaler_values(mu_scaled, target_scaler)
    mu_raw = inverse_local_target_normalizer(mu_local, scale, local_norm).reshape(-1)
    sigma_local = sigma_scaled * target_scaler.std
    sigma_raw = (sigma_local * np.maximum(np.abs(scale), local_norm.floor)).reshape(-1)
    return mu_raw.astype(np.float32), sigma_raw.astype(np.float32)


def pressure_gate_by_date(frame: pd.DataFrame) -> pd.Series:
    tmp = frame[["Date", "buying_pressure", "selling_pressure"]].dropna().copy()
    tmp["pressure_delta"] = tmp["buying_pressure"].astype(float) - tmp["selling_pressure"].astype(float)
    daily = tmp.groupby("Date")["pressure_delta"].mean().sort_index()
    return (daily.rolling(20, min_periods=5).mean() >= 0).astype(bool)


def make_fold_dates(meta_all: pd.DataFrame, *, max_folds: int | None = None) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    val_days = pd.DatetimeIndex(
        sorted(
            set(meta_all.loc[(meta_all["Date"] >= VAL_START) & (meta_all["Date"] <= VAL_END), "Date"].values)
        )
    )
    all_days = pd.DatetimeIndex(sorted(set(meta_all["Date"].values)))
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    start_idx = 0
    while True:
        end_idx = start_idx + TEST_DAYS
        if end_idx > len(val_days):
            break
        test_dates = val_days[start_idx:end_idx]
        test_start = test_dates[0]
        days_before = all_days[all_days < test_start]
        if len(days_before) >= TRAIN_DAYS:
            folds.append((days_before[-TRAIN_DAYS:], test_dates))
        start_idx += STEP_DAYS
        if max_folds is not None and len(folds) >= max_folds:
            break
    return folds


def build_sequence_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.Series]:
    frame = load_training_frame(args.data, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["__tn__"] = frame[args.target_normalizer].astype(float)
    feature_columns = tuple(DEFAULT_FEATURE_COLUMNS)
    required = {"Date", "code", args.target_column, args.target_normalizer, "__tn__", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    train_full = frame.loc[frame["Date"] <= "2020-03-31"].copy()
    scaler = fit_feature_scaler(train_full.dropna(subset=feature_columns), feature_columns)
    scaled = apply_feature_scaler(frame, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled,
        feature_columns,
        args.target_column,
        args.window_size,
        extra_meta_columns=("__tn__",),
        sequence_normalization="none",
    )
    pressure_gate = pressure_gate_by_date(frame)
    return frame, x_all, y_all.astype(np.float32), meta_all.reset_index(drop=True), pressure_gate


def select_by_dates(meta: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    date_values = meta["Date"].values.astype("datetime64[D]")
    target_values = dates.values.astype("datetime64[D]")
    return np.isin(date_values, target_values)


def candidate_signals(
    mu_train: np.ndarray,
    sigma_train: np.ndarray,
    mu_test: np.ndarray,
    sigma_test: np.ndarray,
    meta_test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    cr_train = np.abs(mu_train) / np.maximum(sigma_train, 1e-8)
    cr_test = np.abs(mu_test) / np.maximum(sigma_test, 1e-8)
    cr_thr = {q: float(np.quantile(cr_train, q / 100.0)) for q in (60, 70, 80)}
    abs_mu_thr = {q: float(np.quantile(np.abs(mu_train), q / 100.0)) for q in (60, 70)}
    signals = {
        "full": np.abs(mu_test),
        "conf_ratio_q60": np.where(cr_test >= cr_thr[60], cr_test, 0.0),
        "conf_ratio_q70": np.where(cr_test >= cr_thr[70], cr_test, 0.0),
        "conf_ratio_q80": np.where(cr_test >= cr_thr[80], cr_test, 0.0),
        "abs_mu_q60": np.where(np.abs(mu_test) >= abs_mu_thr[60], np.abs(mu_test), 0.0),
        "abs_mu_q70": np.where(np.abs(mu_test) >= abs_mu_thr[70], np.abs(mu_test), 0.0),
    }
    tmp = pd.DataFrame({"Date": meta_test["Date"].values, "sigma": sigma_test})
    tmp.index = np.arange(len(tmp))
    low_sigma = np.zeros(len(tmp), dtype=bool)
    for _, group in tmp.groupby("Date", sort=False):
        threshold = group["sigma"].quantile(0.50)
        low_sigma[group.index[group["sigma"] <= threshold]] = True
    signals["daily_bot_sig_50pct"] = np.where(low_sigma, np.abs(mu_test), 0.0)
    return {key: value.astype(np.float32) for key, value in signals.items()}


def apply_gate(signal: np.ndarray, meta: pd.DataFrame, gate_name: str, pressure_gate: pd.Series) -> np.ndarray:
    if gate_name == "none":
        return signal
    if gate_name != "pressure_nonneg":
        raise ValueError(f"Unknown gate: {gate_name}")
    active = pd.Series(pd.to_datetime(meta["Date"])).map(pressure_gate).fillna(False).to_numpy(dtype=bool)
    return np.where(active, signal, 0.0).astype(np.float32)


def simulate_policy(
    actual: np.ndarray,
    signal: np.ndarray,
    meta: pd.DataFrame,
    policy: PolicyConfig,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(meta["Date"]).values,
            "code": meta["code"].astype(str).values,
            "actual": actual,
            "signal": signal,
        }
    )
    cost = cost_bps / 10_000.0
    current: dict[str, float] = {}
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for day_idx, (date, day) in enumerate(df.groupby("Date", sort=True)):
        rebalance = day_idx % policy.rebalance_every == 0
        if rebalance:
            eligible = day.loc[day["signal"].astype(float) > 0.0].copy()
            if len(eligible) >= policy.min_positions:
                top = eligible.nlargest(min(policy.top_k, len(eligible)), "signal")
                current = {str(code): 1.0 / len(top) for code in top["code"]}
            else:
                current = {}
        returns = dict(zip(day["code"].astype(str), day["actual"].astype(float)))
        gross = float(sum(weight * returns.get(code, 0.0) for code, weight in current.items()))
        turnover = 0.0
        if rebalance:
            keys = set(current).union(previous)
            turnover = float(sum(abs(current.get(key, 0.0) - previous.get(key, 0.0)) for key in keys))
        net = gross - cost * turnover
        rows.append(
            {
                "Date": date,
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
                "n_positions": len(current),
                "is_rebalance": rebalance,
            }
        )
        previous = dict(current)
    return pd.DataFrame(rows)


def summarize_policy(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {}
    net = daily["net_return"].astype(float)
    equity = (1.0 + net).cumprod()
    tmp = daily.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp["quarter"] = tmp["Date"].dt.to_period("Q")
    quarter_returns = tmp.groupby("quarter")["net_return"].sum()
    return {
        "final_equity": float(equity.iloc[-1]),
        "sharpe": sharpe(net),
        "max_dd": max_drawdown(equity),
        "mean_turnover": float(daily["turnover"].mean()),
        "hit_rate": float((net > 0.0).mean()),
        "mean_positions": float(daily["n_positions"].mean()),
        "active_days": float((daily["n_positions"] > 0).mean()),
        "worst_quarter": float(quarter_returns.min()) if len(quarter_returns) else float("nan"),
        "n_days": int(len(daily)),
    }


def default_policy_grid() -> list[PolicyConfig]:
    configs: list[PolicyConfig] = []
    for rule in ("conf_ratio_q70", "conf_ratio_q80", "daily_bot_sig_50pct", "full", "abs_mu_q60"):
        for gate in ("pressure_nonneg",):
            for rebalance in (10, 20):
                for top_k in (5, 8, 10, 15, 20):
                    for min_positions in (1, 3, 5, 6):
                        name = f"{rule}_{gate}_r{rebalance}_k{top_k}_m{min_positions}"
                        configs.append(PolicyConfig(name, rule, gate, rebalance, top_k, min_positions))
    # Add no-gate references for a small subset
    for rule in ("conf_ratio_q70", "daily_bot_sig_50pct", "full"):
        for rebalance in (10, 20):
            name = f"{rule}_none_r{rebalance}_k10_m1"
            configs.append(PolicyConfig(name, rule, "none", rebalance, 10, 1))
    return configs


def train_predict_fold(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    meta_train: pd.DataFrame,
    x_test: np.ndarray,
    y_test_raw: np.ndarray,
    meta_test: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_scale = meta_train["__tn__"].to_numpy(dtype=np.float32)
    test_scale = meta_test["__tn__"].to_numpy(dtype=np.float32)
    local_norm = fit_local_target_normalizer(train_scale, args.target_normalizer)
    y_train_local = apply_local_target_normalizer(y_train_raw, train_scale, local_norm)
    target_scaler = fit_target_scaler(y_train_local)
    y_train_scaled = apply_target_scaler(y_train_local, target_scaler).reshape(-1, 1)
    y_train_model = np.concatenate([y_train_scaled, train_scale.reshape(-1, 1)], axis=1).astype(np.float32)

    set_global_seed(seed)
    model = build_model(
        x_train.shape[2],
        target_scaler,
        local_norm,
        window_size=args.window_size,
        lstm_units=[int(part) for part in args.lstm_units.split(",")],
        dropout=args.dropout,
        lr=args.lr,
        w_rel=args.w_rel,
        w_nll=args.w_nll,
    )
    callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=args.patience, restore_best_weights=True)]
    model.fit(x_train, y_train_model, epochs=args.epochs, batch_size=args.batch_size, verbose=0, callbacks=callbacks)
    mu_train, sigma_train = predict_raw(model, x_train, train_scale, target_scaler, local_norm)
    mu_test, sigma_test = predict_raw(model, x_test, test_scale, target_scaler, local_norm)
    return mu_train, sigma_train, mu_test, sigma_test


def run_batch(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    frame, x_all, y_all, meta_all, pressure_gate = build_sequence_inputs(args)
    folds = make_fold_dates(meta_all, max_folds=args.max_folds)
    seeds = parse_seeds(args.seeds)
    policies = default_policy_grid()
    print(f"Sequences={len(x_all)} folds={len(folds)} seeds={seeds} policies={len(policies)}")

    fold_rows: list[dict[str, object]] = []
    daily_rows: list[pd.DataFrame] = []
    for seed in seeds:
        for fold_idx, (train_dates, test_dates) in enumerate(folds):
            fold_label = f"seed{seed}_fold{fold_idx:03d}"
            pred_path = pred_dir / f"{fold_label}.npz"
            train_mask = select_by_dates(meta_all, train_dates)
            test_mask = select_by_dates(meta_all, test_dates)
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            meta_train = meta_all.loc[train_mask].reset_index(drop=True)
            meta_test = meta_all.loc[test_mask].reset_index(drop=True)
            if pred_path.exists() and not args.force:
                cached = np.load(pred_path)
                mu_train = cached["mu_train"]
                sigma_train = cached["sigma_train"]
                mu_test = cached["mu_test"]
                sigma_test = cached["sigma_test"]
            else:
                print(f"Training {fold_label} test={test_dates[0].date()}..{test_dates[-1].date()} n_train={train_mask.sum()} n_test={test_mask.sum()}")
                mu_train, sigma_train, mu_test, sigma_test = train_predict_fold(
                    args,
                    x_all[train_mask],
                    y_all[train_mask],
                    meta_train,
                    x_all[test_mask],
                    y_all[test_mask],
                    meta_test,
                    seed,
                )
                np.savez_compressed(
                    pred_path,
                    mu_train=mu_train,
                    sigma_train=sigma_train,
                    mu_test=mu_test,
                    sigma_test=sigma_test,
                    y_train=y_all[train_mask],
                    y_test=y_all[test_mask],
                )
            signals = candidate_signals(mu_train, sigma_train, mu_test, sigma_test, meta_test)
            for policy in policies:
                if policy.rule not in signals:
                    continue
                signal = apply_gate(signals[policy.rule], meta_test, policy.gate, pressure_gate)
                daily = simulate_policy(y_all[test_mask], signal, meta_test, policy, cost_bps=args.cost_bps)
                summary = summarize_policy(daily)
                if not summary:
                    continue
                rel_mask = signal > 0.0
                fold_row = {
                    "seed": seed,
                    "fold_id": fold_idx,
                    "test_start": str(test_dates[0].date()),
                    "test_end": str(test_dates[-1].date()),
                    "policy": policy.name,
                    "rule": policy.rule,
                    "gate": policy.gate,
                    "rebalance_every": policy.rebalance_every,
                    "top_k": policy.top_k,
                    "min_positions": policy.min_positions,
                    "coverage": float(rel_mask.mean()),
                    "rel_score_selected": rel_score_fn(y_all[test_mask][rel_mask], mu_test[rel_mask]) if rel_mask.any() else float("nan"),
                    **summary,
                }
                fold_rows.append(fold_row)
                daily_copy = daily.copy()
                daily_copy["seed"] = seed
                daily_copy["fold_id"] = fold_idx
                daily_copy["policy"] = policy.name
                daily_rows.append(daily_copy)
            # Write incremental after each fold for resume safety
            pd.DataFrame(fold_rows).to_csv(args.output_dir / "per_fold_policy_metrics.csv", index=False)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(args.output_dir / "per_fold_policy_metrics.csv", index=False)
    if daily_rows:
        daily_df = pd.concat(daily_rows, ignore_index=True)
        daily_df.to_csv(args.output_dir / "daily_policy_returns.csv", index=False)
    else:
        daily_df = pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    for policy, group in fold_df.groupby("policy", sort=False):
        row = {
            "policy": policy,
            "rule": group["rule"].iloc[0],
            "gate": group["gate"].iloc[0],
            "rebalance_every": int(group["rebalance_every"].iloc[0]),
            "top_k": int(group["top_k"].iloc[0]),
            "min_positions": int(group["min_positions"].iloc[0]),
            "n_folds": int(len(group)),
            "positive_equity_folds": int((group["final_equity"] > 1.0).sum()),
            "positive_rel_folds": int((group["rel_score_selected"] > 0.0).sum()),
            "mean_fold_equity": float(group["final_equity"].mean()),
            "worst_fold_equity": float(group["final_equity"].min()),
            "mean_sharpe": float(group["sharpe"].mean()),
            "min_sharpe": float(group["sharpe"].min()),
            "mean_max_dd": float(group["max_dd"].mean()),
            "worst_max_dd": float(group["max_dd"].min()),
            "mean_turnover": float(group["mean_turnover"].mean()),
            "max_turnover": float(group["mean_turnover"].max()),
            "mean_positions": float(group["mean_positions"].mean()),
            "mean_active_days": float(group["active_days"].mean()),
            "mean_rel_score_selected": float(group["rel_score_selected"].mean()),
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["mean_sharpe", "worst_fold_equity"], ascending=[False, False])
    summary_df.to_csv(args.output_dir / "policy_summary.csv", index=False)
    summary_df.to_csv(args.gold_dir / "policy_summary.csv", index=False)

    gate_df = summary_df[
        (summary_df["mean_sharpe"] > args.gate_min_mean_sharpe)
        & (summary_df["worst_fold_equity"] > args.gate_min_worst_equity)
        & (summary_df["worst_max_dd"] > -abs(args.gate_max_drawdown))
        & (summary_df["max_turnover"] < args.gate_max_turnover)
        & (summary_df["mean_positions"] >= args.gate_min_positions)
    ].copy()
    gate_df.to_csv(args.output_dir / "gated_policy_summary.csv", index=False)
    gate_df.to_csv(args.gold_dir / "gated_policy_summary.csv", index=False)

    key_cols = [
        "policy", "rule", "gate", "rebalance_every", "top_k", "min_positions",
        "mean_fold_equity", "worst_fold_equity", "mean_sharpe", "min_sharpe",
        "worst_max_dd", "max_turnover", "mean_positions", "mean_rel_score_selected",
    ]
    key_cols = [col for col in key_cols if col in summary_df.columns]
    lines = [
        "# Hetero Long Fine-Tune Batch Summary",
        "",
        f"Seeds: {seeds}; folds: {len(folds)}; epochs={args.epochs}; patience={args.patience}.",
        "Scope: VN validation only. Holdout/test not used.",
        "",
        "## Top 20 policies by mean Sharpe",
        "",
        summary_df.head(20)[key_cols].round(4).to_markdown(index=False) if not summary_df.empty else "No results.",
        "",
        "## Gated policies",
        "",
        gate_df.head(30)[key_cols].round(4).to_markdown(index=False) if not gate_df.empty else "None passed gate.",
        "",
        "## Gate criteria",
        f"- mean_sharpe > {args.gate_min_mean_sharpe}",
        f"- worst_fold_equity > {args.gate_min_worst_equity}",
        f"- worst_max_dd > -{args.gate_max_drawdown}",
        f"- max_turnover < {args.gate_max_turnover}",
        f"- mean_positions >= {args.gate_min_positions}",
    ]
    summary_text = "\n".join(lines)
    (args.output_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    print(summary_text[:4000])
    print(json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seeds", default="43,52,62")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-column", default="target_next_return")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--w-rel", type=float, default=0.7)
    parser.add_argument("--w-nll", type=float, default=0.3)
    parser.add_argument("--cost-bps", type=float, default=COST_BPS)
    parser.add_argument("--gate-min-mean-sharpe", type=float, default=0.5)
    parser.add_argument("--gate-min-worst-equity", type=float, default=1.0)
    parser.add_argument("--gate-max-drawdown", type=float, default=0.25)
    parser.add_argument("--gate-max-turnover", type=float, default=0.20)
    parser.add_argument("--gate-min-positions", type=float, default=5.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_batch(parse_args())
