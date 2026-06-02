"""Holding-period / turnover-constrained selector grid.

Uses saved predictions from hetero_combined_full5 (5 seeds).
No retraining needed — pure offline simulation.

Grid:
  - selection rule: full, conf_ratio_q70, conf_ratio_q60, abs_mu_q60, daily_bot_sig_50pct
  - rebalance_every: 1, 3, 5, 10, 20  (days between portfolio updates)
  - top_k: 8, 10, 15, 20
  - cost_bps: 15 (fixed for now)

For each (rule, rebalance, top_k) combination:
  - Simulate on val period (2020-04-01 → 2022-11-15) using a single portfolio
  - Compute: final equity, Sharpe, MaxDD, turnover, hit-rate, quarterly worst

Decision criteria (must pass ALL):
  - Sharpe > 0.5
  - MaxDD > -25%
  - mean_turnover < 0.35
  - final equity > 1.2

Outputs: grid_summary.csv, summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRED_DIR = (
    ROOT
    / "data/processed/assets/data_info_vn/history/training_runs"
    / "reports/hetero_combined_full5_20260521"
)
OUTPUT_DIR = (
    ROOT
    / "data/processed/assets/data_info_vn/history/training_runs"
    / "reports/hetero_holding_period_grid_20260521"
)
GOLD_DIR = (
    ROOT
    / "gold/vn_transition_pressure_20260512/plots"
    / "hetero_holding_period_grid_20260521"
)

COST_BPS = 15
TOP_K_GRID = [8, 10, 15, 20]
REBALANCE_GRID = [1, 3, 5, 10, 20]
MIN_POSITIONS = 5
SEEDS = [43, 52, 62, 71, 82]

from src.models.config import DEFAULT_FEATURE_COLUMNS
from src.models.training.datasets import (
    build_sequence_dataset,
    split_frame_by_date,
)
from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler
from src.models.training.pipeline import load_frame as load_training_frame


def build_val_meta() -> tuple[np.ndarray, pd.DataFrame]:
    data_path = (
        ROOT
        / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
    )
    feature_columns = DEFAULT_FEATURE_COLUMNS
    frame = load_training_frame(data_path, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["__tn__"] = frame["volatility_20"].astype(float)
    train_df, _, _ = split_frame_by_date(frame, "2020-03-31", "2022-11-15")
    scaler = fit_feature_scaler(
        train_df.dropna(subset=feature_columns), feature_columns
    )
    scaled = apply_feature_scaler(frame, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled,
        feature_columns,
        "target_next_return",
        15,
        extra_meta_columns=("__tn__",),
        sequence_normalization="none",
    )
    # val split
    val_mask = (meta_all["Date"] >= "2020-04-01") & (
        meta_all["Date"] <= "2022-11-15"
    )
    y_val = y_all[val_mask]
    meta_val = meta_all[val_mask].reset_index(drop=True)
    return y_val.astype(np.float32), meta_val


def apply_train_thresholds(
    mu_train: np.ndarray,
    sigma_train: np.ndarray,
    mu_val: np.ndarray,
    sigma_val: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return dict of rule → conf_ratio or signal score arrays for val."""
    cr_train = np.abs(mu_train) / np.maximum(sigma_train, 1e-8)
    cr_val = np.abs(mu_val) / np.maximum(sigma_val, 1e-8)

    thresholds = {
        q: float(np.quantile(cr_train, q / 100))
        for q in [50, 60, 70, 80]
    }
    abs_mu_thr = {
        q: float(np.quantile(np.abs(mu_train), q / 100))
        for q in [50, 60, 70]
    }

    # Each rule returns a float signal array (positive = include, magnitude = ranking signal)
    # Zero means exclude that stock-day from portfolio consideration
    signals: dict[str, np.ndarray] = {
        "full": np.abs(mu_val).astype(np.float32),
        "conf_ratio_q60": np.where(
            cr_val >= thresholds[60], cr_val, 0.0
        ).astype(np.float32),
        "conf_ratio_q70": np.where(
            cr_val >= thresholds[70], cr_val, 0.0
        ).astype(np.float32),
        "conf_ratio_q80": np.where(
            cr_val >= thresholds[80], cr_val, 0.0
        ).astype(np.float32),
        "abs_mu_q60": np.where(
            np.abs(mu_val) >= abs_mu_thr[60], np.abs(mu_val), 0.0
        ).astype(np.float32),
        "abs_mu_q70": np.where(
            np.abs(mu_val) >= abs_mu_thr[70], np.abs(mu_val), 0.0
        ).astype(np.float32),
    }
    return signals


def simulate_portfolio(
    actual: np.ndarray,
    signal: np.ndarray,
    dates: np.ndarray,
    codes: np.ndarray,
    *,
    rebalance_every: int,
    top_k: int,
    cost_bps: float = COST_BPS,
    min_positions: int = MIN_POSITIONS,
) -> pd.DataFrame:
    """Walk-forward portfolio with fixed rebalance period.

    On rebalance days: pick top_k stocks by signal (must be > 0), equal weight.
    On non-rebalance days: hold previous portfolio.
    """
    df = pd.DataFrame(
        {
            "date": dates,
            "code": codes,
            "actual": actual,
            "signal": signal,
        }
    )
    cost = cost_bps / 10_000.0
    prev_held: dict[str, float] = {}
    current_held: dict[str, float] = {}
    rows = []
    day_idx = 0
    for date, grp in df.groupby("date", sort=True):
        is_rebalance = day_idx % rebalance_every == 0
        if is_rebalance:
            eligible = grp[grp["signal"] > 0]
            if len(eligible) >= min_positions:
                top = eligible.nlargest(min(top_k, len(eligible)), "signal")
                current_held = {
                    str(r["code"]): 1.0 / len(top) for _, r in top.iterrows()
                }
            else:
                current_held = {}
        returns = dict(
            zip(grp["code"].astype(str), grp["actual"].astype(float))
        )
        gross = float(
            sum(w * returns.get(c, 0.0) for c, w in current_held.items())
        )
        turnover = float(
            sum(
                abs(current_held.get(c, 0.0) - prev_held.get(c, 0.0))
                for c in set(current_held) | set(prev_held)
            )
        ) if is_rebalance else 0.0
        net = gross - turnover * cost
        rows.append(
            {
                "date": date,
                "gross": gross,
                "net": net,
                "turnover": turnover,
                "n_pos": len(current_held),
                "is_rebalance": is_rebalance,
            }
        )
        prev_held = dict(current_held)
        day_idx += 1

    return pd.DataFrame(rows)


def summarize(sim: pd.DataFrame) -> dict[str, float]:
    if sim.empty or sim["net"].isna().all():
        return {}
    cum = (1 + sim["net"]).cumprod()
    peak = cum.cummax()
    dd = (cum / peak.replace(0, np.nan) - 1).min()
    sharpe = float(
        sim["net"].mean() / max(sim["net"].std(), 1e-8) * np.sqrt(252)
    )
    # worst quarter
    sim2 = sim.copy()
    sim2["date"] = pd.to_datetime(sim2["date"])
    sim2["quarter"] = sim2["date"].dt.to_period("Q")
    q_ret = sim2.groupby("quarter")["net"].sum()
    return {
        "final_equity": float(cum.iloc[-1]),
        "ann_ret": float(cum.iloc[-1] ** (252 / len(cum)) - 1),
        "sharpe": sharpe,
        "max_dd": float(dd),
        "mean_turnover": float(sim["turnover"].mean()),
        "hit_rate": float((sim["net"] > 0).mean()),
        "worst_quarter": float(q_ret.min()),
        "best_quarter": float(q_ret.max()),
        "n_days": len(sim),
        "mean_positions": float(sim["n_pos"].mean()),
    }


def load_ensemble_predictions(seeds: list[int]) -> tuple[np.ndarray, ...]:
    """Return per-seed stacked predictions; also return ensemble mean."""
    all_mu_tr, all_sig_tr, all_mu_v, all_sig_v, all_y_v = [], [], [], [], []
    for seed in seeds:
        npz = np.load(PRED_DIR / f"predictions_seed_{seed}.npz")
        all_mu_tr.append(npz["mu_train"])
        all_sig_tr.append(npz["sigma_train"])
        all_mu_v.append(npz["mu_val"])
        all_sig_v.append(npz["sigma_val"])
        all_y_v.append(npz["y_val"])
    # Ensemble by mean
    mu_tr_ens = np.mean(all_mu_tr, axis=0)
    sig_tr_ens = np.mean(all_sig_tr, axis=0)
    mu_v_ens = np.mean(all_mu_v, axis=0)
    sig_v_ens = np.mean(all_sig_v, axis=0)
    y_v_ref = all_y_v[0]
    return mu_tr_ens, sig_tr_ens, mu_v_ens, sig_v_ens, y_v_ref


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading val meta...")
    y_val, meta_val = build_val_meta()
    dates_val = meta_val["Date"].values
    codes_val = meta_val["code"].values if "code" in meta_val.columns else np.arange(len(y_val))
    print(f"Val: {len(y_val)} samples, {len(np.unique(dates_val))} days")

    print("Loading ensemble predictions (5 seeds)...")
    mu_tr, sig_tr, mu_v, sig_v, y_v = load_ensemble_predictions(SEEDS)
    assert len(y_v) == len(y_val)

    print("Computing selection signals...")
    signals = apply_train_thresholds(mu_tr, sig_tr, mu_v, sig_v)

    # Also include per-day bottom-sigma rule
    df_tmp = pd.DataFrame({"date": dates_val, "sigma": sig_v})
    df_tmp.index = np.arange(len(df_tmp))
    mask_arr = np.zeros(len(mu_v), dtype=bool)
    for _, grp in df_tmp.groupby("date"):
        thresh = grp["sigma"].quantile(0.50)
        mask_arr[grp.index[grp["sigma"] <= thresh]] = True
    signals["daily_bot_sig_50pct"] = (
        np.where(mask_arr, np.abs(mu_v), 0.0).astype(np.float32)
    )

    # Run grid
    grid_rows = []
    combos = [
        (rule, reb, topk)
        for rule in signals
        for reb in REBALANCE_GRID
        for topk in TOP_K_GRID
    ]
    print(f"Running {len(combos)} grid combinations...")
    for i, (rule, reb, topk) in enumerate(combos):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(combos)}", end="\r")
        sig = signals[rule]
        sim = simulate_portfolio(
            y_v, sig, dates_val, codes_val,
            rebalance_every=reb, top_k=topk,
        )
        s = summarize(sim)
        if s:
            s.update(
                {"rule": rule, "rebalance_every": reb, "top_k": topk}
            )
            grid_rows.append(s)

    print()
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUTPUT_DIR / "grid_summary.csv", index=False)
    grid_df.to_csv(GOLD_DIR / "grid_summary.csv", index=False)

    # Gate: all 4 criteria
    gated = grid_df[
        (grid_df["sharpe"] > 0.5)
        & (grid_df["max_dd"] > -0.25)
        & (grid_df["mean_turnover"] < 0.35)
        & (grid_df["final_equity"] > 1.2)
    ].sort_values("sharpe", ascending=False)
    gated.to_csv(OUTPUT_DIR / "gated_candidates.csv", index=False)
    gated.to_csv(GOLD_DIR / "gated_candidates.csv", index=False)

    # Find Pareto-optimal: best Sharpe at each turnover bucket
    grid_df["to_bucket"] = pd.cut(
        grid_df["mean_turnover"],
        bins=[0, 0.10, 0.20, 0.30, 0.50, 1.0, 10.0],
        labels=["≤0.10", "0.10-0.20", "0.20-0.30", "0.30-0.50", "0.50-1.0", ">1.0"],
    )
    pareto = (
        grid_df.sort_values("sharpe", ascending=False)
        .groupby("to_bucket", observed=True)
        .first()
        .reset_index()
    )

    cols_show = [
        "rule", "rebalance_every", "top_k",
        "final_equity", "ann_ret", "sharpe", "max_dd",
        "mean_turnover", "worst_quarter", "mean_positions",
    ]
    cols_show = [c for c in cols_show if c in grid_df.columns]

    # Summary markdown
    lines = [
        "# Holding-Period Grid Results",
        "",
        f"Grid: {len(signals)} selection rules × {len(REBALANCE_GRID)} rebalance periods × "
        f"{len(TOP_K_GRID)} top_k = {len(combos)} combinations.",
        f"Cost: {COST_BPS}bps. Ensemble of 5 seeds (43,52,62,71,82).",
        "Scope: VN validation only. Holdout/test not used.",
        "",
        "## Top 15 by Sharpe (all combos)",
        "",
        grid_df.sort_values("sharpe", ascending=False).head(15)[cols_show].round(4).to_markdown(index=False),
        "",
        f"## Gated candidates (Sharpe>0.5, MaxDD>-25%, Turnover<0.35, Equity>1.2)",
        f"Count: {len(gated)}",
        "",
        gated.head(20)[cols_show].round(4).to_markdown(index=False) if len(gated) else "None passed gate.",
        "",
        "## Best per turnover bucket",
        "",
        pareto[["to_bucket"] + cols_show].round(4).to_markdown(index=False),
        "",
        "## Reference baselines",
        "- stressaux_w20 (production): Sharpe ~0.59, MaxDD -24.3%, Equity 1.186–1.679",
        "- hetero_combined raw (daily rebalance): Sharpe 0.88–0.90, MaxDD -41 to -44%, Turnover ~1.0",
    ]
    text = "\n".join(lines)
    (OUTPUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    (GOLD_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text[:3000])
    print(f"\n... (full at {OUTPUT_DIR / 'summary.md'})")
    print(
        json.dumps(
            {"output_dir": str(OUTPUT_DIR), "gold_dir": str(GOLD_DIR)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
