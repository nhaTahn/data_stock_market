"""Evaluate stricter pre-crash market gates on cached policy daily returns.

Uses validation-only rolling artifacts; no holdout/test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402

INPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_long_finetune_batch_20260522/daily_policy_returns.csv"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_market_gate_overlays_20260524"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/hetero_market_gate_overlays_20260524"

FOCUS = [
    "daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5",
    "daily_bot_sig_50pct_pressure_nonneg_r20_k20_m6",
    "conf_ratio_q70_pressure_nonneg_r20_k20_m5",
    "conf_ratio_q70_pressure_nonneg_r20_k20_m6",
    "abs_mu_q60_pressure_nonneg_r20_k20_m6",
    "full_pressure_nonneg_r20_k20_m6",
]


def sharpe(ret: pd.Series) -> float:
    std = ret.std(ddof=1)
    return float(ret.mean() / std * np.sqrt(252)) if std and std > 0 else float("nan")


def max_dd(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")


def build_gates() -> pd.DataFrame:
    frame = load_frame()
    daily = frame.groupby("Date").agg(
        vnindex_return=("vnindex_return", "mean"),
        buying_pressure=("buying_pressure", "mean"),
        selling_pressure=("selling_pressure", "mean"),
        market_ad_ratio_20=("market_ad_ratio_20", "mean"),
        wyckoff_phase_60d=("wyckoff_phase_60d", "mean"),
    ).sort_index()
    daily["pressure_delta_20"] = (daily["buying_pressure"] - daily["selling_pressure"]).rolling(20, min_periods=5).mean()
    daily["idx_ret_5"] = daily["vnindex_return"].rolling(5, min_periods=3).sum()
    daily["idx_ret_10"] = daily["vnindex_return"].rolling(10, min_periods=5).sum()
    daily["sell_pressure_5"] = daily["selling_pressure"].rolling(5, min_periods=3).mean()

    daily["pressure_only"] = daily["pressure_delta_20"] >= 0
    daily["wyck035"] = daily["pressure_only"] & (daily["wyckoff_phase_60d"] >= 0.35)
    daily["wyck040"] = daily["pressure_only"] & (daily["wyckoff_phase_60d"] >= 0.40)
    daily["wyck035_idx5"] = daily["wyck035"] & (daily["idx_ret_5"] > -0.04)
    daily["wyck035_idx10"] = daily["wyck035"] & (daily["idx_ret_10"] > -0.06)
    daily["wyck035_sell"] = daily["wyck035"] & (daily["sell_pressure_5"] < 0.38)
    daily["wyck035_ad"] = daily["wyck035"] & (daily["market_ad_ratio_20"] > 1.3)
    daily["ultra_safe"] = daily["wyck035"] & (daily["idx_ret_5"] > -0.03) & (daily["sell_pressure_5"] < 0.35)
    return daily


def load_frame() -> pd.DataFrame:
    frame = load_frame.cache
    if frame is not None:
        return frame
    f = load_frame.cache = load_frame_fn()
    return f


def load_frame_fn() -> pd.DataFrame:
    path = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
    frame = load_frame_module(path, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame

load_frame.cache = None
from src.models.training.pipeline import load_frame as load_frame_module  # noqa: E402


def summarize_path(df: pd.DataFrame, ret_col: str) -> dict[str, float]:
    ret = df[ret_col].astype(float)
    eq = (1 + ret).cumprod()
    return {
        "final_equity": float(eq.iloc[-1]) if len(eq) else float("nan"),
        "ann_ret": float(eq.iloc[-1] ** (252 / len(eq)) - 1) if len(eq) else float("nan"),
        "sharpe": sharpe(ret),
        "max_dd": max_dd(ret),
        "hit_rate": float((ret > 0).mean()) if len(ret) else float("nan"),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    gates = build_gates()
    gate_cols = ["pressure_only", "wyck035", "wyck040", "wyck035_idx5", "wyck035_idx10", "wyck035_sell", "wyck035_ad", "ultra_safe"]
    active = {col: float(gates[col].loc[(gates.index >= "2020-04-01") & (gates.index <= "2022-11-15")].mean()) for col in gate_cols}

    daily = pd.read_csv(INPUT)
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily[daily["policy"].isin(FOCUS)].copy()

    rows = []
    fold_rows = []
    for policy in FOCUS:
        base = daily[daily["policy"] == policy].copy()
        for gate in gate_cols:
            gate_map = gates[gate].astype(int)
            df = base.copy()
            df["gate_active"] = df["Date"].map(gate_map).fillna(0).astype(int)
            df["ret_gated"] = df["net_return"] * df["gate_active"]
            for seed, seed_df in df.groupby("seed"):
                s = summarize_path(seed_df.sort_values("Date"), "ret_gated")
                rows.append({"policy": policy, "gate": gate, "seed": seed, "active_days": float(seed_df["gate_active"].mean()), **s})
            for (seed, fold_id), fold_df in df.groupby(["seed", "fold_id"]):
                fs = summarize_path(fold_df.sort_values("Date"), "ret_gated")
                fold_rows.append({"policy": policy, "gate": gate, "seed": seed, "fold_id": fold_id, **fs})

    seed_df = pd.DataFrame(rows)
    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for (policy, gate), group in seed_df.groupby(["policy", "gate"]):
        folds = fold_df[(fold_df["policy"] == policy) & (fold_df["gate"] == gate)]
        summary_rows.append({
            "policy": policy,
            "gate": gate,
            "gate_active_global": active[gate],
            "mean_equity": float(group["final_equity"].mean()),
            "min_equity": float(group["final_equity"].min()),
            "mean_ann_ret": float(group["ann_ret"].mean()),
            "mean_sharpe": float(group["sharpe"].mean()),
            "min_sharpe": float(group["sharpe"].min()),
            "worst_max_dd": float(group["max_dd"].min()),
            "mean_fold_equity": float(folds["final_equity"].mean()),
            "worst_fold_equity": float(folds["final_equity"].min()),
            "positive_folds": int((folds["final_equity"] > 1.0).sum()),
            "n_folds": int(len(folds)),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["worst_fold_equity", "mean_sharpe"], ascending=[False, False])
    summary.to_csv(OUTPUT / "gate_overlay_summary.csv", index=False)
    summary.to_csv(GOLD / "gate_overlay_summary.csv", index=False)
    seed_df.to_csv(OUTPUT / "seed_gate_metrics.csv", index=False)
    fold_df.to_csv(OUTPUT / "fold_gate_metrics.csv", index=False)

    cols = ["policy", "gate", "gate_active_global", "mean_equity", "min_equity", "mean_ann_ret", "mean_sharpe", "worst_max_dd", "worst_fold_equity", "positive_folds", "n_folds"]
    text = "\n".join([
        "# Market Gate Overlay Summary",
        "",
        "Validation-only. Applied to cached rolling daily returns; holdout/test not used.",
        "",
        summary[cols].head(40).round(4).to_markdown(index=False),
    ])
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD / "summary.md").write_text(text, encoding="utf-8")
    print(text[:5000])


if __name__ == "__main__":
    main()
