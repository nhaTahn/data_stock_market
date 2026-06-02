"""Evaluate stop-cash overlay on existing daily policy returns.

Rule: if trailing N-day cumulative net return <= -threshold, go cash for K days.
This is an execution overlay, not model retraining. It uses only past portfolio
returns inside each seed/fold/policy path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_long_finetune_batch_20260522/daily_policy_returns.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_stop_cash_overlay_20260524"
DEFAULT_GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/hetero_stop_cash_overlay_20260524"

FOCUS_POLICIES = [
    "daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5",
    "daily_bot_sig_50pct_pressure_nonneg_r20_k20_m6",
    "abs_mu_q60_pressure_nonneg_r20_k20_m6",
    "conf_ratio_q70_pressure_nonneg_r20_k20_m6",
]


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak.replace(0, np.nan) - 1).min()) if len(equity) else float("nan")


def sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    std = float(returns.std(ddof=1))
    return float(returns.mean() / std * np.sqrt(252)) if std > 0 else float("nan")


def apply_stop_cash(path: pd.DataFrame, lookback: int, threshold: float, cooldown: int) -> pd.DataFrame:
    path = path.sort_values("Date").copy()
    raw = path["net_return"].astype(float).to_numpy()
    out = raw.copy()
    cash_left = 0
    trigger = np.zeros(len(raw), dtype=bool)
    for i in range(len(raw)):
        if cash_left > 0:
            out[i] = 0.0
            cash_left -= 1
            continue
        if i >= lookback:
            trailing = float(np.prod(1.0 + out[i - lookback : i]) - 1.0)
            if trailing <= -abs(threshold):
                out[i] = 0.0
                cash_left = max(0, cooldown - 1)
                trigger[i] = True
    path["net_return_stop"] = out
    path["stop_trigger"] = trigger
    return path


def summarize_path(path: pd.DataFrame, return_col: str) -> dict[str, float]:
    ret = path[return_col].astype(float)
    eq = (1.0 + ret).cumprod()
    return {
        "final_equity": float(eq.iloc[-1]) if len(eq) else float("nan"),
        "sharpe": sharpe(ret),
        "max_dd": max_drawdown(eq),
        "hit_rate": float((ret > 0).mean()) if len(ret) else float("nan"),
        "n_days": int(len(ret)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(args.input)
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily[daily["policy"].isin(FOCUS_POLICIES)].copy()

    grid = [(10, 0.08, 5), (10, 0.10, 5), (14, 0.08, 5), (14, 0.10, 5), (14, 0.10, 10), (21, 0.12, 10)]
    rows: list[dict[str, object]] = []
    adjusted_parts: list[pd.DataFrame] = []

    for policy in FOCUS_POLICIES:
        base = daily[daily["policy"] == policy]
        for lookback, threshold, cooldown in grid:
            per_path_rows = []
            adjusted = []
            for (seed, fold_id), path in base.groupby(["seed", "fold_id"], sort=False):
                adj = apply_stop_cash(path, lookback, threshold, cooldown)
                adjusted.append(adj.assign(lookback=lookback, threshold=threshold, cooldown=cooldown))
                raw_sum = summarize_path(path, "net_return")
                stop_sum = summarize_path(adj, "net_return_stop")
                per_path_rows.append({
                    "seed": seed,
                    "fold_id": fold_id,
                    "raw_equity": raw_sum["final_equity"],
                    "stop_equity": stop_sum["final_equity"],
                    "raw_sharpe": raw_sum["sharpe"],
                    "stop_sharpe": stop_sum["sharpe"],
                    "raw_max_dd": raw_sum["max_dd"],
                    "stop_max_dd": stop_sum["max_dd"],
                    "triggers": int(adj["stop_trigger"].sum()),
                    "cash_days": int((adj["net_return_stop"] == 0.0).sum() - (path["net_return"].astype(float) == 0.0).sum()),
                })
            pr = pd.DataFrame(per_path_rows)
            rows.append({
                "policy": policy,
                "lookback": lookback,
                "threshold": threshold,
                "cooldown": cooldown,
                "mean_raw_equity": float(pr["raw_equity"].mean()),
                "mean_stop_equity": float(pr["stop_equity"].mean()),
                "worst_raw_equity": float(pr["raw_equity"].min()),
                "worst_stop_equity": float(pr["stop_equity"].min()),
                "mean_raw_sharpe": float(pr["raw_sharpe"].mean()),
                "mean_stop_sharpe": float(pr["stop_sharpe"].mean()),
                "min_raw_sharpe": float(pr["raw_sharpe"].min()),
                "min_stop_sharpe": float(pr["stop_sharpe"].min()),
                "worst_raw_max_dd": float(pr["raw_max_dd"].min()),
                "worst_stop_max_dd": float(pr["stop_max_dd"].min()),
                "mean_triggers": float(pr["triggers"].mean()),
                "mean_cash_days": float(pr["cash_days"].mean()),
            })
            adjusted_parts.extend(adjusted)

    result = pd.DataFrame(rows).sort_values(["worst_stop_equity", "mean_stop_sharpe"], ascending=[False, False])
    result.to_csv(args.output_dir / "stop_cash_summary.csv", index=False)
    result.to_csv(args.gold_dir / "stop_cash_summary.csv", index=False)
    if adjusted_parts:
        pd.concat(adjusted_parts, ignore_index=True).to_csv(args.output_dir / "adjusted_daily_returns.csv", index=False)

    cols = ["policy", "lookback", "threshold", "cooldown", "mean_raw_equity", "mean_stop_equity", "worst_raw_equity", "worst_stop_equity", "mean_raw_sharpe", "mean_stop_sharpe", "worst_raw_max_dd", "worst_stop_max_dd", "mean_triggers", "mean_cash_days"]
    text = "\n".join([
        "# Stop-Cash Overlay Results",
        "",
        "Rule uses only trailing portfolio returns within each seed/fold path.",
        "",
        result[cols].head(30).round(4).to_markdown(index=False),
    ])
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text[:4000])


if __name__ == "__main__":
    main()
