"""Synthetic oracle leak test for backtest execution alignment.

Mục đích: verify rằng pipeline backtest KHÔNG cho phép prediction tại close `t`
được pair với realized return cùng ngày `t`. Bằng cách inject prediction =
actual_aligned (perfect oracle on the eval target), Sharpe phải đạt ∞ (hoặc
gần) — nếu hữu hạn vừa phải thì có execution leak.

Cách chạy:

```
python experiments/analysis/synthetic_oracle_leak_test.py \\
    --predictions-csv data/.../history/training_runs/<run>/predictions.csv \\
    --output reports/synthetic_oracle_leak/<date>_<run>.md
```

Test design:

1. Load prediction CSV chuẩn (có cột Date, code, prediction, actual, split).
2. Áp dụng `align_signal_actual` để tạo `signal_date / actual_date / actual_aligned`.
3. Inject `oracle_prediction = actual_aligned`.
4. Chạy `simulate_rebalance` với `prediction_column="oracle_prediction"`.
5. Report Sharpe, gross_equity, max_drawdown của oracle strategy.

Kỳ vọng:
- Sharpe ≫ 5 hoặc inf → alignment đúng, không có leak rõ rệt.
- Sharpe finite (vd 1-3) → cảnh báo: có thể đang leak một phần thông tin
  (vd prediction tại D[t] được pair với return D[t]→D[t+1] cùng ngày).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.selection.holding_period import (
    annualized_sharpe,
    max_drawdown,
    simulate_rebalance,
)


def align_signal_actual(frame: pd.DataFrame) -> pd.DataFrame:
    """Replicates analyze_lstm_filter_signal.align_signal_actual semantics."""
    parts: list[pd.DataFrame] = []
    grouped = frame.sort_values(["code", "split", "Date"], kind="stable").groupby(
        ["code", "split"], sort=False
    )
    for (_, _), group in grouped:
        if len(group) < 3:
            continue
        signal_rows = group.iloc[1:-1].reset_index(drop=True)
        actual_rows = group.iloc[2:].reset_index(drop=True)
        part = signal_rows.copy()
        part["actual_date"] = actual_rows["Date"].to_numpy()
        part["actual_aligned"] = actual_rows["actual"].to_numpy(dtype=float)
        parts.append(part)
    if not parts:
        raise RuntimeError("No alignable groups found in prediction frame.")
    return pd.concat(parts, ignore_index=True)


def run_oracle_backtest(
    aligned: pd.DataFrame,
    *,
    cost_bps: float,
    min_positions: int,
    rebalance_every: int,
) -> dict[str, float]:
    aligned = aligned.copy()
    aligned["oracle_prediction"] = aligned["actual_aligned"].astype(float)
    daily = simulate_rebalance(
        aligned,
        prediction_column="oracle_prediction",
        rebalance_every=rebalance_every,
        cost_bps=cost_bps,
        min_positions=min_positions,
    )
    gross_equity = (1.0 + daily["gross_return"]).cumprod()
    net_equity = (1.0 + daily["net_return"]).cumprod()
    return {
        "n_days": int(len(daily)),
        "avg_positions": float(daily["n_positions"].mean()),
        "avg_turnover": float(daily["turnover"].mean()),
        "gross_equity_final": float(gross_equity.iloc[-1]) if len(gross_equity) else float("nan"),
        "net_equity_final": float(net_equity.iloc[-1]) if len(net_equity) else float("nan"),
        "gross_sharpe": annualized_sharpe(daily["gross_return"]),
        "net_sharpe": annualized_sharpe(daily["net_return"]),
        "net_max_drawdown": max_drawdown(net_equity),
        "net_hit_rate": float((daily["net_return"] > 0.0).mean()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="Path to predictions CSV with columns: Date, code, split, prediction, actual.",
    )
    parser.add_argument("--cost-bps", type=float, default=0.0, help="One-way cost in bps (default 0 for oracle bound).")
    parser.add_argument("--min-positions", type=int, default=5)
    parser.add_argument("--rebalance-every", type=int, default=1)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Which split to test (train/val/test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown report path.",
    )
    return parser.parse_args(argv)


def format_report(args: argparse.Namespace, metrics: dict[str, float]) -> str:
    lines = [
        "# Synthetic Oracle Leak Test",
        "",
        f"- predictions CSV: `{args.predictions_csv}`",
        f"- split: `{args.split}`",
        f"- cost_bps: `{args.cost_bps}`",
        f"- rebalance_every: `{args.rebalance_every}`",
        f"- min_positions: `{args.min_positions}`",
        "",
        "## Result",
        "",
        f"- n_days: `{metrics['n_days']}`",
        f"- avg_positions: `{metrics['avg_positions']:.2f}`",
        f"- avg_turnover: `{metrics['avg_turnover']:.4f}`",
        f"- gross_equity_final: `{metrics['gross_equity_final']:.4f}`",
        f"- net_equity_final: `{metrics['net_equity_final']:.4f}`",
        f"- gross_sharpe: `{metrics['gross_sharpe']:.4f}`",
        f"- net_sharpe: `{metrics['net_sharpe']:.4f}`",
        f"- net_max_drawdown: `{metrics['net_max_drawdown']:.4f}`",
        f"- net_hit_rate: `{metrics['net_hit_rate']:.4f}`",
        "",
        "## Interpretation",
        "",
        "- gross_sharpe >> 5 hoặc ∞: alignment đúng, không có execution leak rõ rệt.",
        "- gross_sharpe finite trong khoảng [1, 5]: WARNING — có thể đang trade với một phần info chưa đáng ra biết.",
        "- gross_sharpe <= 1: cần điều tra; lý do thường là min_positions quá cao hoặc actual_aligned chứa NaN.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    df = pd.read_csv(args.predictions_csv, parse_dates=["Date"])
    if "split" in df.columns:
        df = df.loc[df["split"] == args.split].copy()
    if df.empty:
        raise SystemExit(f"No rows found for split={args.split} in {args.predictions_csv}")

    required = {"Date", "code", "prediction", "actual", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    aligned = align_signal_actual(df.dropna(subset=["prediction", "actual"]))
    aligned = aligned.dropna(subset=["actual_aligned", "actual_date"]).copy()
    aligned["actual_date"] = pd.to_datetime(aligned["actual_date"])

    metrics = run_oracle_backtest(
        aligned,
        cost_bps=args.cost_bps,
        min_positions=args.min_positions,
        rebalance_every=args.rebalance_every,
    )

    report = format_report(args, metrics)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
