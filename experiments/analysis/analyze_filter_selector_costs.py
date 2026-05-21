from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
FILTER_ROOT = RUN_ROOT / "reports" / "filter_signal"
DEFAULT_RUN = "portable_lstm_filter_signal_20260509_r06_selector_module"
DEFAULT_CANDIDATES = (
    "prediction_base",
    "prediction_gate",
    "prediction_move_top_20",
    "prediction_move_top_train_ic_selected",
    "prediction_move_top_train_ic_selected_risk_scaled",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cost and turnover read for filter selector prediction streams.")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--output-name", default="cost_turnover_summary")
    parser.add_argument("--min-positions", type=int, default=5)
    return parser.parse_args(argv)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak.replace(0.0, np.nan) - 1.0
    return float(drawdown.min())


def annualized_sharpe(daily_returns: pd.Series) -> float:
    clean = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std <= 0.0:
        return float("nan")
    return float(clean.mean() / std * np.sqrt(252.0))


def build_daily_weights(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    min_positions: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    required = ["actual_date", "code", prediction_column, "actual_aligned"]
    clean = frame.dropna(subset=required).copy()
    clean = clean.loc[clean[prediction_column].astype(float) != 0.0]
    for actual_date, day in clean.groupby("actual_date", sort=True):
        day = day.copy()
        longs = day[prediction_column] > 0.0
        shorts = day[prediction_column] < 0.0
        n_long = int(longs.sum())
        n_short = int(shorts.sum())
        if n_long + n_short < min_positions:
            continue

        day["weight"] = 0.0
        if n_long > 0 and n_short > 0:
            day.loc[longs, "weight"] = 0.5 / n_long
            day.loc[shorts, "weight"] = -0.5 / n_short
        elif n_long > 0:
            day.loc[longs, "weight"] = 1.0 / n_long
        elif n_short > 0:
            day.loc[shorts, "weight"] = -1.0 / n_short
        day["actual_date"] = actual_date
        rows.append(day[["actual_date", "code", "weight", "actual_aligned"]])
    if not rows:
        return pd.DataFrame(columns=["actual_date", "code", "weight", "actual_aligned"])
    return pd.concat(rows, ignore_index=True)


def daily_turnover(weights: pd.DataFrame) -> pd.Series:
    if weights.empty:
        return pd.Series(dtype=float)
    pivot = (
        weights.pivot_table(index="actual_date", columns="code", values="weight", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    previous = pivot.shift(1).fillna(0.0)
    return (pivot - previous).abs().sum(axis=1)


def portfolio_daily_returns(weights: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(
            columns=[
                "actual_date",
                "gross_return",
                "turnover",
                "cost_return",
                "net_return",
                "n_positions",
                "gross_exposure",
            ]
        )
    tmp = weights.assign(weighted_return=weights["weight"] * weights["actual_aligned"])
    daily = (
        tmp.groupby("actual_date", sort=True)
        .agg(
            gross_return=("weighted_return", "sum"),
            n_positions=("weight", lambda s: int((s != 0.0).sum())),
            gross_exposure=("weight", lambda s: float(s.abs().sum())),
        )
        .reset_index()
    )
    turnover = daily_turnover(weights).rename("turnover").reset_index()
    turnover.columns = ["actual_date", "turnover"]
    daily = daily.merge(turnover, on="actual_date", how="left")
    daily["cost_return"] = daily["turnover"].fillna(0.0) * cost_bps / 10_000.0
    daily["net_return"] = daily["gross_return"] - daily["cost_return"]
    return daily


def summarize_daily_returns(daily: pd.DataFrame, *, split: str, candidate: str, cost_bps: float) -> dict[str, object]:
    if daily.empty:
        return {
            "split": split,
            "candidate": candidate,
            "cost_bps": cost_bps,
            "n_days": 0,
            "avg_positions": float("nan"),
            "avg_turnover": float("nan"),
            "gross_equity": float("nan"),
            "net_equity": float("nan"),
            "gross_sharpe": float("nan"),
            "net_sharpe": float("nan"),
            "net_max_drawdown": float("nan"),
            "net_hit_rate": float("nan"),
        }
    gross_equity = (1.0 + daily["gross_return"]).cumprod()
    net_equity = (1.0 + daily["net_return"]).cumprod()
    return {
        "split": split,
        "candidate": candidate,
        "cost_bps": cost_bps,
        "n_days": int(len(daily)),
        "avg_positions": float(daily["n_positions"].mean()),
        "avg_turnover": float(daily["turnover"].mean()),
        "gross_equity": float(gross_equity.iloc[-1]),
        "net_equity": float(net_equity.iloc[-1]),
        "gross_sharpe": annualized_sharpe(daily["gross_return"]),
        "net_sharpe": annualized_sharpe(daily["net_return"]),
        "net_max_drawdown": max_drawdown(net_equity),
        "net_hit_rate": float((daily["net_return"] > 0.0).mean()),
    }


def write_markdown(output_dir: Path, summary: pd.DataFrame, cost_bps: float) -> None:
    lines = [
        "# Filter Selector Cost And Turnover Read",
        "",
        "Scope: train/validation prediction artifact only. Holdout/test data is not used.",
        "",
        f"Assumed one-way transaction cost: `{cost_bps:.1f}` bps per unit turnover.",
        "",
        "Portfolio construction: selected non-zero predictions become daily equal-weight long/short weights.",
        "",
        "## Summary",
        "",
        "| Split | Candidate | Net equity | Gross equity | Net Sharpe | Gross Sharpe | Avg turnover | Avg positions | Net max DD | Days |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values(["split", "net_equity"], ascending=[True, False], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['split']}` | `{row['candidate']}` | {float(row['net_equity']):.3f} | "
            f"{float(row['gross_equity']):.3f} | {float(row['net_sharpe']):+.2f} | "
            f"{float(row['gross_sharpe']):+.2f} | {float(row['avg_turnover']):.2f} | "
            f"{float(row['avg_positions']):.1f} | {float(row['net_max_drawdown']):.1%} | {int(row['n_days'])} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- If net equity collapses versus gross equity, reduce turnover before further model training.",
            "- If a sparse candidate has better net equity but too few positions, treat it as a tactical overlay rather than full portfolio signal.",
            "- Use this as a cost sanity check; it is not a replacement for rolling validation.",
        ]
    )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = FILTER_ROOT / args.run
    predictions_path = run_dir / "filter_predictions.csv.gz"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["Date", "actual_date"])
    candidates = tuple(candidate for candidate in DEFAULT_CANDIDATES if candidate in predictions.columns)
    if not candidates:
        raise ValueError("No candidate prediction columns found.")

    output_dir = run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    daily_parts: list[pd.DataFrame] = []
    for split, split_frame in predictions.groupby("split", sort=True):
        for candidate in candidates:
            weights = build_daily_weights(split_frame, candidate, min_positions=args.min_positions)
            daily = portfolio_daily_returns(weights, args.cost_bps)
            daily.insert(0, "candidate", candidate)
            daily.insert(0, "split", split)
            daily_parts.append(daily)
            summary_rows.append(
                summarize_daily_returns(daily, split=str(split), candidate=candidate, cost_bps=args.cost_bps)
            )

    summary = pd.DataFrame(summary_rows)
    daily_returns = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    summary.to_csv(output_dir / "cost_turnover_summary.csv", index=False)
    daily_returns.to_csv(output_dir / "cost_turnover_daily.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run": args.run,
                "cost_bps": args.cost_bps,
                "min_positions": args.min_positions,
                "candidates": list(candidates),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, summary, args.cost_bps)
    print(json.dumps({"output_dir": str(output_dir), "candidates": list(candidates), "rows": int(len(summary))}, indent=2))


if __name__ == "__main__":
    main()
