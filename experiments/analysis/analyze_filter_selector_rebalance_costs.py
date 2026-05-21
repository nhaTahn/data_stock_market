from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.selection.holding_period import (  # noqa: E402
    DEFAULT_REBALANCE_DAYS,
    select_by_full_sharpe,
    select_by_worst_year,
    simulate_rebalance,
    summarize_rebalance,
    window_year_score,
)


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
FILTER_ROOT = RUN_ROOT / "reports" / "filter_signal"
DEFAULT_RUN = "portable_lstm_filter_signal_20260509_r06_selector_module"
DEFAULT_CANDIDATES = (
    "prediction_gate",
    "prediction_move_top_20",
    "prediction_move_top_train_ic_selected",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebalance-frequency cost read for filter selector candidates.")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--min-positions", type=int, default=5)
    parser.add_argument("--output-name", default="rebalance_cost_summary")
    return parser.parse_args(argv)


def train_selected_rebalance(summary: pd.DataFrame, candidate: str) -> int:
    train = summary.loc[(summary["split"] == "train") & (summary["candidate"] == candidate)].copy()
    if train.empty:
        raise ValueError(f"No train rows for candidate={candidate}")
    train = train.assign(full_equity=train["net_equity"])
    selected = select_by_full_sharpe(train)
    return int(selected["rebalance_every"])


def robust_train_selected_rebalance(daily_returns: pd.DataFrame, candidate: str) -> int:
    train = daily_returns.loc[
        (daily_returns["split"] == "train") & (daily_returns["candidate"] == candidate)
    ].copy()
    if train.empty:
        raise ValueError(f"No train daily rows for candidate={candidate}")
    rows: list[dict[str, object]] = []
    for rebalance_every, group in train.groupby("rebalance_every", sort=True):
        rows.append(
            {
                "rebalance_every": int(rebalance_every),
                **window_year_score(group),
            }
        )
    selected = select_by_worst_year(pd.DataFrame(rows))
    return int(selected["rebalance_every"])


def build_selected_rows(
    summary: pd.DataFrame,
    candidates: tuple[str, ...],
    selector_name: str,
    selector_fn: Callable[[str], int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        selected_rebalance = int(selector_fn(candidate))
        train_row = summary.loc[
            (summary["split"] == "train")
            & (summary["candidate"] == candidate)
            & (summary["rebalance_every"] == selected_rebalance)
        ].iloc[0]
        val_row = summary.loc[
            (summary["split"] == "val")
            & (summary["candidate"] == candidate)
            & (summary["rebalance_every"] == selected_rebalance)
        ].iloc[0]
        rows.append(
            {
                "selector": selector_name,
                "candidate": candidate,
                "rebalance_every": selected_rebalance,
                "train_net_equity": float(train_row["net_equity"]),
                "val_net_equity": float(val_row["net_equity"]),
                "train_net_sharpe": float(train_row["net_sharpe"]),
                "val_net_sharpe": float(val_row["net_sharpe"]),
                "val_net_max_drawdown": float(val_row["net_max_drawdown"]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(output_dir: Path, summary: pd.DataFrame, selected: pd.DataFrame, cost_bps: float) -> None:
    lines = [
        "# Filter Selector Rebalance Cost Read",
        "",
        "Scope: train/validation prediction artifact only. Holdout/test data is not used.",
        "",
        f"Assumed one-way transaction cost: `{cost_bps:.1f}` bps per unit turnover.",
        "",
        "## Train-Selected Rebalance",
        "",
        "| Selector | Candidate | Rebalance days | Train net equity | Val net equity | Train net Sharpe | Val net Sharpe | Val max DD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in selected.iterrows():
        lines.append(
            "| "
            f"`{row['selector']}` | `{row['candidate']}` | {int(row['rebalance_every'])} | "
            f"{float(row['train_net_equity']):.3f} | "
            f"{float(row['val_net_equity']):.3f} | {float(row['train_net_sharpe']):+.2f} | "
            f"{float(row['val_net_sharpe']):+.2f} | {float(row['val_net_max_drawdown']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Best Validation Rows",
            "",
            "| Split | Candidate | Rebalance | Net equity | Gross equity | Net Sharpe | Avg turnover | Avg positions | Max DD |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    best_val = summary.loc[summary["split"] == "val"].sort_values("net_equity", ascending=False, kind="stable").head(12)
    for _, row in best_val.iterrows():
        lines.append(
            "| "
            f"`{row['split']}` | `{row['candidate']}` | {int(row['rebalance_every'])} | "
            f"{float(row['net_equity']):.3f} | {float(row['gross_equity']):.3f} | "
            f"{float(row['net_sharpe']):+.2f} | {float(row['avg_turnover']):.2f} | "
            f"{float(row['avg_positions']):.1f} | {float(row['net_max_drawdown']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Robust train selection uses worst-year train net equity first; it is more conservative than full-period train Sharpe.",
            "- Slower rebalancing is useful only if a train-only selector also improves validation net equity.",
            "- If all validation net equity stays below 1 after costs, improve turnover/holding logic before training larger models.",
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
    output_dir = run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    daily_parts: list[pd.DataFrame] = []
    for split, split_frame in predictions.groupby("split", sort=True):
        for candidate in candidates:
            for rebalance_every in DEFAULT_REBALANCE_DAYS:
                daily = simulate_rebalance(
                    split_frame,
                    candidate,
                    rebalance_every=rebalance_every,
                    cost_bps=args.cost_bps,
                    min_positions=args.min_positions,
                )
                daily.insert(0, "rebalance_every", rebalance_every)
                daily.insert(0, "candidate", candidate)
                daily.insert(0, "split", split)
                daily_parts.append(daily)
                summary_rows.append(summarize_rebalance(daily, str(split), candidate, rebalance_every, args.cost_bps))

    summary = pd.DataFrame(summary_rows)
    daily_returns = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    selected = pd.concat(
        [
            build_selected_rows(
                summary,
                candidates,
                "full_train_net_sharpe",
                lambda candidate: train_selected_rebalance(summary, candidate),
            ),
            build_selected_rows(
                summary,
                candidates,
                "worst_year_net_equity",
                lambda candidate: robust_train_selected_rebalance(daily_returns, candidate),
            ),
        ],
        ignore_index=True,
    )
    summary.to_csv(output_dir / "rebalance_cost_summary.csv", index=False)
    selected.to_csv(output_dir / "train_selected_rebalance.csv", index=False)
    daily_returns.to_csv(output_dir / "rebalance_cost_daily.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run": args.run,
                "cost_bps": args.cost_bps,
                "min_positions": args.min_positions,
                "rebalance_days": list(DEFAULT_REBALANCE_DAYS),
                "candidates": list(candidates),
                "train_selected": selected.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, summary, selected, args.cost_bps)
    print(json.dumps({"output_dir": str(output_dir), "rows": int(len(summary))}, indent=2))


if __name__ == "__main__":
    main()
