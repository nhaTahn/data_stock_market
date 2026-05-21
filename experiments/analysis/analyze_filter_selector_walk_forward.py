from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.selection.holding_period import (  # noqa: E402
    aggregate_selected_daily,
    build_walk_forward_folds,
    evaluate_rebalance_grid,
    select_by_constrained_worst_year,
    select_by_full_sharpe,
    select_by_worst_year,
    summarize_rebalance,
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
    parser = argparse.ArgumentParser(description="Walk-forward validation for filter selector holding-period rules.")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--min-positions", type=int, default=5)
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--max-train-avg-turnover", type=float)
    parser.add_argument("--max-train-drawdown", type=float)
    parser.add_argument("--min-train-worst-year-equity", type=float)
    parser.add_argument("--output-name", default="walk_forward_cost_summary")
    return parser.parse_args(argv)


def write_markdown(
    output_dir: Path,
    fold_results: pd.DataFrame,
    selector_summary: pd.DataFrame,
    cost_bps: float,
    source_split: str,
) -> None:
    lines = [
        "# Filter Selector Walk-Forward Cost Validation",
        "",
        f"Source split: `{source_split}`. Holdout/test data is not used.",
        f"Assumed one-way transaction cost: `{cost_bps:.1f}` bps per unit turnover.",
        "",
        "Each fold selects candidate + rebalance on the trailing train window, then tests the next window.",
        "",
        "## Selector Summary",
        "",
        "| Selector | Folds | Positive folds | Stitched net equity | Stitched gross equity | Net Sharpe | Avg turnover | Avg positions | Max DD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in selector_summary.iterrows():
        lines.append(
            "| "
            f"`{row['selector']}` | {int(row['folds'])} | {float(row['positive_fold_rate']):.1%} | "
            f"{float(row['stitched_net_equity']):.3f} | {float(row['stitched_gross_equity']):.3f} | "
            f"{float(row['stitched_net_sharpe']):+.2f} | {float(row['avg_turnover']):.2f} | "
            f"{float(row['avg_positions']):.1f} | {float(row['stitched_max_drawdown']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Fold Results",
            "",
            "| Fold | Selector | Candidate | Rebalance | Test period | Test net equity | Test Sharpe | Train worst-year | Train turnover | Relaxed |",
            "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in fold_results.sort_values(["fold", "selector"], kind="stable").iterrows():
        period = f"{pd.Timestamp(row['test_start']).date()} to {pd.Timestamp(row['test_end']).date()}"
        lines.append(
            "| "
            f"{int(row['fold'])} | `{row['selector']}` | `{row['candidate']}` | {int(row['rebalance_every'])} | "
            f"{period} | {float(row['test_net_equity']):.3f} | {float(row['test_net_sharpe']):+.2f} | "
            f"{float(row['train_worst_year_equity']):.3f} | {float(row.get('train_avg_turnover', float('nan'))):.2f} | "
            f"{bool(row.get('constraints_relaxed', False))} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- This is a stricter check than the single train->validation read, but it still uses existing train/validation artifacts only.",
            "- Prefer selectors with positive stitched net equity, positive-fold rate above 50%, and tolerable drawdown.",
            "- If a selector fails here, do not train a larger model around it.",
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
    frame = predictions.loc[predictions["split"] == args.source_split].copy()
    if frame.empty:
        raise ValueError(f"No rows for source_split={args.source_split!r}")
    candidates = tuple(candidate for candidate in DEFAULT_CANDIDATES if candidate in frame.columns)
    dates = list(pd.Series(frame["actual_date"].dropna().unique()).sort_values())
    folds = build_walk_forward_folds(dates, args.train_days, args.test_days, args.step_days)
    if not folds:
        raise ValueError("No walk-forward folds could be built.")

    fold_rows: list[dict[str, object]] = []
    selected_daily_parts: list[pd.DataFrame] = []
    for fold in folds:
        train_frame = frame.loc[frame["actual_date"].isin(fold["train_dates"])]
        test_frame = frame.loc[frame["actual_date"].isin(fold["test_dates"])]
        train_grid, _ = evaluate_rebalance_grid(train_frame, candidates, args.cost_bps, args.min_positions)
        _, test_daily_map = evaluate_rebalance_grid(test_frame, candidates, args.cost_bps, args.min_positions)
        selectors = {
            "full_train_net_sharpe": select_by_full_sharpe(train_grid),
            "worst_year_net_equity": select_by_worst_year(train_grid),
        }
        if (
            args.max_train_avg_turnover is not None
            or args.max_train_drawdown is not None
            or args.min_train_worst_year_equity is not None
        ):
            selectors["constrained_worst_year_net_equity"] = select_by_constrained_worst_year(
                train_grid,
                max_avg_turnover=args.max_train_avg_turnover,
                max_drawdown=args.max_train_drawdown,
                min_worst_year_equity=args.min_train_worst_year_equity,
            )
        for selector_name, selected in selectors.items():
            candidate = str(selected["candidate"])
            rebalance_every = int(selected["rebalance_every"])
            test_daily = test_daily_map[(candidate, rebalance_every)].copy()
            test_summary = summarize_rebalance(test_daily, "test", candidate, rebalance_every, args.cost_bps)
            test_daily.insert(0, "selector", selector_name)
            test_daily.insert(0, "fold", int(fold["fold"]))
            selected_daily_parts.append(test_daily)
            fold_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "selector": selector_name,
                    "candidate": candidate,
                    "rebalance_every": rebalance_every,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "train_worst_year_equity": float(selected["worst_year_equity"]),
                    "train_full_equity": float(selected["full_equity"]),
                    "train_net_sharpe": float(selected["net_sharpe"]),
                    "train_avg_turnover": float(selected.get("summary_avg_turnover", float("nan"))),
                    "train_net_max_drawdown": float(selected.get("summary_net_max_drawdown", float("nan"))),
                    "constraints_relaxed": bool(selected.get("constraints_relaxed", False)),
                    "test_net_equity": float(test_summary["net_equity"]),
                    "test_gross_equity": float(test_summary["gross_equity"]),
                    "test_net_sharpe": float(test_summary["net_sharpe"]),
                    "test_avg_turnover": float(test_summary["avg_turnover"]),
                    "test_avg_positions": float(test_summary["avg_positions"]),
                    "test_max_drawdown": float(test_summary["net_max_drawdown"]),
                }
            )

    fold_results = pd.DataFrame(fold_rows)
    selected_daily = pd.concat(selected_daily_parts, ignore_index=True) if selected_daily_parts else pd.DataFrame()
    summary_rows: list[dict[str, object]] = []
    for selector_name, group in selected_daily.groupby("selector", sort=True):
        fold_group = fold_results.loc[fold_results["selector"] == selector_name]
        row = {
            "selector": selector_name,
            "folds": int(len(fold_group)),
            "positive_fold_rate": float((fold_group["test_net_equity"] > 1.0).mean()),
            "avg_fold_net_equity": float(fold_group["test_net_equity"].mean()),
            "worst_fold_net_equity": float(fold_group["test_net_equity"].min()),
            **aggregate_selected_daily(group),
        }
        summary_rows.append(row)
    selector_summary = pd.DataFrame(summary_rows).sort_values("stitched_net_equity", ascending=False, kind="stable")

    output_dir = run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(output_dir / "walk_forward_folds.csv", index=False)
    selector_summary.to_csv(output_dir / "walk_forward_selector_summary.csv", index=False)
    selected_daily.to_csv(output_dir / "walk_forward_selected_daily.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run": args.run,
                "source_split": args.source_split,
                "cost_bps": args.cost_bps,
                "min_positions": args.min_positions,
                "train_days": args.train_days,
                "test_days": args.test_days,
                "step_days": args.step_days,
                "max_train_avg_turnover": args.max_train_avg_turnover,
                "max_train_drawdown": args.max_train_drawdown,
                "min_train_worst_year_equity": args.min_train_worst_year_equity,
                "folds": len(fold_results["fold"].unique()),
                "summary": selector_summary.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, fold_results, selector_summary, args.cost_bps, args.source_split)
    print(json.dumps({"output_dir": str(output_dir), "folds": int(len(fold_results["fold"].unique()))}, indent=2))


if __name__ == "__main__":
    main()
