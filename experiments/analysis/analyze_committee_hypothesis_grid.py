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

from src.models.selection.filter_signal import apply_daily_top_selection  # noqa: E402
from src.models.selection.holding_period import (  # noqa: E402
    aggregate_selected_daily,
    build_walk_forward_folds,
    evaluate_rebalance_grid,
    select_by_constrained_worst_year,
    summarize_rebalance,
)


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
FILTER_ROOT = RUN_ROOT / "reports" / "filter_signal"
DEFAULT_RUN = "portable_lstm_filter_signal_20260509_r06_selector_module"
DEFAULT_COVERAGE_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
LEGACY_CANDIDATES = (
    "prediction_gate",
    "prediction_move_top_20",
    "prediction_move_top_train_ic_selected",
)


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward ablation grid for modular committee hypotheses.")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--source-split", default="val")
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--min-positions", type=int, default=5)
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--coverage-grid", default=",".join(str(x) for x in DEFAULT_COVERAGE_GRID))
    parser.add_argument("--max-train-avg-turnover", type=float, default=0.25)
    parser.add_argument("--max-train-drawdown", type=float, default=0.15)
    parser.add_argument("--output-name", default="committee_hypothesis_grid_val_split")
    return parser.parse_args(argv)


def coverage_token(coverage: float) -> str:
    return f"{int(round(coverage * 100)):02d}"


def add_committee_candidate_columns(frame: pd.DataFrame, coverage_grid: tuple[float, ...]) -> dict[str, tuple[str, ...]]:
    out_groups: dict[str, list[str]] = {
        "h0_forecast_abs": [],
        "h1_tradeability_filter": [],
        "h2_risk_conditioned": [],
        "h3_rank_committee": [],
        "legacy_filter_shortlist": [col for col in LEGACY_CANDIDATES if col in frame.columns],
    }
    frame["score_base_abs"] = frame["base_prediction"].abs()
    frame["score_expected_move"] = frame["base_prediction"].abs() * frame["filter_probability"]
    if "market_risk_scale" not in frame.columns:
        frame["market_risk_scale"] = 1.0
    frame["score_risk_move"] = frame["score_expected_move"] * frame["market_risk_scale"].clip(lower=0.0)

    rank_parts = []
    for score_col in ("score_base_abs", "filter_probability", "market_risk_scale"):
        rank_col = f"{score_col}_daily_rank"
        frame[rank_col] = frame.groupby(["split", "Date"], sort=False)[score_col].rank(pct=True, method="average")
        rank_parts.append(rank_col)
    frame["score_rank_committee"] = frame[rank_parts].mean(axis=1)

    score_specs = {
        "h0_forecast_abs": "score_base_abs",
        "h1_tradeability_filter": "score_expected_move",
        "h2_risk_conditioned": "score_risk_move",
        "h3_rank_committee": "score_rank_committee",
    }
    for group_name, score_column in score_specs.items():
        for coverage in coverage_grid:
            token = coverage_token(coverage)
            output_column = f"committee_{group_name}_top_{token}"
            apply_daily_top_selection(
                frame,
                coverage,
                output_column,
                score_column,
                base_prediction_column="base_prediction",
                split_column="split",
                date_column="Date",
            )
            out_groups[group_name].append(output_column)

    all_candidates: list[str] = []
    for values in out_groups.values():
        all_candidates.extend(values)
    out_groups["all_committee_candidates"] = all_candidates
    return {key: tuple(values) for key, values in out_groups.items() if values}


def write_markdown(
    output_dir: Path,
    hypothesis_summary: pd.DataFrame,
    fold_results: pd.DataFrame,
    cost_bps: float,
    source_split: str,
    max_train_avg_turnover: float,
    max_train_drawdown: float,
) -> None:
    lines = [
        "# Modular Committee Hypothesis Grid",
        "",
        f"Source split: `{source_split}`. Holdout/test data is not used.",
        f"Assumed one-way transaction cost: `{cost_bps:.1f}` bps per unit turnover.",
        f"Train selector constraints: avg turnover <= `{max_train_avg_turnover:.2f}`, max DD no worse than `{max_train_drawdown:.0%}`.",
        "",
        "## Hypothesis Summary",
        "",
        "| Hypothesis | Folds | Positive folds | Net equity | Net Sharpe | Max DD | Avg turnover | Avg positions | Relaxed folds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in hypothesis_summary.iterrows():
        lines.append(
            "| "
            f"`{row['hypothesis']}` | {int(row['folds'])} | {float(row['positive_fold_rate']):.1%} | "
            f"{float(row['stitched_net_equity']):.3f} | {float(row['stitched_net_sharpe']):+.2f} | "
            f"{float(row['stitched_max_drawdown']):.1%} | {float(row['avg_turnover']):.2f} | "
            f"{float(row['avg_positions']):.1f} | {int(row['relaxed_folds'])} |"
        )
    lines.extend(
        [
            "",
            "## Fold Selections",
            "",
            "| Fold | Hypothesis | Candidate | Rebalance | Test period | Net equity | Net Sharpe | Train turnover | Train max DD |",
            "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in fold_results.sort_values(["fold", "hypothesis"], kind="stable").iterrows():
        period = f"{pd.Timestamp(row['test_start']).date()} to {pd.Timestamp(row['test_end']).date()}"
        lines.append(
            "| "
            f"{int(row['fold'])} | `{row['hypothesis']}` | `{row['candidate']}` | {int(row['rebalance_every'])} | "
            f"{period} | {float(row['test_net_equity']):.3f} | {float(row['test_net_sharpe']):+.2f} | "
            f"{float(row['train_avg_turnover']):.2f} | {float(row['train_net_max_drawdown']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Hypotheses",
            "",
            "- H0 forecast-only: select by raw forecast magnitude.",
            "- H1 tradeability filter: select by forecast magnitude multiplied by filter probability.",
            "- H2 risk-conditioned: H1 score multiplied by market risk scale.",
            "- H3 rank committee: average daily ranks from forecast magnitude, filter probability, and market risk scale.",
            "- Legacy shortlist: existing filter selector columns kept for continuity.",
        ]
    )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    coverage_grid = parse_float_list(args.coverage_grid)
    run_dir = FILTER_ROOT / args.run
    predictions_path = run_dir / "filter_predictions.csv.gz"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["Date", "actual_date"])
    predictions = predictions.loc[predictions["split"] == args.source_split].copy()
    if predictions.empty:
        raise ValueError(f"No rows for source_split={args.source_split!r}")

    candidate_groups = add_committee_candidate_columns(predictions, coverage_grid)
    all_unique_candidates = tuple(dict.fromkeys(candidate for group in candidate_groups.values() for candidate in group))
    dates = list(pd.Series(predictions["actual_date"].dropna().unique()).sort_values())
    folds = build_walk_forward_folds(dates, args.train_days, args.test_days, args.step_days)
    if not folds:
        raise ValueError("No walk-forward folds could be built.")

    fold_rows: list[dict[str, object]] = []
    daily_parts: list[pd.DataFrame] = []
    for fold in folds:
        train_frame = predictions.loc[predictions["actual_date"].isin(fold["train_dates"])]
        test_frame = predictions.loc[predictions["actual_date"].isin(fold["test_dates"])]
        train_grid_all, _ = evaluate_rebalance_grid(
            train_frame,
            all_unique_candidates,
            args.cost_bps,
            args.min_positions,
        )
        _, test_daily_map_all = evaluate_rebalance_grid(
            test_frame,
            all_unique_candidates,
            args.cost_bps,
            args.min_positions,
        )
        for hypothesis, candidates in candidate_groups.items():
            train_grid = train_grid_all.loc[train_grid_all["candidate"].isin(candidates)].copy()
            selected = select_by_constrained_worst_year(
                train_grid,
                max_avg_turnover=args.max_train_avg_turnover,
                max_drawdown=args.max_train_drawdown,
            )
            candidate = str(selected["candidate"])
            rebalance_every = int(selected["rebalance_every"])
            test_daily = test_daily_map_all[(candidate, rebalance_every)].copy()
            test_summary = summarize_rebalance(test_daily, "test", candidate, rebalance_every, args.cost_bps)
            test_daily.insert(0, "hypothesis", hypothesis)
            test_daily.insert(0, "fold", int(fold["fold"]))
            daily_parts.append(test_daily)
            fold_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "hypothesis": hypothesis,
                    "candidate": candidate,
                    "rebalance_every": rebalance_every,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "train_worst_year_equity": float(selected["worst_year_equity"]),
                    "train_full_equity": float(selected["full_equity"]),
                    "train_net_sharpe": float(selected["net_sharpe"]),
                    "train_avg_turnover": float(selected.get("summary_avg_turnover", np.nan)),
                    "train_net_max_drawdown": float(selected.get("summary_net_max_drawdown", np.nan)),
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
    selected_daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    summary_rows: list[dict[str, object]] = []
    for hypothesis, group in selected_daily.groupby("hypothesis", sort=True):
        fold_group = fold_results.loc[fold_results["hypothesis"] == hypothesis]
        summary_rows.append(
            {
                "hypothesis": hypothesis,
                "folds": int(len(fold_group)),
                "positive_fold_rate": float((fold_group["test_net_equity"] > 1.0).mean()),
                "avg_fold_net_equity": float(fold_group["test_net_equity"].mean()),
                "worst_fold_net_equity": float(fold_group["test_net_equity"].min()),
                "relaxed_folds": int(fold_group["constraints_relaxed"].sum()),
                **aggregate_selected_daily(group),
            }
        )
    hypothesis_summary = pd.DataFrame(summary_rows).sort_values(
        ["stitched_net_equity", "stitched_net_sharpe"],
        ascending=[False, False],
        kind="stable",
    )

    output_dir = run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    hypothesis_summary.to_csv(output_dir / "committee_hypothesis_summary.csv", index=False)
    fold_results.to_csv(output_dir / "committee_hypothesis_folds.csv", index=False)
    selected_daily.to_csv(output_dir / "committee_hypothesis_selected_daily.csv", index=False)
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
                "coverage_grid": list(coverage_grid),
                "max_train_avg_turnover": args.max_train_avg_turnover,
                "max_train_drawdown": args.max_train_drawdown,
                "candidate_groups": {key: list(value) for key, value in candidate_groups.items()},
                "summary": hypothesis_summary.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(
        output_dir,
        hypothesis_summary,
        fold_results,
        args.cost_bps,
        args.source_split,
        args.max_train_avg_turnover,
        args.max_train_drawdown,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "folds": int(len(fold_results["fold"].unique())),
                "hypotheses": int(hypothesis_summary.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
