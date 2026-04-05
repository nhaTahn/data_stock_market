from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize run metrics and best threshold backtests across horizons.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pick_backtest_summary(run_dir: Path, target_mode: str) -> tuple[str, Path]:
    if target_mode in {"return_3d", "return_5d"}:
        non_overlap = run_dir / "threshold_backtest_summary_non_overlap.json"
        if non_overlap.exists():
            return "non_overlap", non_overlap
    return "overlap", run_dir / "threshold_backtest_summary.json"


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str | float | int]] = []

    for run_dir in args.run_dirs:
        with (run_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        target_mode = config["target_mode"]
        backtest_mode, backtest_path = pick_backtest_summary(run_dir, target_mode)
        with backtest_path.open("r", encoding="utf-8") as f:
            backtest = json.load(f)

        for model_name, split_metrics in metrics.items():
            test_metrics = split_metrics["test"]
            backtest_metrics = backtest.get(model_name, {})
            rows.append(
                {
                    "run_name": run_dir.name,
                    "stock": config.get("stocks", ""),
                    "target_mode": target_mode,
                    "feature_columns": ",".join(config.get("feature_columns", [])),
                    "model": model_name,
                    "backtest_mode": backtest_mode,
                    "test_rel_score": test_metrics.get("rel_score"),
                    "test_directional_accuracy": test_metrics.get("directional_accuracy"),
                    "best_threshold": backtest_metrics.get("threshold"),
                    "holding_period": backtest_metrics.get("holding_period"),
                    "trade_count": backtest_metrics.get("trade_count"),
                    "coverage": backtest_metrics.get("coverage"),
                    "threshold_directional_accuracy": backtest_metrics.get("directional_accuracy"),
                    "avg_strategy_return": backtest_metrics.get("avg_strategy_return"),
                    "cumulative_strategy_return": backtest_metrics.get("cumulative_strategy_return"),
                    "final_equity": backtest_metrics.get("final_equity"),
                    "sampled_buy_hold_equity": backtest_metrics.get("sampled_buy_hold_equity"),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(args.output)


if __name__ == "__main__":
    main()
