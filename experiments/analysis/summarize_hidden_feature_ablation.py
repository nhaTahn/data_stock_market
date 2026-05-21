from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
MODELS = (
    "lstm_top2_by_val",
    "lstm_best_by_val",
    "lstm_ensemble",
    "linear_regression",
    "arima",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize hidden-feature ablation runs.")
    parser.add_argument("--runs", nargs="+", required=True)
    return parser


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print("run,model,val_rel_score,val_rmse,val_directional_accuracy,val_quartile_final_equity")
    for run_name in args.runs:
        core_dir = RUN_ROOT / run_name / "reports" / "core"
        metrics = load_json(core_dir / "metrics.json")
        evaluation = load_json(core_dir / "evaluation_summary.json")
        for model_name in MODELS:
            if model_name not in metrics:
                continue
            val_metrics = metrics[model_name]["val"]
            quartile = (evaluation.get(model_name, {}).get("val") or {}).get("quartile_long_short") or {}
            print(
                ",".join(
                    [
                        run_name,
                        model_name,
                        f"{val_metrics['rel_score']:.6f}",
                        f"{val_metrics['rmse']:.6f}",
                        f"{val_metrics['directional_accuracy']:.6f}",
                        f"{quartile.get('final_equity', float('nan')):.6f}",
                    ]
                )
            )


if __name__ == "__main__":
    main()
