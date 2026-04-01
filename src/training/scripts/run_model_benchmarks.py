from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from model_benchmark.classical import run_arima, run_garch, run_gaussian_nb
from model_benchmark.data import prepare_sequence_data
from model_benchmark.reporting import plot_summary, save_model_metrics, save_summary
from model_benchmark.transformer import run_transformer
from tf_lstm.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TRAIN_END_DATE, VAL_END_DATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Transformer, ARIMA, GARCH, and Naive Bayes benchmarks.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to the cleaned training dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "model_benchmark", help="Output directory.")
    parser.add_argument("--train-end", default=TRAIN_END_DATE, help="Last date in the training split.")
    parser.add_argument("--val-end", default=VAL_END_DATE, help="Last date in the validation split.")
    parser.add_argument("--window-size", type=int, default=30, help="Sequence length for Transformer and Naive Bayes.")
    parser.add_argument("--feature-groups", default="base,liquidity", help="Comma-separated feature groups for sequence models.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for Transformer.")
    parser.add_argument("--epochs", type=int, default=15, help="Transformer training epochs.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for Transformer.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for Transformer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--transformer-head-size", type=int, default=16, help="Transformer attention head size.")
    parser.add_argument("--transformer-num-heads", type=int, default=4, help="Transformer number of attention heads.")
    parser.add_argument("--transformer-ff-dim", type=int, default=64, help="Transformer feed-forward width.")
    parser.add_argument("--arima-order", default="1,0,0", help="ARIMA order as p,d,q.")
    parser.add_argument("--classical-min-history", type=int, default=60, help="Minimum history for ARIMA/GARCH rolling forecasts.")
    parser.add_argument("--garch-refit-interval", type=int, default=30, help="Refit interval for rolling GARCH forecasts.")
    parser.add_argument("--max-tickers", type=int, default=None, help="Optional cap on the number of tickers for smoke runs.")
    return parser.parse_args()


def _metrics_to_rows(model_name: str, metrics: dict[str, dict[str, float]], extra: dict[str, object]) -> list[dict]:
    rows: list[dict] = []
    for split, split_metrics in metrics.items():
        rows.append({"model": model_name, "split": split, **split_metrics, **extra})
    return rows


def main() -> None:
    args = parse_args()
    feature_groups = [group.strip() for group in args.feature_groups.split(",") if group.strip()]
    arima_order = tuple(int(x.strip()) for x in args.arima_order.split(","))

    summary_rows: list[dict] = []

    bundle = prepare_sequence_data(
        data_path=args.data_path,
        train_end=args.train_end,
        val_end=args.val_end,
        window_size=args.window_size,
        feature_groups=feature_groups,
        max_tickers=args.max_tickers,
    )

    baseline_config = {
        "data_path": str(args.data_path),
        "feature_groups": bundle.feature_groups,
        "window_size": args.window_size,
        "baseline": "predict next-day return = 0",
        "train_sequences": len(bundle.train_targets),
        "val_sequences": len(bundle.val_targets),
        "test_sequences": len(bundle.test_targets),
    }
    save_model_metrics(args.output_dir, "baseline", bundle.baseline_metrics, baseline_config)
    summary_rows.extend(_metrics_to_rows("baseline", bundle.baseline_metrics, {"evaluated_on": "sequence"}))

    transformer_metrics, history_df, transformer_model = run_transformer(bundle, args)
    transformer_config = {
        "feature_groups": bundle.feature_groups,
        "feature_count": len(bundle.feature_columns),
        "window_size": args.window_size,
        "head_size": args.transformer_head_size,
        "num_heads": args.transformer_num_heads,
        "ff_dim": args.transformer_ff_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
    }
    save_model_metrics(args.output_dir, "transformer", transformer_metrics, transformer_config)
    history_df.to_csv(args.output_dir / "transformer_fit_history.csv", index=False)
    transformer_model.save(args.output_dir / "transformer_model.keras")
    summary_rows.extend(_metrics_to_rows("transformer", transformer_metrics, {"evaluated_on": "sequence"}))

    naive_bayes_result = run_gaussian_nb(bundle)
    nb_config = {
        "feature_groups": bundle.feature_groups,
        "feature_count": len(bundle.feature_columns),
        "window_size": args.window_size,
        **naive_bayes_result.extra,
    }
    save_model_metrics(args.output_dir, "naive_bayes", naive_bayes_result.metrics, nb_config)
    summary_rows.extend(_metrics_to_rows("naive_bayes", naive_bayes_result.metrics, {"evaluated_on": "sequence"}))

    arima_result = run_arima(
        data_path=args.data_path,
        train_end=args.train_end,
        val_end=args.val_end,
        order=arima_order,
        min_history=args.classical_min_history,
        max_tickers=args.max_tickers,
    )
    arima_config = {
        "order": arima_order,
        **arima_result.extra,
    }
    save_model_metrics(args.output_dir, "arima", arima_result.metrics, arima_config)
    summary_rows.extend(_metrics_to_rows("arima", arima_result.metrics, {"evaluated_on": "rolling_univariate"}))

    garch_result = run_garch(
        data_path=args.data_path,
        train_end=args.train_end,
        val_end=args.val_end,
        min_history=args.classical_min_history,
        refit_interval=args.garch_refit_interval,
        max_tickers=args.max_tickers,
    )
    garch_config = {
        **garch_result.extra,
    }
    save_model_metrics(args.output_dir, "garch", garch_result.metrics, garch_config)
    summary_rows.extend(_metrics_to_rows("garch", garch_result.metrics, {"evaluated_on": "rolling_univariate"}))

    summary_df = pd.DataFrame(summary_rows)
    save_summary(args.output_dir, summary_df)
    plot_summary(args.output_dir, summary_df)
    print(f"Saved benchmark outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
