from __future__ import annotations

from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from fk_lstm_classifier.config import namespace_to_config, parse_args
from fk_lstm_classifier.data import load_market_data, prepare_datasets
from fk_lstm_classifier.evaluation import (
    build_prediction_frame,
    compute_classification_metrics,
    compute_long_short_returns,
)
from fk_lstm_classifier.model import build_lstm_classifier, set_seed
from fk_lstm_classifier.training import train_classifier


def save_artifacts(
    output_dir: Path,
    history: dict[str, list[float]],
    validation_predictions: pd.DataFrame,
    strategy_returns: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "fit_history.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    strategy_returns.to_csv(output_dir / "validation_long_short_returns.csv", index=False)


def main() -> None:
    args = parse_args()
    config = namespace_to_config(args)

    set_seed(config.seed)
    price_frame = load_market_data(config.data_dir, config.markets)
    _, train_dataset, validation_dataset = prepare_datasets(
        price_frame=price_frame,
        lookback=config.lookback,
        train_ratio=config.train_ratio,
        min_cross_sectional_count=config.min_cross_sectional_count,
    )

    model = build_lstm_classifier(
        lookback=config.lookback,
        feature_dim=1,
        lstm_units=config.lstm_units,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        use_attention=(config.model_type == "attention"),
    )
    train_result = train_classifier(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
    )

    validation_probabilities = model.predict(validation_dataset.features, verbose=0)
    validation_predictions = build_prediction_frame(validation_dataset, validation_probabilities)
    strategy_returns = compute_long_short_returns(
        validation_predictions,
        top_k=config.top_k,
    )

    metrics = compute_classification_metrics(
        labels=validation_dataset.labels,
        prob_class_1=validation_probabilities[:, 1],
    )

    output_dir = config.output_dir / f"{'_'.join(config.markets).lower()}_{config.model_type}"
    save_artifacts(output_dir, train_result.history, validation_predictions, strategy_returns)

    print("\nValidation classification metrics:")
    print(pd.DataFrame([metrics]).to_string(index=False))
    if not strategy_returns.empty:
        summary = {
            "days": int(len(strategy_returns)),
            "mean_daily_return": float(strategy_returns["strategy_return"].mean()),
            "vol_daily_return": float(strategy_returns["strategy_return"].std(ddof=0)),
            "final_equity": float(strategy_returns["equity_curve"].iloc[-1]),
        }
        print("\nValidation long-short strategy summary:")
        print(pd.DataFrame([summary]).to_string(index=False))
    else:
        print("\nValidation long-short strategy summary:")
        print("No strategy returns were produced. Increase history or lower top_k.")

    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
