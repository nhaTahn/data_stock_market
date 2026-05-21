from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_high_error_regime_filter import (  # noqa: E402
    DAILY_FEATURE_COLUMNS,
    DEFAULT_DATA,
    DEFAULT_GOLD_DIR,
    DEFAULT_PREDICTIONS,
    build_daily_regime_frame,
    pct,
    read_predictions,
    read_raw,
    read_universe,
    score_classifier,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a daily LSTM high-error regime filter.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output-name", default="high_error_lstm_filter_probe")
    parser.add_argument("--universe", default="vn100", choices=["vn30", "vn100", "all"])
    parser.add_argument("--spike-threshold", type=float, default=0.035)
    parser.add_argument("--leader-top-k", type=int, default=10)
    parser.add_argument("--leader-window", type=int, default=60)
    parser.add_argument("--leader-min-periods", type=int, default=20)
    parser.add_argument("--segment-year", type=int, default=2017)
    parser.add_argument("--segment-start-day", type=int, default=200)
    parser.add_argument("--segment-end-day", type=int, default=250)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args(argv)


def fit_scaler(train: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    med = train.loc[:, DAILY_FEATURE_COLUMNS].median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    filled = train.loc[:, DAILY_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(med)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return med, mean, std


def apply_scaler(frame: pd.DataFrame, med: pd.Series, mean: pd.Series, std: pd.Series) -> np.ndarray:
    filled = frame.loc[:, DAILY_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(med)
    return ((filled - mean) / std).to_numpy(dtype=np.float32)


def build_sequences(
    daily: pd.DataFrame,
    med: pd.Series,
    mean: pd.Series,
    std: pd.Series,
    window: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    x_map: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    y_map: dict[str, list[float]] = {"train": [], "val": []}
    meta_map: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    for split, group in daily.sort_values(["split", "Date"], kind="stable").groupby("split", sort=False):
        if split not in x_map or len(group) < window:
            continue
        values = apply_scaler(group, med, mean, std)
        labels = group["high_error_label"].to_numpy(dtype=np.float32)
        rows = group.reset_index(drop=True)
        for end_idx in range(window - 1, len(rows)):
            x_map[str(split)].append(values[end_idx - window + 1 : end_idx + 1])
            y_map[str(split)].append(float(labels[end_idx]))
            meta_map[str(split)].append(rows.iloc[end_idx].to_dict())
    return (
        {split: np.asarray(items, dtype=np.float32) for split, items in x_map.items()},
        {split: np.asarray(items, dtype=np.float32) for split, items in y_map.items()},
        {split: pd.DataFrame(items) for split, items in meta_map.items()},
    )


def build_model(window: int, n_features: int, seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, n_features)),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(12, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def class_weight(y_train: np.ndarray) -> dict[int, float]:
    positive = float(np.mean(y_train))
    if positive <= 0.0 or positive >= 1.0:
        return {0: 1.0, 1: 1.0}
    return {0: 0.5 / (1.0 - positive), 1: 0.5 / positive}


def add_scores(
    model: tf.keras.Model,
    x_map: dict[str, np.ndarray],
    meta_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    scored_parts: list[pd.DataFrame] = []
    for split in ("train", "val"):
        part = meta_map[split].copy()
        part["lstm_high_error_probability"] = model.predict(x_map[split], verbose=0).reshape(-1)
        scored_parts.append(part)
    return pd.concat(scored_parts, ignore_index=True)


def score_lstm(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, group in scored.groupby("split", sort=True):
        row = {"model": "daily_lstm", "split": split}
        row.update(score_classifier(group, "lstm_high_error_probability"))
        rows.append(row)
    return pd.DataFrame(rows)


def segment_frame(scored: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    work = scored[scored["Date"].dt.year.eq(args.segment_year)].sort_values("Date", kind="stable").reset_index(drop=True)
    work["trading_day_in_year"] = np.arange(len(work))
    return work[
        work["trading_day_in_year"].between(args.segment_start_day, args.segment_end_day, inclusive="both")
    ].copy()


def segment_metrics(segment: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    row = {
        "model": "daily_lstm",
        "split": f"segment_{args.segment_year}_d{args.segment_start_day}_{args.segment_end_day}",
    }
    row.update(score_classifier(segment, "lstm_high_error_probability"))
    return pd.DataFrame([row])


def write_summary(
    output_dir: Path,
    metrics: pd.DataFrame,
    segment_stats: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    val = metrics[metrics["split"].eq("val")].iloc[0]
    train = metrics[metrics["split"].eq("train")].iloc[0]
    seg = segment_stats.iloc[0]
    display = pd.concat([metrics, segment_stats], ignore_index=True).copy()
    for column in [
        "spike_rate",
        "top20_spike_rate",
        "rest_spike_rate",
        "top20_median_q90_abs_error",
        "rest_median_q90_abs_error",
    ]:
        display[column] = display[column].map(pct)
    lines = [
        "# Daily LSTM High-Error Filter Probe",
        "",
        "Purpose: test whether a small LSTM over daily regime features can detect days where the frozen base LSTM has high q90 prediction error.",
        f"Target label: `q90(|actual_return - predicted_return|) > {args.spike_threshold:.1%}`. Holdout/test is not used.",
        "",
        "## Result",
        "",
        f"- Train AUC: `{train['auc']:.3f}`; validation AUC: `{val['auc']:.3f}`.",
        f"- Validation baseline spike rate: `{pct(float(val['spike_rate']))}`.",
        f"- Validation top-20% LSTM-risk spike rate: `{pct(float(val['top20_spike_rate']))}`.",
        f"- 2017 segment top-20% LSTM-risk spike rate: `{pct(float(seg['top20_spike_rate']))}`.",
        f"- 2017 segment top-risk median q90(|E|): `{pct(float(seg['top20_median_q90_abs_error']))}` vs rest `{pct(float(seg['rest_median_q90_abs_error']))}`.",
        "",
        "Interpretation: this filter should be used as a risk/no-trade or position-sizing layer. It does not change base next-day return predictions; it tests whether the timing of large errors is learnable from regime features.",
        "",
        "## Metrics",
        "",
        display.to_markdown(index=False),
        "",
        "## Next",
        "",
        "If this is promoted, the next full experiment should add these regime features to the existing per-stock LSTM filter and evaluate trading metrics after cost, not only q90 error timing.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, symbols = read_universe(args.universe)
    output_dir = args.gold_dir / "plots" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(args.predictions, symbols)
    raw = read_raw(args.data, symbols)
    daily = build_daily_regime_frame(predictions, raw, args)
    train = daily[daily["split"].eq("train")].copy()
    med, mean, std = fit_scaler(train)
    x_map, y_map, meta_map = build_sequences(daily, med, mean, std, args.window)
    if len(x_map["train"]) == 0 or len(x_map["val"]) == 0:
        raise ValueError("Not enough sequences for train/val.")

    model = build_model(args.window, len(DAILY_FEATURE_COLUMNS), args.seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_map["train"],
        y_map["train"],
        validation_data=(x_map["val"], y_map["val"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight(y_map["train"]),
        callbacks=callbacks,
        verbose=0,
    )
    scored = add_scores(model, x_map, meta_map)
    metrics = score_lstm(scored)
    segment = segment_frame(scored, args)
    segment_stats = segment_metrics(segment, args)

    scored.to_csv(output_dir / "daily_lstm_high_error_scores.csv", index=False)
    metrics.to_csv(output_dir / "daily_lstm_high_error_metrics.csv", index=False)
    segment.to_csv(output_dir / "segment_2017_lstm_high_error_scores.csv", index=False)
    segment_stats.to_csv(output_dir / "segment_lstm_high_error_metrics.csv", index=False)
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)
    model.save(output_dir / "daily_lstm_high_error_filter.keras")
    write_summary(output_dir, metrics, segment_stats, args)
    output_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "predictions": str(args.predictions),
                "data": str(args.data),
                "universe": args.universe,
                "spike_threshold": args.spike_threshold,
                "window": args.window,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "feature_columns": DAILY_FEATURE_COLUMNS,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "train_sequences": int(len(x_map["train"])),
                "val_sequences": int(len(x_map["val"])),
                "val_auc": float(metrics.loc[metrics["split"].eq("val"), "auc"].iloc[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
