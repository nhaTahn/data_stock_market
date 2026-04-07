from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class FischerKraussScaler:
    mean: float
    std: float
    input_column: str = "fk_daily_return"
    output_column: str = "fk_daily_return_std"


def resolve_price_column(df: pd.DataFrame) -> str:
    if "adjust" in df.columns:
        return "adjust"
    if "close" in df.columns:
        return "close"
    raise ValueError("Dataset must contain either 'adjust' or 'close' to run the Fischer-Krauss benchmark.")


def prepare_fischer_krauss_frame(
    df: pd.DataFrame,
    price_column: str | None = None,
) -> pd.DataFrame:
    """Build the daily-return frame and the cross-sectional binary target."""
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.sort_values(["code", "Date"]).reset_index(drop=True)
    price_column = price_column or resolve_price_column(work)

    work["fk_daily_return"] = work.groupby("code")[price_column].pct_change()
    work["fk_next_return"] = work.groupby("code")["fk_daily_return"].shift(-1)
    median_next_return = (
        work.groupby("Date", sort=False)["fk_next_return"]
        .median()
        .rename("fk_cross_sectional_median")
        .reset_index()
    )
    work = work.merge(median_next_return, on="Date", how="left")
    work["fk_target_class"] = (
        work["fk_next_return"] >= work["fk_cross_sectional_median"]
    ).astype(np.float32)
    return work


def resolve_fk_train_end_date(
    df: pd.DataFrame,
    validation_end_date: str,
    train_fraction: float = 0.8,
) -> pd.Timestamp:
    """Use the earliest 80% of pre-test dates as train and the rest as validation."""
    pretest_dates = np.sort(pd.to_datetime(df.loc[df["Date"] <= pd.Timestamp(validation_end_date), "Date"]).unique())
    if len(pretest_dates) < 2:
        raise ValueError("Not enough pre-test dates to build Fischer-Krauss train/validation split.")
    cutoff_idx = int(np.floor(len(pretest_dates) * train_fraction))
    cutoff_idx = min(max(cutoff_idx, 1), len(pretest_dates) - 1)
    return pd.Timestamp(pretest_dates[cutoff_idx - 1])


def fit_fischer_krauss_scaler(
    df: pd.DataFrame,
    train_end_date: pd.Timestamp,
    input_column: str = "fk_daily_return",
) -> FischerKraussScaler:
    train_values = (
        df.loc[df["Date"] <= train_end_date, input_column]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float32)
    )
    if len(train_values) == 0:
        raise ValueError("No valid training returns found for Fischer-Krauss scaling.")
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    return FischerKraussScaler(mean=mean, std=std, input_column=input_column)


def apply_fischer_krauss_scaler(
    df: pd.DataFrame,
    scaler: FischerKraussScaler,
) -> pd.DataFrame:
    work = df.copy()
    values = work[scaler.input_column].to_numpy(dtype=np.float32)
    work[scaler.output_column] = ((values - scaler.mean) / scaler.std).astype(np.float32)
    return work


def build_fischer_krauss_sequences(
    df: pd.DataFrame,
    window_size: int = 240,
    input_column: str = "fk_daily_return_std",
    target_column: str = "fk_target_class",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create overlapping 240-day univariate return sequences per stock."""
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict[str, object]] = []
    required_cols = [input_column, target_column, "fk_next_return", "Date", "code"]

    for code, group in df.sort_values(["code", "Date"]).groupby("code"):
        group = group.dropna(subset=required_cols).reset_index(drop=True)
        if len(group) < window_size:
            continue

        feature_values = group[[input_column]].to_numpy(dtype=np.float32)
        target_values = group[target_column].to_numpy(dtype=np.int32)
        next_returns = group["fk_next_return"].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(group["Date"])

        for end_idx in range(window_size - 1, len(group)):
            start_idx = end_idx - window_size + 1
            x_list.append(feature_values[start_idx : end_idx + 1])
            y_list.append(int(target_values[end_idx]))
            meta_rows.append(
                {
                    "code": str(code),
                    "Date": dates.iloc[end_idx],
                    "actual_return": float(next_returns[end_idx]),
                    "target_class": int(target_values[end_idx]),
                }
            )

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    meta = pd.DataFrame(meta_rows)
    return x, y, meta


def split_fischer_krauss_sequences(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    train_end_date: pd.Timestamp,
    validation_end_date: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    dates = pd.to_datetime(meta["Date"])
    val_end_ts = pd.Timestamp(validation_end_date)
    train_mask = dates <= train_end_date
    val_mask = (dates > train_end_date) & (dates <= val_end_ts)
    test_mask = dates > val_end_ts
    return {
        "train": (x[train_mask], y[train_mask], meta.loc[train_mask].reset_index(drop=True)),
        "val": (x[val_mask], y[val_mask], meta.loc[val_mask].reset_index(drop=True)),
        "test": (x[test_mask], y[test_mask], meta.loc[test_mask].reset_index(drop=True)),
    }


def build_fischer_krauss_model(
    window_size: int = 240,
    hidden_units: int = 25,
    dropout: float = 0.16,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Fischer & Krauss benchmark: 240x1 input, one LSTM(25), softmax(2), RMSprop."""
    inputs = layers.Input(shape=(window_size, 1), name="fk_input")
    x = layers.LSTM(
        hidden_units,
        dropout=dropout,
        recurrent_dropout=dropout,
        name="fk_lstm",
    )(inputs)
    outputs = layers.Dense(2, activation="softmax", name="fk_softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_fischer_krauss_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    window_size: int = 240,
    hidden_units: int = 25,
    dropout: float = 0.16,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 1000,
    patience: int = 10,
) -> tuple[keras.Model, keras.callbacks.History]:
    model = build_fischer_krauss_model(
        window_size=window_size,
        hidden_units=hidden_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train,
        keras.utils.to_categorical(y_train, num_classes=2),
        validation_data=(x_val, keras.utils.to_categorical(y_val, num_classes=2)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )
    return model, history


def predict_fischer_krauss_probabilities(model: keras.Model, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict(x, verbose=0)
    return np.asarray(probabilities, dtype=np.float32)


def probability_to_score(probabilities: np.ndarray) -> np.ndarray:
    prob_class_1 = np.asarray(probabilities, dtype=np.float32)[:, 1]
    return (2.0 * prob_class_1 - 1.0).astype(np.float32)


def compute_fischer_krauss_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    prob_class_1 = probabilities[:, 1]
    pred_class = (prob_class_1 >= 0.5).astype(np.int32)

    accuracy = float(np.mean(pred_class == y_true)) if len(y_true) else float("nan")
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = float(np.mean(pred_class[pos_mask] == 1)) if np.any(pos_mask) else float("nan")
    tnr = float(np.mean(pred_class[neg_mask] == 0)) if np.any(neg_mask) else float("nan")
    balanced_accuracy = float(np.nanmean([tpr, tnr]))
    clipped = np.clip(prob_class_1, 1e-7, 1.0 - 1e-7)
    log_loss = float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1.0 - clipped)))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred_class)),
        "mean_prob_class_1": float(np.mean(prob_class_1)),
        "log_loss": log_loss,
    }


def build_long_short_portfolio_returns(
    meta: pd.DataFrame,
    probabilities: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    """Rank stocks by class-1 probability and form equal-weight long/short legs."""
    if meta.empty:
        return pd.DataFrame(columns=["Date", "long_return", "short_return", "long_short_return", "selected_k"])

    work = meta.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["prob_class_1"] = np.asarray(probabilities, dtype=np.float32)[:, 1]
    rows: list[dict[str, float | int | pd.Timestamp]] = []

    for date, day_df in work.groupby("Date", sort=True):
        day_df = day_df.sort_values("prob_class_1", ascending=False).reset_index(drop=True)
        k = min(top_k, len(day_df) // 2)
        if k <= 0:
            continue
        long_df = day_df.head(k)
        short_df = day_df.tail(k)
        long_return = float(long_df["actual_return"].mean())
        short_return = float(short_df["actual_return"].mean())
        rows.append(
            {
                "Date": date,
                "long_return": long_return,
                "short_return": short_return,
                "long_short_return": long_return - short_return,
                "selected_k": int(k),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["equity_curve"] = (1.0 + result["long_short_return"]).cumprod()
    return result


def summarize_long_short_portfolio(portfolio_returns: pd.DataFrame) -> dict[str, float]:
    if portfolio_returns.empty:
        return {
            "num_days": 0,
            "avg_daily_return": float("nan"),
            "vol_daily_return": float("nan"),
            "sharpe_annualized": float("nan"),
            "final_equity": float("nan"),
            "max_drawdown": float("nan"),
        }

    daily = portfolio_returns["long_short_return"].to_numpy(dtype=np.float32)
    equity = portfolio_returns["equity_curve"].to_numpy(dtype=np.float32)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(running_max, 1e-8) - 1.0
    daily_std = float(np.std(daily))
    sharpe = float(np.mean(daily) / daily_std * np.sqrt(252.0)) if daily_std > 0 else float("nan")
    return {
        "num_days": int(len(portfolio_returns)),
        "avg_daily_return": float(np.mean(daily)),
        "vol_daily_return": daily_std,
        "sharpe_annualized": sharpe,
        "final_equity": float(equity[-1]),
        "max_drawdown": float(np.min(drawdown)),
    }
