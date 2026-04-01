from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tf_lstm.config import TARGET_COLUMN
from tf_lstm.data import load_dataset, split_dataset
from tf_lstm.metrics import compute_metrics

try:
    from sklearn.naive_bayes import GaussianNB
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "scikit-learn is not installed in the current Python environment.\n"
        "Use a local virtualenv with scikit-learn, then rerun this script.\n"
        "Example: pip install scikit-learn"
    ) from exc

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "statsmodels is not installed in the current Python environment.\n"
        "Use a local virtualenv with statsmodels, then rerun this script.\n"
        "Example: pip install statsmodels"
    ) from exc

try:
    from arch import arch_model
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "arch is not installed in the current Python environment.\n"
        "Use a local virtualenv with arch, then rerun this script.\n"
        "Example: pip install arch"
    ) from exc


@dataclass
class ModelRunResult:
    metrics: dict[str, dict[str, float]]
    evaluated_samples: dict[str, int]
    extra: dict[str, object]


def _zero_baseline_metrics(targets: np.ndarray) -> dict[str, float]:
    return compute_metrics(targets, np.zeros_like(targets, dtype=np.float32))


def run_gaussian_nb(bundle) -> ModelRunResult:
    x_train = bundle.train_seq.features.reshape(len(bundle.train_seq.features), -1)
    x_val = bundle.val_seq.features.reshape(len(bundle.val_seq.features), -1)
    x_test = bundle.test_seq.features.reshape(len(bundle.test_seq.features), -1)

    y_train = bundle.train_targets
    labels_train = np.sign(y_train).astype(int)

    model = GaussianNB()
    model.fit(x_train, labels_train)

    class_mean_return = {int(cls): float(y_train[labels_train == cls].mean()) for cls in np.unique(labels_train)}

    def predict_expected_return(features: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(features)
        class_means = np.array([class_mean_return.get(int(cls), 0.0) for cls in model.classes_], dtype=np.float32)
        return proba @ class_means

    train_pred = predict_expected_return(x_train)
    val_pred = predict_expected_return(x_val)
    test_pred = predict_expected_return(x_test)

    metrics = {
        "train": compute_metrics(bundle.train_targets, train_pred),
        "val": compute_metrics(bundle.val_targets, val_pred),
        "test": compute_metrics(bundle.test_targets, test_pred),
    }
    return ModelRunResult(
        metrics=metrics,
        evaluated_samples={
            "train": len(bundle.train_targets),
            "val": len(bundle.val_targets),
            "test": len(bundle.test_targets),
        },
        extra={
            "classes": [int(cls) for cls in model.classes_],
            "class_mean_return": class_mean_return,
        },
    )


def _recursive_arima_for_code(
    train_values: np.ndarray,
    val_values: np.ndarray,
    test_values: np.ndarray,
    order: tuple[int, int, int],
    min_history: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    outputs = {split: (np.array([], dtype=np.float32), np.array([], dtype=np.float32)) for split in ["train", "val", "test"]}
    if len(train_values) <= min_history:
        return outputs

    try:
        result = ARIMA(
            train_values[:min_history],
            order=order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
    except Exception:
        return outputs

    train_preds: list[float] = []
    train_true: list[float] = []
    for actual in train_values[min_history:]:
        try:
            pred = float(result.forecast(steps=1)[0])
        except Exception:
            pred = 0.0
        train_preds.append(pred)
        train_true.append(float(actual))
        try:
            result = result.append([actual], refit=False)
        except Exception:
            result = ARIMA(
                np.array(train_true[-min_history:], dtype=np.float32),
                order=order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()

    def roll_forward(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds: list[float] = []
        true: list[float] = []
        nonlocal result
        for actual in observations:
            try:
                pred = float(result.forecast(steps=1)[0])
            except Exception:
                pred = 0.0
            preds.append(pred)
            true.append(float(actual))
            try:
                result = result.append([actual], refit=False)
            except Exception:
                break
        return np.array(true, dtype=np.float32), np.array(preds, dtype=np.float32)

    outputs["train"] = (np.array(train_true, dtype=np.float32), np.array(train_preds, dtype=np.float32))
    outputs["val"] = roll_forward(val_values)
    outputs["test"] = roll_forward(test_values)
    return outputs


def run_arima(
    data_path: Path,
    train_end: str,
    val_end: str,
    order: tuple[int, int, int],
    min_history: int,
    max_tickers: int | None = None,
) -> ModelRunResult:
    df, _, _ = load_dataset(data_path, feature_groups=["base"])
    if max_tickers is not None:
        selected_codes = sorted(df["code"].unique())[:max_tickers]
        df = df[df["code"].isin(selected_codes)].copy()
    train_df, val_df, test_df = split_dataset(df, train_end, val_end)

    pooled: dict[str, list[np.ndarray]] = {split: [] for split in ["train", "val", "test"]}
    pooled_pred: dict[str, list[np.ndarray]] = {split: [] for split in ["train", "val", "test"]}
    for code in sorted(df["code"].unique()):
        train_values = train_df.loc[train_df["code"] == code, TARGET_COLUMN].to_numpy(dtype=np.float32)
        val_values = val_df.loc[val_df["code"] == code, TARGET_COLUMN].to_numpy(dtype=np.float32)
        test_values = test_df.loc[test_df["code"] == code, TARGET_COLUMN].to_numpy(dtype=np.float32)
        code_outputs = _recursive_arima_for_code(train_values, val_values, test_values, order, min_history)
        for split, (y_true, y_pred) in code_outputs.items():
            if len(y_true) == 0:
                continue
            pooled[split].append(y_true)
            pooled_pred[split].append(y_pred)

    metrics: dict[str, dict[str, float]] = {}
    samples: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        y_true = np.concatenate(pooled[split]) if pooled[split] else np.array([], dtype=np.float32)
        y_pred = np.concatenate(pooled_pred[split]) if pooled_pred[split] else np.array([], dtype=np.float32)
        metrics[split] = compute_metrics(y_true, y_pred) if len(y_true) else _zero_baseline_metrics(np.array([0.0], dtype=np.float32))
        samples[split] = int(len(y_true))

    return ModelRunResult(
        metrics=metrics,
        evaluated_samples=samples,
        extra={"order": order, "min_history": min_history},
    )


def _fit_garch(history: np.ndarray):
    if len(history) < 20 or np.allclose(np.std(history), 0.0):
        return None
    try:
        model = arch_model(
            history,
            mean="AR",
            lags=1,
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        return model.fit(disp="off", show_warning=False)
    except Exception:
        return None


def _recursive_garch_for_code(
    train_values: np.ndarray,
    val_values: np.ndarray,
    test_values: np.ndarray,
    min_history: int,
    refit_interval: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    outputs = {split: (np.array([], dtype=np.float32), np.array([], dtype=np.float32)) for split in ["train", "val", "test"]}
    if len(train_values) <= min_history:
        return outputs

    history = list(train_values[:min_history].astype(np.float32))
    fitted = _fit_garch(np.array(history, dtype=np.float32))
    steps_since_refit = 0

    def roll_forward(observations: np.ndarray, allow_history_growth: bool = True) -> tuple[np.ndarray, np.ndarray]:
        nonlocal fitted, steps_since_refit, history
        preds: list[float] = []
        true: list[float] = []
        for actual in observations:
            if fitted is None or steps_since_refit >= refit_interval:
                fitted = _fit_garch(np.array(history, dtype=np.float32))
                steps_since_refit = 0
            if fitted is None:
                pred = 0.0
            else:
                try:
                    pred = float(fitted.forecast(horizon=1, reindex=False).mean.iloc[-1, 0])
                except Exception:
                    pred = 0.0
            preds.append(pred)
            true.append(float(actual))
            if allow_history_growth:
                history.append(float(actual))
            steps_since_refit += 1
        return np.array(true, dtype=np.float32), np.array(preds, dtype=np.float32)

    outputs["train"] = roll_forward(train_values[min_history:])
    outputs["val"] = roll_forward(val_values)
    outputs["test"] = roll_forward(test_values)
    return outputs


def run_garch(
    data_path: Path,
    train_end: str,
    val_end: str,
    min_history: int,
    refit_interval: int,
    max_tickers: int | None = None,
) -> ModelRunResult:
    df, _, _ = load_dataset(data_path, feature_groups=["base"])
    if max_tickers is not None:
        selected_codes = sorted(df["code"].unique())[:max_tickers]
        df = df[df["code"].isin(selected_codes)].copy()
    train_df, val_df, test_df = split_dataset(df, train_end, val_end)

    pooled: dict[str, list[np.ndarray]] = {split: [] for split in ["train", "val", "test"]}
    pooled_pred: dict[str, list[np.ndarray]] = {split: [] for split in ["train", "val", "test"]}
    for code in sorted(df["code"].unique()):
        train_values = train_df.loc[train_df["code"] == code, TARGET_COLUMN].to_numpy(dtype=np.float32)
        val_values = val_df.loc[val_df["code"] == code, TARGET_COLUMN].to_numpy(dtype=np.float32)
        test_values = test_df.loc[test_df["code"] == code, TARGET_COLUMN].to_numpy(dtype=np.float32)
        code_outputs = _recursive_garch_for_code(train_values, val_values, test_values, min_history, refit_interval)
        for split, (y_true, y_pred) in code_outputs.items():
            if len(y_true) == 0:
                continue
            pooled[split].append(y_true)
            pooled_pred[split].append(y_pred)

    metrics: dict[str, dict[str, float]] = {}
    samples: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        y_true = np.concatenate(pooled[split]) if pooled[split] else np.array([], dtype=np.float32)
        y_pred = np.concatenate(pooled_pred[split]) if pooled_pred[split] else np.array([], dtype=np.float32)
        metrics[split] = compute_metrics(y_true, y_pred) if len(y_true) else _zero_baseline_metrics(np.array([0.0], dtype=np.float32))
        samples[split] = int(len(y_true))

    return ModelRunResult(
        metrics=metrics,
        evaluated_samples=samples,
        extra={"min_history": min_history, "refit_interval": refit_interval},
    )
