from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MetricFn = Callable[[pd.DataFrame, str], dict[str, float]]

DEFAULT_DAILY_COVERAGE_CANDIDATES: tuple[float, ...] = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
DEFAULT_GATE_THRESHOLD_CANDIDATES: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


@dataclass(frozen=True)
class FilterSignalSelectionParams:
    gate_threshold: float
    daily_coverage: float
    move_daily_coverage: float
    move_daily_ic_coverage: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def daily_top_fraction_mask(
    frame: pd.DataFrame,
    coverage: float,
    score_column: str,
    *,
    split_column: str = "split",
    date_column: str = "Date",
) -> pd.Series:
    if coverage <= 0.0 or coverage > 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    required = {split_column, date_column, score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns for daily top selection: {sorted(missing)}")

    def mark_top(group: pd.Series) -> pd.Series:
        keep_count = max(1, int(np.ceil(len(group) * coverage)))
        ranks = group.rank(method="first", ascending=False)
        return ranks <= keep_count

    return frame.groupby([split_column, date_column], sort=False)[score_column].transform(mark_top).astype(bool)


def apply_daily_top_selection(
    frame: pd.DataFrame,
    coverage: float,
    output_column: str,
    score_column: str,
    *,
    base_prediction_column: str = "base_prediction",
    split_column: str = "split",
    date_column: str = "Date",
) -> None:
    active = daily_top_fraction_mask(
        frame,
        coverage,
        score_column,
        split_column=split_column,
        date_column=date_column,
    )
    frame[output_column] = np.where(active, frame[base_prediction_column], 0.0)


def choose_gate_threshold(
    train_frame: pd.DataFrame,
    metric_fn: MetricFn,
    *,
    threshold_candidates: Sequence[float] = DEFAULT_GATE_THRESHOLD_CANDIDATES,
    metric_name: str = "rel_score",
    base_prediction_column: str = "base_prediction",
    filter_probability_column: str = "filter_probability",
) -> float:
    best_threshold = float(threshold_candidates[0])
    best_score = -np.inf
    for threshold in threshold_candidates:
        col = f"gate_{threshold:.2f}"
        train_frame[col] = np.where(
            train_frame[filter_probability_column] >= threshold,
            train_frame[base_prediction_column],
            0.0,
        )
        score = metric_fn(train_frame, col).get(metric_name, float("nan"))
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def choose_daily_coverage(
    train_frame: pd.DataFrame,
    score_column: str,
    metric_fn: MetricFn,
    *,
    coverage_candidates: Sequence[float] = DEFAULT_DAILY_COVERAGE_CANDIDATES,
    metric_name: str = "rel_score",
    base_prediction_column: str = "base_prediction",
    split_column: str = "split",
    date_column: str = "Date",
) -> float:
    best_coverage = float(coverage_candidates[0])
    best_score = -np.inf
    for coverage in coverage_candidates:
        col = f"daily_top_{str(float(coverage)).replace('.', '_')}"
        apply_daily_top_selection(
            train_frame,
            float(coverage),
            col,
            score_column,
            base_prediction_column=base_prediction_column,
            split_column=split_column,
            date_column=date_column,
        )
        score = metric_fn(train_frame, col).get(metric_name, float("nan"))
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_coverage = float(coverage)
    return best_coverage


def fit_filter_signal_selection(
    scored: pd.DataFrame,
    metric_fn: MetricFn,
    *,
    train_split: str = "train",
    threshold_candidates: Sequence[float] = DEFAULT_GATE_THRESHOLD_CANDIDATES,
    coverage_candidates: Sequence[float] = DEFAULT_DAILY_COVERAGE_CANDIDATES,
    base_prediction_column: str = "base_prediction",
    filter_probability_column: str = "filter_probability",
    expected_move_column: str = "filter_expected_move",
    split_column: str = "split",
    date_column: str = "Date",
) -> FilterSignalSelectionParams:
    required = {base_prediction_column, filter_probability_column, split_column, date_column}
    missing = required.difference(scored.columns)
    if missing:
        raise KeyError(f"Missing required columns for filter signal selection: {sorted(missing)}")

    train_frame = scored.loc[scored[split_column] == train_split].copy()
    if train_frame.empty:
        raise ValueError(f"No rows found for train_split={train_split!r}")

    train_frame[expected_move_column] = (
        train_frame[base_prediction_column].abs() * train_frame[filter_probability_column]
    )
    return FilterSignalSelectionParams(
        gate_threshold=choose_gate_threshold(
            train_frame.copy(),
            metric_fn,
            threshold_candidates=threshold_candidates,
            base_prediction_column=base_prediction_column,
            filter_probability_column=filter_probability_column,
        ),
        daily_coverage=choose_daily_coverage(
            train_frame.copy(),
            filter_probability_column,
            metric_fn,
            coverage_candidates=coverage_candidates,
            base_prediction_column=base_prediction_column,
            split_column=split_column,
            date_column=date_column,
        ),
        move_daily_coverage=choose_daily_coverage(
            train_frame.copy(),
            expected_move_column,
            metric_fn,
            coverage_candidates=coverage_candidates,
            base_prediction_column=base_prediction_column,
            split_column=split_column,
            date_column=date_column,
        ),
        move_daily_ic_coverage=choose_daily_coverage(
            train_frame.copy(),
            expected_move_column,
            metric_fn,
            coverage_candidates=coverage_candidates,
            metric_name="mean_daily_ic",
            base_prediction_column=base_prediction_column,
            split_column=split_column,
            date_column=date_column,
        ),
    )


def apply_filter_signal_selection(
    scored: pd.DataFrame,
    params: FilterSignalSelectionParams,
    *,
    base_prediction_column: str = "base_prediction",
    filter_probability_column: str = "filter_probability",
    expected_move_column: str = "filter_expected_move",
    risk_scale_column: str = "market_risk_scale",
    split_column: str = "split",
    date_column: str = "Date",
) -> tuple[pd.DataFrame, dict[str, str]]:
    out = scored.copy()
    out[expected_move_column] = out[base_prediction_column].abs() * out[filter_probability_column]
    if risk_scale_column not in out.columns:
        out[risk_scale_column] = 1.0

    out["prediction_base"] = out[base_prediction_column]
    out["prediction_probability_scaled"] = out[base_prediction_column] * out[filter_probability_column]
    out["prediction_gate"] = np.where(
        out[filter_probability_column] >= params.gate_threshold,
        out[base_prediction_column],
        0.0,
    )
    apply_daily_top_selection(
        out,
        0.10,
        "prediction_daily_top_10",
        filter_probability_column,
        base_prediction_column=base_prediction_column,
        split_column=split_column,
        date_column=date_column,
    )
    apply_daily_top_selection(
        out,
        0.20,
        "prediction_daily_top_20",
        filter_probability_column,
        base_prediction_column=base_prediction_column,
        split_column=split_column,
        date_column=date_column,
    )
    apply_daily_top_selection(
        out,
        params.daily_coverage,
        "prediction_daily_top_train_selected",
        filter_probability_column,
        base_prediction_column=base_prediction_column,
        split_column=split_column,
        date_column=date_column,
    )
    apply_daily_top_selection(
        out,
        0.10,
        "prediction_move_top_10",
        expected_move_column,
        base_prediction_column=base_prediction_column,
        split_column=split_column,
        date_column=date_column,
    )
    apply_daily_top_selection(
        out,
        0.20,
        "prediction_move_top_20",
        expected_move_column,
        base_prediction_column=base_prediction_column,
        split_column=split_column,
        date_column=date_column,
    )
    apply_daily_top_selection(
        out,
        params.move_daily_coverage,
        "prediction_move_top_train_selected",
        expected_move_column,
        base_prediction_column=base_prediction_column,
        split_column=split_column,
        date_column=date_column,
    )
    apply_daily_top_selection(
        out,
        params.move_daily_ic_coverage,
        "prediction_move_top_train_ic_selected",
        expected_move_column,
        base_prediction_column=base_prediction_column,
        split_column=split_column,
        date_column=date_column,
    )
    out["prediction_probability_risk_scaled"] = (
        out[base_prediction_column] * out[filter_probability_column] * out[risk_scale_column]
    )
    out["prediction_gate_risk_scaled"] = out["prediction_gate"] * out[risk_scale_column]
    out["prediction_daily_top_train_selected_risk_scaled"] = (
        out["prediction_daily_top_train_selected"] * out[risk_scale_column]
    )
    out["prediction_move_top_train_selected_risk_scaled"] = (
        out["prediction_move_top_train_selected"] * out[risk_scale_column]
    )
    out["prediction_move_top_train_ic_selected_risk_scaled"] = (
        out["prediction_move_top_train_ic_selected"] * out[risk_scale_column]
    )

    candidate_columns = {
        "base": "prediction_base",
        "probability_scaled": "prediction_probability_scaled",
        "gate": "prediction_gate",
        "daily_top_10": "prediction_daily_top_10",
        "daily_top_20": "prediction_daily_top_20",
        "daily_top_train_selected": "prediction_daily_top_train_selected",
        "move_top_10": "prediction_move_top_10",
        "move_top_20": "prediction_move_top_20",
        "move_top_train_selected": "prediction_move_top_train_selected",
        "move_top_train_ic_selected": "prediction_move_top_train_ic_selected",
        "probability_risk_scaled": "prediction_probability_risk_scaled",
        "gate_risk_scaled": "prediction_gate_risk_scaled",
        "daily_top_train_selected_risk_scaled": "prediction_daily_top_train_selected_risk_scaled",
        "move_top_train_selected_risk_scaled": "prediction_move_top_train_selected_risk_scaled",
        "move_top_train_ic_selected_risk_scaled": "prediction_move_top_train_ic_selected_risk_scaled",
    }
    return out, candidate_columns


def candidate_coverage_bundle(
    frame: pd.DataFrame,
    candidate_columns: dict[str, str],
    *,
    split_column: str = "split",
    actual_column: str = "actual_aligned",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, group in frame.groupby(split_column, sort=True):
        for candidate, column in candidate_columns.items():
            active = group[column].to_numpy(dtype=float) != 0.0
            active_group = group.loc[active]
            active_hit = (
                active_group[column].to_numpy(dtype=float) * active_group[actual_column].to_numpy(dtype=float)
            ) > 0.0
            rows.append(
                {
                    "split": split,
                    "candidate": candidate,
                    "coverage": float(np.mean(active)) if len(group) else float("nan"),
                    "active_hit_rate": float(np.mean(active_hit)) if len(active_group) else float("nan"),
                }
            )
    return rows
