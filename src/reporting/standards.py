from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STANDARD_PATH = ROOT / "configs" / "reporting_standard_vn.json"


@dataclass(frozen=True)
class TimeWindow:
    split_name: str
    label: str
    start_date: str | None
    end_date: str | None
    display: str


@dataclass(frozen=True)
class ReportingStandard:
    name: str
    market: str
    target_mode: str
    split_aliases: dict[str, str]
    time_windows: dict[str, TimeWindow]
    error_quantiles: tuple[float, float]
    error_quantile_labels: tuple[str, str]
    large_error_quantile: float
    long_short_quantile: float
    default_report_splits: tuple[str, ...]
    holdout_split: str
    required_metrics: tuple[str, ...]

    def split_label(self, split_name: str) -> str:
        return self.split_aliases.get(split_name, split_name)

    def window(self, split_name: str) -> TimeWindow:
        if split_name not in self.time_windows:
            raise KeyError(f"Unknown split '{split_name}' in reporting standard '{self.name}'.")
        return self.time_windows[split_name]

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "market": self.market,
            "target_mode": self.target_mode,
            "split_aliases": self.split_aliases,
            "time_windows": {
                split_name: {
                    "label": window.label,
                    "start_date": window.start_date,
                    "end_date": window.end_date,
                    "display": window.display,
                }
                for split_name, window in self.time_windows.items()
            },
            "evaluation": {
                "error_quantiles": list(self.error_quantiles),
                "error_quantile_labels": list(self.error_quantile_labels),
                "large_error_quantile": self.large_error_quantile,
                "long_short_quantile": self.long_short_quantile,
                "default_report_splits": list(self.default_report_splits),
                "holdout_split": self.holdout_split,
                "required_metrics": list(self.required_metrics),
            },
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_reporting_standard(path: Path = DEFAULT_STANDARD_PATH) -> ReportingStandard:
    payload = _load_json(path)
    split_aliases = {str(key): str(value) for key, value in payload["split_aliases"].items()}
    time_windows = {
        split_name: TimeWindow(
            split_name=split_name,
            label=split_aliases.get(split_name, split_name),
            start_date=window.get("start_date"),
            end_date=window.get("end_date"),
            display=str(window["display"]),
        )
        for split_name, window in payload["time_windows"].items()
    }
    evaluation = payload["evaluation"]
    return ReportingStandard(
        name=str(payload["name"]),
        market=str(payload["market"]),
        target_mode=str(payload.get("defaults", {}).get("target_mode", "return")),
        split_aliases=split_aliases,
        time_windows=time_windows,
        error_quantiles=tuple(float(value) for value in evaluation.get("error_quantiles", [0.2, 0.8])),
        error_quantile_labels=tuple(str(value) for value in evaluation.get("error_quantile_labels", ["q2", "q8"])),
        large_error_quantile=float(evaluation.get("large_error_quantile", 0.95)),
        long_short_quantile=float(evaluation.get("long_short_quantile", 0.25)),
        default_report_splits=tuple(str(value) for value in evaluation.get("default_report_splits", ["train", "val"])),
        holdout_split=str(evaluation.get("holdout_split", "test")),
        required_metrics=tuple(str(value) for value in evaluation.get("required_metrics", ["rel_score", "rmse", "directional_accuracy"])),
    )


def get_default_reporting_standard() -> ReportingStandard:
    return load_reporting_standard()


def _format_display(start_date: str | None, end_date: str | None) -> str:
    start_text = "..." if start_date is None else pd.Timestamp(start_date).strftime("%d/%m/%Y")
    end_text = "now" if end_date is None else pd.Timestamp(end_date).strftime("%d/%m/%Y")
    return f"[{start_text}, {end_text}]"


def _next_day(date_text: str) -> str:
    return (pd.Timestamp(date_text) + timedelta(days=1)).strftime("%Y-%m-%d")


def validate_training_standard(
    *,
    market: str,
    target_mode: str,
    train_end_date: str,
    val_end_date: str,
    allow_nonstandard_time: bool = False,
    standard: ReportingStandard | None = None,
) -> ReportingStandard:
    standard = standard or get_default_reporting_standard()
    if allow_nonstandard_time:
        return standard
    if market != standard.market:
        raise ValueError(
            f"market must be '{standard.market}' for reporting standard '{standard.name}', got '{market}'."
        )
    if target_mode != standard.target_mode:
        raise ValueError(
            f"target_mode must be '{standard.target_mode}' for reporting standard '{standard.name}', got '{target_mode}'."
        )
    expected_train_end = standard.window("train").end_date
    expected_val_end = standard.window("val").end_date
    if train_end_date != expected_train_end or val_end_date != expected_val_end:
        raise ValueError(
            "Non-standard time split detected. "
            f"Expected train_end_date={expected_train_end}, val_end_date={expected_val_end}; "
            f"got train_end_date={train_end_date}, val_end_date={val_end_date}. "
            "Use --allow-nonstandard-time only when you intentionally want to override the VN reporting standard."
        )
    return standard


def resolve_standard_from_config(
    config_payload: dict[str, Any] | None = None,
    *,
    strict: bool = False,
) -> ReportingStandard:
    standard = get_default_reporting_standard()
    if not config_payload:
        return standard
    market = str(config_payload.get("market", standard.market))
    target_mode = str(config_payload.get("target_mode", standard.target_mode))
    train_end_date = str(config_payload.get("train_end_date", standard.window("train").end_date))
    val_end_date = str(config_payload.get("val_end_date", standard.window("val").end_date))
    if strict:
        validate_training_standard(
            market=market,
            target_mode=target_mode,
            train_end_date=train_end_date,
            val_end_date=val_end_date,
            allow_nonstandard_time=bool(config_payload.get("allow_nonstandard_time", False)),
            standard=standard,
        )
        return standard

    split_aliases = {
        str(key): str(value)
        for key, value in config_payload.get("split_aliases", standard.split_aliases).items()
    }
    if (
        market == standard.market
        and target_mode == standard.target_mode
        and train_end_date == standard.window("train").end_date
        and val_end_date == standard.window("val").end_date
    ):
        return standard

    train_window = TimeWindow(
        split_name="train",
        label=split_aliases.get("train", "train"),
        start_date=config_payload.get("start_date"),
        end_date=train_end_date,
        display=_format_display(config_payload.get("start_date"), train_end_date),
    )
    val_window = TimeWindow(
        split_name="val",
        label=split_aliases.get("val", standard.split_label("val")),
        start_date=_next_day(train_end_date),
        end_date=val_end_date,
        display=_format_display(_next_day(train_end_date), val_end_date),
    )
    test_window = TimeWindow(
        split_name="test",
        label=split_aliases.get("test", standard.split_label("test")),
        start_date=_next_day(val_end_date),
        end_date=None,
        display=_format_display(_next_day(val_end_date), None),
    )
    return ReportingStandard(
        name=str(config_payload.get("report_standard", f"derived_{market.lower()}_report")),
        market=market,
        target_mode=target_mode,
        split_aliases=split_aliases,
        time_windows={
            "train": train_window,
            "val": val_window,
            "test": test_window,
        },
        error_quantiles=standard.error_quantiles,
        error_quantile_labels=standard.error_quantile_labels,
        large_error_quantile=standard.large_error_quantile,
        long_short_quantile=standard.long_short_quantile,
        default_report_splits=standard.default_report_splits,
        holdout_split=standard.holdout_split,
        required_metrics=standard.required_metrics,
    )
