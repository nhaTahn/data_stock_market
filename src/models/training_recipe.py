from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_SUMMARY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "stock_sector_search_summary.csv"
)
DEFAULT_CONTEXT_FEATURES = (
    "alpha_sector",
    "market_leader_return",
    "vnindex_return",
    "a_d_ratio",
    "day_of_week",
    "rsi_14",
)


@dataclass(frozen=True)
class TrainingRecipe:
    source: str
    sector: str | None
    selected_stocks: tuple[str, ...]
    feature_columns: tuple[str, ...]
    stock_summary: list[dict[str, object]]
    feature_summary: list[dict[str, object]]


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for column in (
        "best_val_rel_score",
        "best_test_rel_score",
        "best_test_val_rel_score",
    ):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def load_stock_search_summary(summary_path: Path = DEFAULT_SEARCH_SUMMARY_PATH) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Stock search summary not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    required_columns = {
        "stock",
        "sector",
        "best_by_val",
        "best_val_rel_score",
    }
    missing = sorted(required_columns.difference(summary.columns))
    if missing:
        raise ValueError(f"Stock search summary is missing columns: {missing}")
    return _coerce_numeric_columns(summary)


def _split_feature_combo(raw_value: object) -> tuple[str, ...]:
    if not isinstance(raw_value, str):
        return ()
    features = [item.strip() for item in raw_value.split(",") if item.strip()]
    return tuple(dict.fromkeys(features))


def select_stock_rows(
    summary: pd.DataFrame,
    *,
    sector: str | None = None,
    stocks: tuple[str, ...] | None = None,
    min_best_val_rel_score: float = 0.03,
    max_stocks: int | None = None,
) -> pd.DataFrame:
    work = summary.copy()
    if sector:
        work = work[work["sector"] == sector].copy()
    if stocks:
        requested = [stock for stock in stocks if stock]
        if not requested:
            raise ValueError("Explicit stocks list is empty.")
        work = work[work["stock"].isin(requested)].copy()
        order_map = {stock: idx for idx, stock in enumerate(requested)}
        work["_requested_order"] = work["stock"].map(order_map)
        work = work.sort_values("_requested_order", kind="stable").drop(columns="_requested_order")
    else:
        work = work[work["best_val_rel_score"].fillna(float("-inf")) >= float(min_best_val_rel_score)].copy()
        work = work.sort_values(
            ["best_val_rel_score", "best_test_rel_score", "stock"],
            ascending=[False, False, True],
            kind="stable",
        )

    if work.empty:
        scope = f"sector '{sector}'" if sector else "the requested stocks"
        raise ValueError(f"No eligible stocks found in search summary for {scope}.")

    if max_stocks is not None:
        work = work.head(max(1, int(max_stocks))).copy()
    return work.reset_index(drop=True)


def rank_features_from_stock_rows(
    stock_rows: pd.DataFrame,
    *,
    top_k: int = 10,
    available_columns: set[str] | None = None,
    extra_features: tuple[str, ...] = DEFAULT_CONTEXT_FEATURES,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    feature_scores: dict[str, dict[str, object]] = {}
    for row in stock_rows.itertuples():
        weight = float(max(getattr(row, "best_val_rel_score", 0.0) or 0.0, 0.0))
        for feature in _split_feature_combo(getattr(row, "best_by_val", None)):
            if available_columns is not None and feature not in available_columns:
                continue
            stats = feature_scores.setdefault(
                feature,
                {
                    "feature": feature,
                    "score": 0.0,
                    "stocks_hit": set(),
                    "max_best_val_rel_score": float("-inf"),
                },
            )
            stats["score"] += weight
            stats["stocks_hit"].add(getattr(row, "stock"))
            stats["max_best_val_rel_score"] = max(
                float(stats["max_best_val_rel_score"]),
                float(weight),
            )

    ranked = sorted(
        (
            {
                "feature": feature,
                "score": float(stats["score"]),
                "stocks_hit": len(stats["stocks_hit"]),
                "max_best_val_rel_score": float(stats["max_best_val_rel_score"]),
            }
            for feature, stats in feature_scores.items()
        ),
        key=lambda item: (
            item["score"],
            item["stocks_hit"],
            item["max_best_val_rel_score"],
            item["feature"],
        ),
        reverse=True,
    )

    selected = [item["feature"] for item in ranked[: max(1, int(top_k))]]
    for feature in extra_features:
        if available_columns is not None and feature not in available_columns:
            continue
        if feature not in selected:
            selected.append(feature)

    return tuple(selected), ranked


def build_training_recipe(
    *,
    summary_path: Path = DEFAULT_SEARCH_SUMMARY_PATH,
    sector: str | None = None,
    stocks: tuple[str, ...] | None = None,
    min_best_val_rel_score: float = 0.03,
    max_stocks: int | None = None,
    top_features: int = 10,
    available_columns: set[str] | None = None,
    extra_features: tuple[str, ...] = DEFAULT_CONTEXT_FEATURES,
) -> TrainingRecipe:
    summary = load_stock_search_summary(summary_path)
    selected_rows = select_stock_rows(
        summary,
        sector=sector,
        stocks=stocks,
        min_best_val_rel_score=min_best_val_rel_score,
        max_stocks=max_stocks,
    )
    feature_columns, feature_summary = rank_features_from_stock_rows(
        selected_rows,
        top_k=top_features,
        available_columns=available_columns,
        extra_features=extra_features,
    )
    return TrainingRecipe(
        source=str(summary_path),
        sector=sector,
        selected_stocks=tuple(selected_rows["stock"].astype(str).tolist()),
        feature_columns=feature_columns,
        stock_summary=selected_rows.to_dict(orient="records"),
        feature_summary=feature_summary,
    )
