from __future__ import annotations

import numpy as np
import pandas as pd

from fk_lstm_classifier.data import (
    ADV20_VALUE_COLUMN,
    CLEAN_PROFILE_COLUMN,
    HARD_ISSUE_COLUMN,
    LIMIT_DOWN_LIKE_COLUMN,
    LIMIT_UP_LIKE_COLUMN,
    PANEL_ROW_ID_COLUMN,
    PROFILE_EVENT_COLUMN,
    SequenceDataset,
    VALUE_TRADED_COLUMN,
)


def compute_classification_metrics(labels: np.ndarray, prob_class_1: np.ndarray) -> dict[str, float]:
    predictions = (prob_class_1 >= 0.5).astype(np.int32)
    accuracy = float(np.mean(predictions == labels))

    positives = predictions == 1
    true_positives = float(np.sum((labels == 1) & positives))
    false_positives = float(np.sum((labels == 0) & positives))
    false_negatives = float(np.sum((labels == 1) & (~positives)))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0

    rank = pd.Series(prob_class_1).rank(method="average")
    positives_rank_sum = float(rank[labels == 1].sum())
    n_pos = float(np.sum(labels == 1))
    n_neg = float(np.sum(labels == 0))
    if n_pos > 0 and n_neg > 0:
        auc = (positives_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    else:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc_rank": float(auc),
    }


def build_prediction_frame(
    dataset: SequenceDataset,
    predicted_probabilities: np.ndarray,
    panel_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prediction_frame = pd.DataFrame(
        {
            "anchor_date": pd.to_datetime(dataset.anchor_dates),
            "realized_date": pd.to_datetime(dataset.realized_dates),
            "code": dataset.codes,
            "target_class": dataset.labels,
            "next_return": dataset.next_returns,
            "prob_class_0": predicted_probabilities[:, 0],
            "prob_class_1": predicted_probabilities[:, 1],
            PANEL_ROW_ID_COLUMN: dataset.panel_row_ids,
        }
    )
    if panel_frame is not None:
        metadata_columns = [
            PANEL_ROW_ID_COLUMN,
            VALUE_TRADED_COLUMN,
            ADV20_VALUE_COLUMN,
            LIMIT_UP_LIKE_COLUMN,
            LIMIT_DOWN_LIKE_COLUMN,
            HARD_ISSUE_COLUMN,
            PROFILE_EVENT_COLUMN,
            CLEAN_PROFILE_COLUMN,
        ]
        available_columns = [column for column in metadata_columns if column in panel_frame.columns]
        prediction_frame = prediction_frame.merge(
            panel_frame[available_columns].drop_duplicates(subset=[PANEL_ROW_ID_COLUMN]),
            on=PANEL_ROW_ID_COLUMN,
            how="left",
        )
    return prediction_frame.sort_values(["realized_date", "prob_class_1"], ascending=[True, False]).reset_index(drop=True)


def _apply_position_caps(
    selected: pd.DataFrame,
    gross_target: float,
    max_position_adv_fraction: float | None,
    portfolio_notional: float | None,
) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=np.float64)

    equal_weights = pd.Series(gross_target / len(selected), index=selected.index, dtype=np.float64)
    if max_position_adv_fraction is None or portfolio_notional is None or ADV20_VALUE_COLUMN not in selected.columns:
        return equal_weights

    adv20_value = pd.to_numeric(selected[ADV20_VALUE_COLUMN], errors="coerce")
    capped_weights = adv20_value * max_position_adv_fraction / portfolio_notional
    capped_weights = capped_weights.clip(lower=0.0)
    if capped_weights.notna().sum() == 0:
        return equal_weights

    result = pd.Series(0.0, index=selected.index, dtype=np.float64)
    remaining = list(selected.index)
    remaining_budget = gross_target

    while remaining and remaining_budget > 1e-12:
        proposed = remaining_budget / len(remaining)
        newly_capped: list[int] = []
        for idx in list(remaining):
            cap = capped_weights.loc[idx]
            if pd.notna(cap) and cap < proposed:
                result.loc[idx] = cap
                remaining_budget -= cap
                remaining.remove(idx)
                newly_capped.append(idx)
        if not newly_capped:
            for idx in remaining:
                cap = capped_weights.loc[idx]
                alloc = proposed if pd.isna(cap) else min(proposed, cap)
                result.loc[idx] = alloc
            remaining_budget = 0.0
            break

    return result


def build_long_short_holdings(
    prediction_frame: pd.DataFrame,
    top_k: int,
    bottom_k: int | None = None,
    gross_normalize: bool = True,
    allow_short: bool = True,
    min_daily_value_traded: float | None = None,
    min_adv20_value_traded: float | None = None,
    max_position_adv_fraction: float | None = None,
    portfolio_notional: float | None = None,
    block_limit_up_entry: bool = False,
    exclude_hard_issues: bool = False,
) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    bottom_k = top_k if bottom_k is None else bottom_k
    if allow_short and bottom_k <= 0:
        raise ValueError("bottom_k must be positive.")

    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    long_notional = 0.5 if (gross_normalize and allow_short) else 1.0
    long_weight = long_notional / top_k
    short_weight = (-(0.5 if gross_normalize else 1.0) / bottom_k) if allow_short else 0.0

    for realized_date, group in prediction_frame.groupby("realized_date", sort=True):
        clean = group.dropna(subset=["prob_class_1", "next_return"])
        if min_daily_value_traded is not None and VALUE_TRADED_COLUMN in clean.columns:
            clean = clean[pd.to_numeric(clean[VALUE_TRADED_COLUMN], errors="coerce") >= min_daily_value_traded]
        if min_adv20_value_traded is not None and ADV20_VALUE_COLUMN in clean.columns:
            clean = clean[pd.to_numeric(clean[ADV20_VALUE_COLUMN], errors="coerce") >= min_adv20_value_traded]
        if block_limit_up_entry and LIMIT_UP_LIKE_COLUMN in clean.columns:
            clean = clean[~clean[LIMIT_UP_LIKE_COLUMN].fillna(False)]
        if exclude_hard_issues and HARD_ISSUE_COLUMN in clean.columns:
            clean = clean[~clean[HARD_ISSUE_COLUMN].fillna(False)]
        required_count = top_k + bottom_k if allow_short else top_k
        if len(clean) < required_count:
            continue

        ranked = clean.sort_values("prob_class_1", ascending=False)
        long_leg = ranked.head(top_k)
        short_leg = ranked.tail(bottom_k) if allow_short else ranked.iloc[0:0]
        long_weights = _apply_position_caps(
            selected=long_leg,
            gross_target=long_notional,
            max_position_adv_fraction=max_position_adv_fraction,
            portfolio_notional=portfolio_notional,
        )

        for _, row in long_leg.iterrows():
            weight = float(long_weights.loc[row.name]) if row.name in long_weights.index else long_weight
            rows.append(
                {
                    "realized_date": realized_date,
                    "anchor_date": row["anchor_date"],
                    "code": row["code"],
                    "side": "long",
                    "weight": weight,
                    "prob_class_1": float(row["prob_class_1"]),
                    "target_class": int(row["target_class"]),
                    "next_return": float(row["next_return"]),
                    "return_contribution": weight * float(row["next_return"]),
                }
            )
        for _, row in short_leg.iterrows():
            rows.append(
                {
                    "realized_date": realized_date,
                    "anchor_date": row["anchor_date"],
                    "code": row["code"],
                    "side": "short",
                    "weight": short_weight,
                    "prob_class_1": float(row["prob_class_1"]),
                    "target_class": int(row["target_class"]),
                    "next_return": float(row["next_return"]),
                    "return_contribution": short_weight * float(row["next_return"]),
                }
            )

    holdings = pd.DataFrame(rows)
    if holdings.empty:
        return holdings
    return holdings.sort_values(["realized_date", "side", "prob_class_1"], ascending=[True, True, False]).reset_index(drop=True)


def compute_long_short_returns_from_holdings(
    holdings: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    buy_cost_bps: float | None = None,
    sell_cost_bps: float | None = None,
    sell_tax_bps: float = 0.0,
) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()

    default_cost_rate = transaction_cost_bps / 10_000.0
    buy_cost_rate = default_cost_rate if buy_cost_bps is None else buy_cost_bps / 10_000.0
    sell_cost_rate = default_cost_rate if sell_cost_bps is None else sell_cost_bps / 10_000.0
    sell_tax_rate = sell_tax_bps / 10_000.0
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    previous_weights: dict[str, float] = {}

    for realized_date, group in holdings.groupby("realized_date", sort=True):
        weights = dict(zip(group["code"], group["weight"], strict=False))
        turnover = 0.0
        buy_turnover = 0.0
        sell_turnover = 0.0
        for code in set(previous_weights) | set(weights):
            delta = weights.get(code, 0.0) - previous_weights.get(code, 0.0)
            abs_delta = abs(delta)
            turnover += abs_delta
            if delta > 0:
                buy_turnover += delta
            elif delta < 0:
                sell_turnover += -delta

        long_leg = group[group["side"] == "long"]
        short_leg = group[group["side"] == "short"]
        gross_return = float(group["return_contribution"].sum())
        long_return = float(long_leg["next_return"].mean()) if not long_leg.empty else float("nan")
        short_return = float(short_leg["next_return"].mean()) if not short_leg.empty else float("nan")
        spread = long_return - short_return if pd.notna(long_return) and pd.notna(short_return) else float("nan")
        buy_cost = buy_turnover * buy_cost_rate
        sell_cost = sell_turnover * sell_cost_rate
        sell_tax = sell_turnover * sell_tax_rate
        transaction_cost = buy_cost + sell_cost + sell_tax
        net_return = gross_return - transaction_cost

        rows.append(
            {
                "realized_date": realized_date,
                "long_count": int(len(long_leg)),
                "short_count": int(len(short_leg)),
                "long_return": long_return,
                "short_return": short_return,
                "spread_return": spread,
                "strategy_return_gross": gross_return,
                "turnover": turnover,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "buy_cost": buy_cost,
                "sell_cost": sell_cost,
                "sell_tax": sell_tax,
                "transaction_cost": transaction_cost,
                "strategy_return": net_return,
            }
        )
        previous_weights = weights

    result = pd.DataFrame(rows)
    result["equity_curve_gross"] = (1.0 + result["strategy_return_gross"]).cumprod()
    result["equity_curve"] = (1.0 + result["strategy_return"]).cumprod()
    return result


def compute_long_short_returns(
    prediction_frame: pd.DataFrame,
    top_k: int,
    bottom_k: int | None = None,
    gross_normalize: bool = True,
    transaction_cost_bps: float = 0.0,
    allow_short: bool = True,
    buy_cost_bps: float | None = None,
    sell_cost_bps: float | None = None,
    sell_tax_bps: float = 0.0,
) -> pd.DataFrame:
    holdings = build_long_short_holdings(
        prediction_frame=prediction_frame,
        top_k=top_k,
        bottom_k=bottom_k,
        gross_normalize=gross_normalize,
        allow_short=allow_short,
    )
    return compute_long_short_returns_from_holdings(
        holdings=holdings,
        transaction_cost_bps=transaction_cost_bps,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        sell_tax_bps=sell_tax_bps,
    )
