from __future__ import annotations

import numpy as np
import pandas as pd

from fk_lstm_classifier.data import SequenceDataset


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
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anchor_date": pd.to_datetime(dataset.anchor_dates),
            "realized_date": pd.to_datetime(dataset.realized_dates),
            "code": dataset.codes,
            "target_class": dataset.labels,
            "next_return": dataset.next_returns,
            "prob_class_0": predicted_probabilities[:, 0],
            "prob_class_1": predicted_probabilities[:, 1],
        }
    ).sort_values(["realized_date", "prob_class_1"], ascending=[True, False]).reset_index(drop=True)


def compute_long_short_returns(
    prediction_frame: pd.DataFrame,
    top_k: int,
    bottom_k: int | None = None,
    gross_normalize: bool = True,
) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    bottom_k = top_k if bottom_k is None else bottom_k
    if bottom_k <= 0:
        raise ValueError("bottom_k must be positive.")

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for realized_date, group in prediction_frame.groupby("realized_date", sort=True):
        clean = group.dropna(subset=["prob_class_1", "next_return"])
        if len(clean) < top_k + bottom_k:
            continue

        ranked = clean.sort_values("prob_class_1", ascending=False)
        long_leg = ranked.head(top_k)
        short_leg = ranked.tail(bottom_k)

        long_return = float(long_leg["next_return"].mean())
        short_return = float(short_leg["next_return"].mean())
        spread = long_return - short_return
        strategy_return = 0.5 * spread if gross_normalize else spread

        rows.append(
            {
                "realized_date": realized_date,
                "long_count": int(len(long_leg)),
                "short_count": int(len(short_leg)),
                "long_return": long_return,
                "short_return": short_return,
                "spread_return": spread,
                "strategy_return": strategy_return,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["equity_curve"] = (1.0 + result["strategy_return"]).cumprod()
    return result
