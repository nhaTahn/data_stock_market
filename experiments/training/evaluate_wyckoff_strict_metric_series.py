"""Evaluate rel_score and abs(E) metric series for wyckoff_strict candidates.

Uses cached rolling predictions from `hetero_long_finetune_batch_20260522`.
No retraining and no holdout/test access.

Outputs:
- daily_metric_series.csv: per seed/fold/policy/date selected-sample metrics
- fold_metric_summary.csv: per seed/fold/policy aggregate metrics
- policy_metric_summary.csv: cross-fold summary by policy
- summary.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.run_hetero_long_finetune_batch import (  # noqa: E402
    DEFAULT_DATA,
    TRAIN_DAYS,
    TEST_DAYS,
    STEP_DAYS,
    VAL_START,
    VAL_END,
    build_sequence_inputs,
    candidate_signals,
    make_fold_dates,
    select_by_dates,
)

DEFAULT_PRED_DIR = (
    ROOT
    / "data/processed/assets/data_info_vn/history/training_runs/reports"
    / "hetero_long_finetune_batch_20260522/predictions"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/processed/assets/data_info_vn/history/training_runs/reports"
    / "hetero_wyckoff_metric_series_20260524"
)
DEFAULT_GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/hetero_wyckoff_metric_series_20260524"

POLICIES = {
    "conf_ratio_q70_wyckoff_strict_r20_k20_m5": {
        "rule": "conf_ratio_q70",
        "top_k": 20,
        "min_positions": 5,
    },
    "conf_ratio_q70_wyckoff_strict_r20_k20_m6": {
        "rule": "conf_ratio_q70",
        "top_k": 20,
        "min_positions": 6,
    },
    "daily_bot_sig_wyckoff_strict_r20_k20_m5": {
        "rule": "daily_bot_sig_50pct",
        "top_k": 20,
        "min_positions": 5,
    },
    "abs_mu_q60_wyckoff_strict_r20_k20_m6": {
        "rule": "abs_mu_q60",
        "top_k": 20,
        "min_positions": 6,
    },
}


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.50) + 0.5 * np.quantile(np.abs(clean), 0.90))


def rel_score(actual: np.ndarray, pred: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float(1.0 - robust_loss(actual - pred) / base)


def metric_row(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = actual - pred
    abs_error = np.abs(error)
    return {
        "n": int(len(actual)),
        "rel_score": rel_score(actual, pred),
        "absE_median": float(np.quantile(abs_error, 0.50)) if len(abs_error) else float("nan"),
        "absE_q90": float(np.quantile(abs_error, 0.90)) if len(abs_error) else float("nan"),
        "absE_robust": robust_loss(error),
        "base_robust": robust_loss(actual),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))) if len(actual) else float("nan"),
        "pred_actual_q90_ratio": (
            float(np.quantile(np.abs(pred), 0.90) / max(np.quantile(np.abs(actual), 0.90), 1e-8))
            if len(actual)
            else float("nan")
        ),
    }


def wyckoff_strict_gate(frame: pd.DataFrame) -> pd.Series:
    daily = (
        frame[["Date", "buying_pressure", "selling_pressure", "wyckoff_phase_60d"]]
        .dropna()
        .groupby("Date")
        .mean()
        .sort_index()
    )
    pressure_delta_20 = (daily["buying_pressure"] - daily["selling_pressure"]).rolling(20, min_periods=5).mean()
    gate = (pressure_delta_20 >= 0) & (daily["wyckoff_phase_60d"] >= 0.35)
    return gate.astype(bool)


def select_policy_rows(signal: np.ndarray, meta: pd.DataFrame, *, top_k: int, min_positions: int) -> np.ndarray:
    selected = np.zeros(len(meta), dtype=bool)
    tmp = meta[["Date", "code"]].copy().reset_index(drop=True)
    tmp["signal"] = signal
    for _, day in tmp.groupby("Date", sort=False):
        eligible = day.loc[day["signal"].astype(float) > 0.0]
        if len(eligible) < min_positions:
            continue
        top = eligible.nlargest(min(top_k, len(eligible)), "signal")
        selected[top.index.to_numpy(dtype=int)] = True
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seeds", default="43,52,62")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)

    # Lightweight argparse-compatible object for build_sequence_inputs.
    class BuildArgs:
        data = args.data
        target_normalizer = "volatility_20"
        target_column = "target_next_return"
        window_size = 15

    frame, x_all, y_all, meta_all, _ = build_sequence_inputs(BuildArgs())
    strict_gate = wyckoff_strict_gate(frame)
    folds = make_fold_dates(meta_all)
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]

    daily_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for seed in seeds:
        for fold_idx, (_, test_dates) in enumerate(folds):
            pred_path = args.pred_dir / f"seed{seed}_fold{fold_idx:03d}.npz"
            if not pred_path.exists():
                print(f"skip missing {pred_path}")
                continue
            test_mask = select_by_dates(meta_all, test_dates)
            meta_test = meta_all.loc[test_mask].reset_index(drop=True)
            y_test = y_all[test_mask]
            pred = np.load(pred_path)
            mu_train = pred["mu_train"]
            sigma_train = pred["sigma_train"]
            mu_test = pred["mu_test"]
            sigma_test = pred["sigma_test"]
            signals = candidate_signals(mu_train, sigma_train, mu_test, sigma_test, meta_test)
            date_gate = pd.Series(pd.to_datetime(meta_test["Date"])).map(strict_gate).fillna(False).to_numpy(dtype=bool)

            for policy_name, cfg in POLICIES.items():
                raw_signal = signals[cfg["rule"]]
                gated_signal = np.where(date_gate, raw_signal, 0.0).astype(np.float32)
                selected = select_policy_rows(
                    gated_signal,
                    meta_test,
                    top_k=int(cfg["top_k"]),
                    min_positions=int(cfg["min_positions"]),
                )
                fold_metric = metric_row(y_test[selected], mu_test[selected]) if selected.any() else metric_row(np.array([]), np.array([]))
                fold_metric.update(
                    {
                        "seed": seed,
                        "fold_id": fold_idx,
                        "test_start": str(test_dates[0].date()),
                        "test_end": str(test_dates[-1].date()),
                        "policy": policy_name,
                        "coverage": float(selected.mean()),
                        "active_days": float(pd.Series(selected).groupby(meta_test["Date"].values).any().mean()),
                    }
                )
                fold_rows.append(fold_metric)

                tmp = pd.DataFrame(
                    {
                        "Date": pd.to_datetime(meta_test["Date"]),
                        "actual": y_test,
                        "pred": mu_test,
                        "selected": selected,
                    }
                )
                for date, day in tmp.groupby("Date", sort=True):
                    active = day.loc[day["selected"]]
                    row = {
                        "seed": seed,
                        "fold_id": fold_idx,
                        "Date": date,
                        "policy": policy_name,
                        "n_selected": int(len(active)),
                    }
                    if len(active) > 0:
                        row.update(metric_row(active["actual"].to_numpy(), active["pred"].to_numpy()))
                    else:
                        row.update(metric_row(np.array([]), np.array([])))
                    daily_rows.append(row)

    daily_df = pd.DataFrame(daily_rows)
    fold_df = pd.DataFrame(fold_rows)
    daily_df.to_csv(args.output_dir / "daily_metric_series.csv", index=False)
    fold_df.to_csv(args.output_dir / "fold_metric_summary.csv", index=False)
    daily_df.to_csv(args.gold_dir / "daily_metric_series.csv", index=False)
    fold_df.to_csv(args.gold_dir / "fold_metric_summary.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for policy, group in fold_df.groupby("policy"):
        summary_rows.append(
            {
                "policy": policy,
                "n_folds": int(len(group)),
                "mean_rel_score": float(group["rel_score"].mean()),
                "median_rel_score": float(group["rel_score"].median()),
                "positive_rel_folds": int((group["rel_score"] > 0).sum()),
                "mean_absE_robust": float(group["absE_robust"].mean()),
                "median_absE_robust": float(group["absE_robust"].median()),
                "p90_absE_robust": float(group["absE_robust"].quantile(0.90)),
                "mean_absE_q90": float(group["absE_q90"].mean()),
                "mean_directional_accuracy": float(group["directional_accuracy"].mean()),
                "mean_pred_actual_q90_ratio": float(group["pred_actual_q90_ratio"].mean()),
                "mean_coverage": float(group["coverage"].mean()),
                "mean_active_days": float(group["active_days"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_rel_score", ascending=False)
    summary_df.to_csv(args.output_dir / "policy_metric_summary.csv", index=False)
    summary_df.to_csv(args.gold_dir / "policy_metric_summary.csv", index=False)

    cols = [
        "policy",
        "mean_rel_score",
        "median_rel_score",
        "positive_rel_folds",
        "mean_absE_robust",
        "p90_absE_robust",
        "mean_absE_q90",
        "mean_directional_accuracy",
        "mean_pred_actual_q90_ratio",
        "mean_coverage",
        "mean_active_days",
    ]
    text = "\n".join(
        [
            "# Wyckoff Strict Metric Series Summary",
            "",
            "Scope: VN validation rolling folds only. Holdout/test not used.",
            "",
            summary_df[cols].round(5).to_markdown(index=False),
            "",
            "Outputs:",
            "- daily_metric_series.csv",
            "- fold_metric_summary.csv",
            "- policy_metric_summary.csv",
        ]
    )
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
