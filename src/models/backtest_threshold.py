from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.report_layout import report_backtest_path, resolve_run_artifact


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple threshold backtest from predictions.csv.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--thresholds", default="0,0.0025,0.005,0.01,0.015,0.02")
    parser.add_argument("--non-overlap", action="store_true")
    parser.add_argument("--holding-period", type=int, default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--uncertainty-model", default=None)
    parser.add_argument("--uncertainty-column", default="prediction_uncertainty")
    parser.add_argument("--uncertainty-side", choices=["low", "high"], default="low")
    parser.add_argument("--uncertainty-quantiles", default=None)
    parser.add_argument("--max-uncertainty", type=float, default=None)
    parser.add_argument("--output-suffix", default=None)
    return parser.parse_args()


def infer_holding_period(run_dir: Path) -> int:
    config_path = resolve_run_artifact(run_dir, "config.json", bucket="core")
    if not config_path.exists():
        return 1
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    target_mode = config.get("target_mode", "return")
    if target_mode == "return_3d":
        return 3
    if target_mode == "return_5d":
        return 5
    return 1


def select_non_overlap_trades(
    model_df: pd.DataFrame,
    threshold: float,
    holding_period: int,
    uncertainty_column: str | None = None,
    max_uncertainty: float | None = None,
    uncertainty_side: str = "low",
) -> pd.DataFrame:
    selected_indices: list[int] = []
    i = 0
    model_df = model_df.reset_index(drop=True)
    while i < len(model_df):
        row = model_df.loc[i]
        eligible = float(row["prediction"]) >= threshold
        if eligible and uncertainty_column and max_uncertainty is not None:
            uncertainty_value = row.get(uncertainty_column)
            if pd.isna(uncertainty_value):
                eligible = False
            elif uncertainty_side == "high":
                eligible = float(uncertainty_value) >= max_uncertainty
            else:
                eligible = float(uncertainty_value) <= max_uncertainty
        if eligible:
            selected_indices.append(i)
            i += holding_period
            continue
        i += 1
    if not selected_indices:
        return model_df.iloc[0:0].copy()
    return model_df.iloc[selected_indices].copy()


def summarize_active_rows(
    active: pd.DataFrame,
    total_rows: int,
    model_name: str,
    threshold: float,
    holding_period: int,
    uncertainty_model: str | None = None,
    uncertainty_quantile: float | None = None,
    max_uncertainty: float | None = None,
    uncertainty_side: str | None = None,
) -> dict[str, float | str | int]:
    if active.empty:
        return {
            "model": model_name,
            "threshold": threshold,
            "holding_period": holding_period,
            "uncertainty_model": uncertainty_model,
            "uncertainty_quantile": uncertainty_quantile,
            "max_uncertainty": max_uncertainty,
            "uncertainty_side": uncertainty_side,
            "trade_count": 0,
            "coverage": 0.0,
            "directional_accuracy": np.nan,
            "avg_actual_return": np.nan,
            "avg_strategy_return": np.nan,
            "cumulative_strategy_return": 0.0,
            "final_equity": 1.0,
            "sampled_buy_hold_equity": 1.0,
        }
    strategy_return = active["actual"]
    sampled_buy_hold_equity = float((1.0 + active["actual"]).cumprod().iloc[-1])
    final_equity = float((1.0 + strategy_return).cumprod().iloc[-1])
    return {
        "model": model_name,
        "threshold": threshold,
        "holding_period": holding_period,
        "uncertainty_model": uncertainty_model,
        "uncertainty_quantile": uncertainty_quantile,
        "max_uncertainty": max_uncertainty,
        "uncertainty_side": uncertainty_side,
        "trade_count": int(len(active)),
        "coverage": float(len(active) / total_rows),
        "directional_accuracy": float(np.mean(active["actual"] >= 0.0)),
        "avg_actual_return": float(active["actual"].mean()),
        "avg_strategy_return": float(strategy_return.mean()),
        "cumulative_strategy_return": float(strategy_return.sum()),
        "final_equity": final_equity,
        "sampled_buy_hold_equity": sampled_buy_hold_equity,
    }


def resolve_merge_keys(signal_df: pd.DataFrame, uncertainty_df: pd.DataFrame) -> list[str]:
    merge_keys = [column for column in ("code", "Date", "split") if column in signal_df.columns and column in uncertainty_df.columns]
    if not merge_keys and "Date" in signal_df.columns and "Date" in uncertainty_df.columns:
        merge_keys = ["Date"]
    if not merge_keys:
        raise ValueError("Could not resolve merge keys for uncertainty sidecar.")
    return merge_keys


def attach_uncertainty_column(
    df: pd.DataFrame,
    signal_model: str,
    uncertainty_model: str | None,
    uncertainty_column: str,
) -> pd.DataFrame:
    model_df = df[(df["model"] == signal_model) & (df["split"] == "test")].sort_values("Date").copy()
    if model_df.empty:
        return model_df

    if uncertainty_model is None:
        if uncertainty_column in model_df.columns:
            model_df["__uncertainty__"] = pd.to_numeric(model_df[uncertainty_column], errors="coerce")
        return model_df

    uncertainty_df = df[(df["model"] == uncertainty_model) & (df["split"] == "test")].sort_values("Date").copy()
    if uncertainty_df.empty:
        raise ValueError(f"Uncertainty model '{uncertainty_model}' has no test rows.")
    if uncertainty_column not in uncertainty_df.columns:
        raise ValueError(f"Column '{uncertainty_column}' not found for uncertainty model '{uncertainty_model}'.")

    merge_keys = resolve_merge_keys(model_df, uncertainty_df)
    merged = model_df.merge(
        uncertainty_df[merge_keys + [uncertainty_column]].rename(columns={uncertainty_column: "__uncertainty__"}),
        on=merge_keys,
        how="left",
    )
    return merged


def resolve_uncertainty_settings(
    model_df: pd.DataFrame,
    quantiles: list[float],
    explicit_max_uncertainty: float | None,
) -> list[tuple[float | None, float | None]]:
    if explicit_max_uncertainty is None and not quantiles:
        return [(None, None)]
    if "__uncertainty__" not in model_df.columns:
        return []
    values = model_df["__uncertainty__"].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return []

    settings: list[tuple[float | None, float | None]] = []
    if explicit_max_uncertainty is not None:
        settings.append((None, float(explicit_max_uncertainty)))
    for quantile in quantiles:
        settings.append((float(quantile), float(np.quantile(values, quantile))))
    deduped: list[tuple[float | None, float | None]] = []
    seen: set[tuple[float | None, float | None]] = set()
    for item in settings:
        key = (None if item[0] is None else round(item[0], 6), None if item[1] is None else round(item[1], 12))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped or [(None, None)]


def compute_backtest_rows(
    df: pd.DataFrame,
    thresholds: list[float],
    non_overlap: bool,
    holding_period: int,
    model_names: list[str] | None = None,
    uncertainty_model: str | None = None,
    uncertainty_column: str = "prediction_uncertainty",
    uncertainty_side: str = "low",
    uncertainty_quantiles: list[float] | None = None,
    max_uncertainty: float | None = None,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    selected_model_names = model_names or sorted(df["model"].unique())
    uncertainty_quantiles = uncertainty_quantiles or []
    for model_name in selected_model_names:
        model_df = attach_uncertainty_column(df, model_name, uncertainty_model, uncertainty_column)
        if model_df.empty:
            continue
        uncertainty_settings = resolve_uncertainty_settings(model_df, uncertainty_quantiles, max_uncertainty)
        if not uncertainty_settings:
            continue
        for threshold in thresholds:
            for uncertainty_quantile, uncertainty_cutoff in uncertainty_settings:
                if non_overlap and holding_period > 1:
                    active = select_non_overlap_trades(
                        model_df,
                        threshold,
                        holding_period,
                        uncertainty_column="__uncertainty__" if uncertainty_cutoff is not None else None,
                        max_uncertainty=uncertainty_cutoff,
                        uncertainty_side=uncertainty_side,
                    )
                else:
                    active = model_df[model_df["prediction"] >= threshold].copy()
                    if uncertainty_cutoff is not None:
                        if uncertainty_side == "high":
                            active = active[active["__uncertainty__"] >= uncertainty_cutoff].copy()
                        else:
                            active = active[active["__uncertainty__"] <= uncertainty_cutoff].copy()
                rows.append(
                    summarize_active_rows(
                        active,
                        len(model_df),
                        model_name,
                        threshold,
                        holding_period,
                        uncertainty_model=uncertainty_model,
                        uncertainty_quantile=uncertainty_quantile,
                        max_uncertainty=uncertainty_cutoff,
                        uncertainty_side=uncertainty_side if uncertainty_cutoff is not None else None,
                    )
                )
    return rows


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    model_names = parse_csv_list(args.models)
    uncertainty_quantiles = parse_float_list(args.uncertainty_quantiles)
    holding_period = args.holding_period or infer_holding_period(run_dir)
    predictions = pd.read_csv(resolve_run_artifact(run_dir, "predictions.csv", bucket="core"))
    predictions["Date"] = pd.to_datetime(predictions["Date"])

    result_df = pd.DataFrame(
        compute_backtest_rows(
            predictions,
            thresholds,
            args.non_overlap,
            holding_period,
            model_names=model_names or None,
            uncertainty_model=args.uncertainty_model,
            uncertainty_column=args.uncertainty_column,
            uncertainty_side=args.uncertainty_side,
            uncertainty_quantiles=uncertainty_quantiles,
            max_uncertainty=args.max_uncertainty,
        )
    )
    suffix = "_non_overlap" if args.non_overlap and holding_period > 1 else ""
    if args.uncertainty_model or args.max_uncertainty is not None or uncertainty_quantiles:
        suffix += "_uncertainty"
    if args.output_suffix:
        suffix += f"_{args.output_suffix.strip('_')}"
    output_csv = report_backtest_path(run_dir, f"threshold_backtest{suffix}.csv")
    result_df.to_csv(output_csv, index=False)

    best_rows = (
        result_df.dropna(subset=["avg_strategy_return"])
        .sort_values(["final_equity", "avg_strategy_return", "directional_accuracy", "coverage"], ascending=[False, False, False, False])
        .groupby("model", as_index=False)
        .head(1)
    )
    summary = {
        row["model"]: {
            "threshold": float(row["threshold"]),
            "holding_period": int(row["holding_period"]),
            "uncertainty_model": row.get("uncertainty_model"),
            "uncertainty_quantile": None if pd.isna(row.get("uncertainty_quantile")) else float(row["uncertainty_quantile"]),
            "max_uncertainty": None if pd.isna(row.get("max_uncertainty")) else float(row["max_uncertainty"]),
            "uncertainty_side": row.get("uncertainty_side"),
            "trade_count": int(row["trade_count"]),
            "coverage": float(row["coverage"]),
            "directional_accuracy": float(row["directional_accuracy"]),
            "avg_actual_return": float(row["avg_actual_return"]),
            "avg_strategy_return": float(row["avg_strategy_return"]),
            "cumulative_strategy_return": float(row["cumulative_strategy_return"]),
            "final_equity": float(row["final_equity"]),
            "sampled_buy_hold_equity": float(row["sampled_buy_hold_equity"]),
        }
        for _, row in best_rows.iterrows()
    }
    summary_path = report_backtest_path(run_dir, f"threshold_backtest_summary{suffix}.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", output_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
