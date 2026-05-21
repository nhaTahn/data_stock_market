from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "regime_analysis"


DEFAULT_RUNS = (
    "broad_signmag_prune_compact_core12_20260424_r01",
    "broad_signmag_prune_no_fast_overlap_20260424_r01",
    "broad_signmag_prune_general_sector_breadth_20260424_r04",
    "broad_signmag_prune_general_sector_full_20260424_r04",
)
DEFAULT_MODELS = ("lstm_signmag_best_by_val",)
DEFAULT_SPLITS = ("train", "val")


@dataclass(frozen=True)
class RegimeRuleConfig:
    trend_window_fast: int = 20
    trend_window_slow: int = 60
    breadth_up: float = 0.53
    breadth_down: float = 0.47


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing VN model predictions by point-in-time market regime."
    )
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="Comma-separated training run names.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model names in predictions.csv.")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated splits to evaluate.")
    parser.add_argument("--stamp", default="20260425_r01", help="Report stamp.")
    parser.add_argument("--output-name", default="current_best_regime", help="Report folder prefix.")
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def loss_fn(values: pd.Series | np.ndarray) -> float:
    array = np.abs(np.asarray(values, dtype=float))
    if len(array) == 0:
        return float("nan")
    return float(np.quantile(array, 0.5) + 0.5 * np.quantile(array, 0.9))


def rel_score(error: pd.Series, base: pd.Series) -> float:
    base_loss = loss_fn(base)
    abs_loss = loss_fn(error)
    if not np.isfinite(base_loss) or base_loss == 0:
        return float("nan")
    return float(1.0 - abs_loss / base_loss)


def local_rel_score_proxy(error: pd.Series, base: pd.Series) -> np.ndarray:
    base_abs = np.abs(base.to_numpy(dtype=float))
    error_abs = np.abs(error.to_numpy(dtype=float))
    proxy_floor = max(loss_fn(base), 1e-4)
    denominator = np.maximum(base_abs, proxy_floor)
    return np.clip(1.0 - error_abs / denominator, -1.5, 1.0)


def load_predictions(run_name: str) -> pd.DataFrame:
    path = RUN_ROOT / run_name / "reports" / "core" / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions.csv for run: {run_name}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_daily_regimes(prediction_df: pd.DataFrame, config: RegimeRuleConfig) -> pd.DataFrame:
    base_df = (
        prediction_df[prediction_df["model"] == prediction_df["model"].iloc[0]]
        .drop_duplicates(["code", "Date"])
        .copy()
    )
    daily = (
        base_df.groupby("Date", as_index=False)
        .agg(
            market_return=("actual", "mean"),
            breadth=("actual", lambda values: float(np.mean(np.asarray(values, dtype=float) > 0.0))),
            stock_count=("code", "nunique"),
        )
        .sort_values("Date", kind="stable")
        .reset_index(drop=True)
    )
    daily["market_return_20"] = daily["market_return"].rolling(config.trend_window_fast, min_periods=10).mean()
    daily["market_return_60"] = daily["market_return"].rolling(config.trend_window_slow, min_periods=30).mean()
    daily["breadth_20"] = daily["breadth"].rolling(config.trend_window_fast, min_periods=10).mean()
    daily["breadth_60"] = daily["breadth"].rolling(config.trend_window_slow, min_periods=30).mean()
    daily["market_volatility_20"] = daily["market_return"].rolling(config.trend_window_fast, min_periods=10).std()
    daily["volatility_expanding_median"] = daily["market_volatility_20"].expanding(min_periods=60).median()

    regimes = np.full(len(daily), "sideways", dtype=object)
    fast_up = daily["market_return_20"] > 0.0
    slow_up = daily["market_return_60"] > 0.0
    fast_down = daily["market_return_20"] < 0.0
    slow_down = daily["market_return_60"] < 0.0
    breadth_strong = daily["breadth_20"] >= config.breadth_up
    breadth_weak = daily["breadth_20"] <= config.breadth_down
    high_vol = daily["market_volatility_20"] >= daily["volatility_expanding_median"]

    regimes[fast_up & slow_up & breadth_strong] = "uptrend"
    regimes[fast_down & slow_down & breadth_weak] = "downtrend"
    regimes[fast_up & (~slow_up) & breadth_strong] = "recovery"
    regimes[(slow_up | (daily["market_return_60"] >= -0.0005)) & breadth_weak & (high_vol | fast_down)] = "distribution"

    daily["regime"] = regimes
    return daily


def align_predictions(prediction_df: pd.DataFrame, run_name: str, model_name: str, splits: set[str]) -> pd.DataFrame:
    model_df = prediction_df[
        (prediction_df["model"] == model_name) & (prediction_df["split"].isin(splits))
    ].copy()
    if model_df.empty:
        return pd.DataFrame()

    aligned_parts: list[pd.DataFrame] = []
    group_columns = ["code", "split"]
    for (code, split), group in model_df.sort_values(["code", "split", "Date"], kind="stable").groupby(group_columns, sort=False):
        if len(group) < 3:
            continue
        signal_rows = group.iloc[1:-1].reset_index(drop=True)
        actual_rows = group.iloc[2:].reset_index(drop=True)
        part = pd.DataFrame(
            {
                "run_name": run_name,
                "model": model_name,
                "code": code,
                "split": split,
                "signal_date": signal_rows["Date"],
                "actual_date": actual_rows["Date"],
                "prediction": signal_rows["prediction"].to_numpy(dtype=float),
                "actual": actual_rows["actual"].to_numpy(dtype=float),
            }
        )
        aligned_parts.append(part)

    if not aligned_parts:
        return pd.DataFrame()

    aligned = pd.concat(aligned_parts, ignore_index=True)
    aligned["error"] = aligned["actual"] - aligned["prediction"]
    aligned["abs_error"] = aligned["error"].abs()
    aligned["abs_actual"] = aligned["actual"].abs()
    aligned["direction_ok"] = np.sign(aligned["prediction"]) == np.sign(aligned["actual"])
    return aligned


def summarize_group(group: pd.DataFrame) -> dict[str, object]:
    proxy = local_rel_score_proxy(group["error"], group["actual"])
    return {
        "n_obs": int(len(group)),
        "n_days": int(group["actual_date"].nunique()),
        "n_stocks": int(group["code"].nunique()),
        "rel_score": rel_score(group["error"], group["actual"]),
        "directional_accuracy": float(group["direction_ok"].mean()),
        "mean_error": float(group["error"].mean()),
        "median_abs_error": float(group["abs_error"].median()),
        "error_q2": float(group["error"].quantile(0.2)),
        "error_q8": float(group["error"].quantile(0.8)),
        "proxy_mean": float(np.mean(proxy)),
        "proxy_median": float(np.median(proxy)),
        "proxy_positive_share": float(np.mean(proxy > 0.0)),
    }


def build_summary(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["run_name", "model", "split", "regime"]
    for keys, group in aligned.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        row.update(summarize_group(group))
        rows.append(row)

    for keys, group in aligned.groupby(["run_name", "model", "split"], sort=True):
        row = dict(zip(["run_name", "model", "split"], keys, strict=True))
        row["regime"] = "ALL"
        row.update(summarize_group(group))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["split", "run_name", "model", "regime"], kind="stable")


def build_quartile_summary(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["run_name", "model", "split", "regime"]
    for keys, group in aligned.groupby(group_cols, sort=True):
        day_returns: list[float] = []
        trade_counts: list[int] = []
        for _, day_df in group.groupby("actual_date", sort=True):
            if len(day_df) < 8:
                continue
            lower = day_df["prediction"].quantile(0.25)
            upper = day_df["prediction"].quantile(0.75)
            long_df = day_df[day_df["prediction"] >= upper]
            short_df = day_df[day_df["prediction"] <= lower]
            if long_df.empty or short_df.empty:
                continue
            day_returns.append(float(long_df["actual"].mean() - short_df["actual"].mean()))
            trade_counts.append(int(len(long_df) + len(short_df)))
        equity = float(np.prod(1.0 + np.asarray(day_returns, dtype=float))) if day_returns else float("nan")
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "quartile_days": int(len(day_returns)),
                "quartile_mean_return": float(np.mean(day_returns)) if day_returns else float("nan"),
                "quartile_median_return": float(np.median(day_returns)) if day_returns else float("nan"),
                "quartile_hit_rate": float(np.mean(np.asarray(day_returns) > 0.0)) if day_returns else float("nan"),
                "quartile_final_equity": equity,
                "avg_trade_count": float(np.mean(trade_counts)) if trade_counts else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols, kind="stable")


def build_daily_quartile_returns(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["run_name", "model", "split", "actual_date", "regime"]
    for keys, group in aligned.groupby(group_cols, sort=True):
        if len(group) < 8:
            continue
        lower = group["prediction"].quantile(0.25)
        upper = group["prediction"].quantile(0.75)
        long_df = group[group["prediction"] >= upper]
        short_df = group[group["prediction"] <= lower]
        if long_df.empty or short_df.empty:
            continue
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "long_count": int(len(long_df)),
                "short_count": int(len(short_df)),
                "long_return": float(long_df["actual"].mean()),
                "short_return": float(short_df["actual"].mean()),
                "long_short_return": float(long_df["actual"].mean() - short_df["actual"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "run_name", "model", "actual_date"], kind="stable")


def build_regime_filter_summary(daily_quartile: pd.DataFrame) -> pd.DataFrame:
    filters: dict[str, set[str] | None] = {
        "all_regimes": None,
        "skip_downtrend": {"distribution", "recovery", "sideways", "uptrend"},
        "trade_distribution_sideways": {"distribution", "sideways"},
        "trade_distribution_sideways_recovery": {"distribution", "recovery", "sideways"},
        "trade_distribution_only": {"distribution"},
        "trade_sideways_only": {"sideways"},
    }
    rows: list[dict[str, object]] = []
    for keys, group in daily_quartile.groupby(["run_name", "model", "split"], sort=True):
        for filter_name, regimes in filters.items():
            selected = group if regimes is None else group[group["regime"].isin(regimes)]
            returns = selected["long_short_return"].to_numpy(dtype=float)
            row = dict(zip(["run_name", "model", "split"], keys, strict=True))
            row.update(
                {
                    "filter_name": filter_name,
                    "regimes": "ALL" if regimes is None else ",".join(sorted(regimes)),
                    "trade_days": int(len(selected)),
                    "final_equity": float(np.prod(1.0 + returns)) if len(returns) else float("nan"),
                    "mean_return": float(np.mean(returns)) if len(returns) else float("nan"),
                    "hit_rate": float(np.mean(returns > 0.0)) if len(returns) else float("nan"),
                    "avg_trade_count": float((selected["long_count"] + selected["short_count"]).mean()) if len(selected) else float("nan"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "run_name", "final_equity"], ascending=[True, True, False], kind="stable")


def save_histograms(aligned: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (run_name, model, split), group in aligned.groupby(["run_name", "model", "split"], sort=True):
        regimes = ["ALL"] + sorted([regime for regime in group["regime"].dropna().unique() if regime != "unknown"])
        fig, axes = plt.subplots(len(regimes), 2, figsize=(15, 3.3 * len(regimes)), squeeze=False)
        for row_idx, regime in enumerate(regimes):
            regime_df = group if regime == "ALL" else group[group["regime"] == regime]
            if regime_df.empty:
                axes[row_idx, 0].set_visible(False)
                axes[row_idx, 1].set_visible(False)
                continue
            proxy = local_rel_score_proxy(regime_df["error"], regime_df["actual"])
            axes[row_idx, 0].hist(regime_df["error"], bins=50, color="#1f77b4", alpha=0.78)
            axes[row_idx, 0].axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
            axes[row_idx, 0].set_title(f"{regime} | error E")
            axes[row_idx, 0].grid(True, alpha=0.2)
            axes[row_idx, 1].hist(proxy, bins=np.linspace(-1.5, 1.0, 41), color="#2ca02c", alpha=0.78)
            axes[row_idx, 1].axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
            axes[row_idx, 1].set_title(f"{regime} | stabilized relative_score proxy")
            axes[row_idx, 1].grid(True, alpha=0.2)
        fig.suptitle(f"{run_name} | {model} | {split} regime histograms", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        safe_name = f"{run_name}__{model}__{split}__regime_hist.png"
        fig.savefig(plot_dir / safe_name, dpi=180)
        plt.close(fig)


def write_markdown(
    summary: pd.DataFrame,
    quartile: pd.DataFrame,
    filter_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    val_summary = summary[(summary["split"] == "val") & (summary["regime"] != "ALL")].copy()
    val_quartile = quartile[quartile["split"] == "val"].copy()
    merged = val_summary.merge(
        val_quartile,
        on=["run_name", "model", "split", "regime"],
        how="left",
    )
    merged = merged.sort_values(["run_name", "rel_score"], ascending=[True, False], kind="stable")

    lines = [
        "# Regime Performance Smoke Test",
        "",
        "Scope: existing train/validation predictions only. No test/out-sample data is used.",
        "",
        "Regime is assigned at the signal date from broad market return, breadth, and rolling volatility computed only up to that date.",
        "",
        "Quartile filter equity is recomputed from aligned signal-date prediction to next actual-date return, so use it for within-report comparison rather than as a replacement for the run's official backtest artifact.",
        "",
        "## Validation Summary By Regime",
        "",
        "| Run | Regime | Obs | Days | rel_score | Direction | Error q2/q8 | Proxy > 0 | Quartile equity | Hit rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for _, row in merged.iterrows():
        lines.append(
            "| "
            f"`{row['run_name']}` | `{row['regime']}` | {int(row['n_obs'])} | {int(row['n_days'])} | "
            f"{float(row['rel_score']):+.4f} | {float(row['directional_accuracy']):.1%} | "
            f"{float(row['error_q2']):+.4f} / {float(row['error_q8']):+.4f} | "
            f"{float(row['proxy_positive_share']):.1%} | "
            f"{float(row['quartile_final_equity']):.3f} | {float(row['quartile_hit_rate']):.1%} |"
        )

    all_val = summary[(summary["split"] == "val") & (summary["regime"] == "ALL")].copy()
    lines.extend(
        [
            "",
            "## Overall Validation",
            "",
            "| Run | rel_score | Direction | Error q2/q8 | Proxy > 0 | Obs |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for _, row in all_val.sort_values("rel_score", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['run_name']}` | {float(row['rel_score']):+.4f} | "
            f"{float(row['directional_accuracy']):.1%} | "
            f"{float(row['error_q2']):+.4f} / {float(row['error_q8']):+.4f} | "
            f"{float(row['proxy_positive_share']):.1%} | {int(row['n_obs'])} |"
        )

    val_filters = filter_summary[filter_summary["split"] == "val"].copy()
    lines.extend(
        [
            "",
            "## Validation Regime Filters",
            "",
            "| Run | Filter | Trade days | Equity | Hit rate | Mean return |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in val_filters.sort_values(["run_name", "final_equity"], ascending=[True, False], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['run_name']}` | `{row['filter_name']}` | {int(row['trade_days'])} | "
            f"{float(row['final_equity']):.3f} | {float(row['hit_rate']):.1%} | {float(row['mean_return']):+.4f} |"
        )

    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_names = split_csv(args.runs)
    model_names = split_csv(args.models)
    splits = set(split_csv(args.splits))
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_aligned: list[pd.DataFrame] = []
    all_regimes: list[pd.DataFrame] = []
    for run_name in run_names:
        prediction_df = load_predictions(run_name)
        regimes = build_daily_regimes(prediction_df, RegimeRuleConfig())
        regimes = regimes.assign(run_name=run_name)
        all_regimes.append(regimes)
        regime_lookup = regimes[["Date", "regime"]].rename(columns={"Date": "signal_date"})
        for model_name in model_names:
            aligned = align_predictions(prediction_df, run_name, model_name, splits)
            if aligned.empty:
                continue
            aligned = aligned.merge(regime_lookup, on="signal_date", how="left")
            aligned["regime"] = aligned["regime"].fillna("unknown")
            all_aligned.append(aligned)

    if not all_aligned:
        raise RuntimeError("No aligned predictions were available for the requested runs/models/splits.")

    aligned_df = pd.concat(all_aligned, ignore_index=True)
    regimes_df = pd.concat(all_regimes, ignore_index=True)
    summary_df = build_summary(aligned_df)
    quartile_df = build_quartile_summary(aligned_df)
    daily_quartile_df = build_daily_quartile_returns(aligned_df)
    filter_summary_df = build_regime_filter_summary(daily_quartile_df)

    aligned_df.to_csv(output_dir / "aligned_predictions_with_regime.csv", index=False)
    regimes_df.to_csv(output_dir / "daily_regimes.csv", index=False)
    daily_quartile_df.to_csv(output_dir / "daily_quartile_returns.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    quartile_df.to_csv(output_dir / "quartile_by_regime.csv", index=False)
    filter_summary_df.to_csv(output_dir / "regime_filter_summary.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "runs": run_names,
                "models": model_names,
                "splits": sorted(splits),
                "summary": summary_df.to_dict(orient="records"),
                "quartile_by_regime": quartile_df.to_dict(orient="records"),
                "regime_filter_summary": filter_summary_df.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    save_histograms(aligned_df, output_dir)
    write_markdown(summary_df, quartile_df, filter_summary_df, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(summary_df)}, indent=2))


if __name__ == "__main__":
    main()
