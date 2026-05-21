from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metric import loss_fn  # noqa: E402
from src.reporting import report_benchmark_path, resolve_run_artifact  # noqa: E402


@dataclass(frozen=True)
class RegimeRuleConfig:
    trend_window_fast: int = 20
    trend_window_slow: int = 60
    breadth_up: float = 0.53
    breadth_down: float = 0.47


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    higher_is_better: bool = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package regime-aware and selective-performance diagnostics for an existing training run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", default=None, help="Comma-separated model names. Default: all models in predictions.csv.")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to include.")
    parser.add_argument("--output-name", default="regime_risk_report")
    parser.add_argument("--coverage-grid", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--min-names-per-panel", type=int, default=5)
    parser.add_argument(
        "--score-column",
        default=None,
        help="Optional column used for selective ranking. Higher is treated as better unless --score-lower-is-better is set.",
    )
    parser.add_argument("--score-lower-is-better", action="store_true")
    return parser.parse_args(argv)


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_run_config(run_dir: Path) -> dict[str, object]:
    path = resolve_run_artifact(run_dir, "config.json", bucket="core")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = resolve_run_artifact(run_dir, "predictions.csv", bucket="core")
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions.csv: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def infer_market_from_code(code: str, default_market: str) -> str:
    text = str(code).strip()
    if ":" in text:
        prefix = text.split(":", 1)[0].upper()
        if prefix.isalpha() and 1 <= len(prefix) <= 5:
            return prefix
    return default_market


def align_predictions(pred_df: pd.DataFrame, selected_models: set[str], selected_splits: set[str]) -> pd.DataFrame:
    work = pred_df[pred_df["model"].isin(selected_models) & pred_df["split"].isin(selected_splits)].copy()
    if work.empty:
        return pd.DataFrame()

    aligned_parts: list[pd.DataFrame] = []
    group_cols = ["model", "split", "code"]
    for keys, group in work.sort_values(["model", "split", "code", "Date"], kind="stable").groupby(group_cols, sort=False):
        if len(group) < 3:
            continue
        signal_rows = group.iloc[1:-1].reset_index(drop=True).copy()
        actual_rows = group.iloc[2:].reset_index(drop=True)

        signal_rows = signal_rows.rename(columns={"Date": "signal_date", "target": "signal_target", "actual": "signal_actual"})
        signal_rows["actual_date"] = actual_rows["Date"].to_numpy()
        signal_rows["actual"] = actual_rows["actual"].to_numpy(dtype=float)
        signal_rows["prediction"] = signal_rows["prediction"].to_numpy(dtype=float)
        aligned_parts.append(signal_rows)

    if not aligned_parts:
        return pd.DataFrame()

    aligned = pd.concat(aligned_parts, ignore_index=True)
    aligned["error"] = aligned["actual"].to_numpy(dtype=float) - aligned["prediction"].to_numpy(dtype=float)
    aligned["abs_error"] = np.abs(aligned["error"].to_numpy(dtype=float))
    aligned["direction_ok"] = np.sign(aligned["prediction"].to_numpy(dtype=float)) == np.sign(aligned["actual"].to_numpy(dtype=float))
    return aligned


def build_daily_regimes(aligned: pd.DataFrame, config: RegimeRuleConfig) -> pd.DataFrame:
    base = (
        aligned.drop_duplicates(["market", "code", "actual_date"])
        .loc[:, ["market", "code", "actual_date", "actual"]]
        .copy()
    )
    daily = (
        base.groupby(["market", "actual_date"], as_index=False)
        .agg(
            market_return=("actual", "mean"),
            breadth=("actual", lambda values: float(np.mean(np.asarray(values, dtype=float) > 0.0))),
            stock_count=("code", "nunique"),
        )
        .sort_values(["market", "actual_date"], kind="stable")
        .reset_index(drop=True)
    )

    output_parts: list[pd.DataFrame] = []
    for market, group in daily.groupby("market", sort=False):
        part = group.sort_values("actual_date", kind="stable").reset_index(drop=True).copy()
        part["market_return_20"] = part["market_return"].rolling(config.trend_window_fast, min_periods=10).mean()
        part["market_return_60"] = part["market_return"].rolling(config.trend_window_slow, min_periods=30).mean()
        part["breadth_20"] = part["breadth"].rolling(config.trend_window_fast, min_periods=10).mean()
        part["market_volatility_20"] = part["market_return"].rolling(config.trend_window_fast, min_periods=10).std()
        part["volatility_expanding_median"] = part["market_volatility_20"].expanding(min_periods=60).median()

        regimes = np.full(len(part), "sideways", dtype=object)
        fast_up = part["market_return_20"] > 0.0
        slow_up = part["market_return_60"] > 0.0
        fast_down = part["market_return_20"] < 0.0
        slow_down = part["market_return_60"] < 0.0
        breadth_strong = part["breadth_20"] >= config.breadth_up
        breadth_weak = part["breadth_20"] <= config.breadth_down
        high_vol = part["market_volatility_20"] >= part["volatility_expanding_median"]

        regimes[fast_up & slow_up & breadth_strong] = "uptrend"
        regimes[fast_down & slow_down & breadth_weak] = "downtrend"
        regimes[fast_up & (~slow_up) & breadth_strong] = "recovery"
        regimes[(slow_up | (part["market_return_60"] >= -0.0005)) & breadth_weak & (high_vol | fast_down)] = "distribution"
        part["regime"] = regimes
        output_parts.append(part)

    return pd.concat(output_parts, ignore_index=True)


def spearman_ic(prediction: pd.Series, actual: pd.Series) -> float:
    if len(prediction) < 5:
        return float("nan")
    if prediction.nunique(dropna=True) < 2 or actual.nunique(dropna=True) < 2:
        return float("nan")
    value = prediction.corr(actual, method="spearman")
    return float(value) if pd.notna(value) else float("nan")


def compute_panel_metrics(group: pd.DataFrame) -> dict[str, float]:
    ordered = group.sort_values("prediction", kind="stable").copy()
    ic = spearman_ic(ordered["prediction"], ordered["actual"])
    if ordered.empty:
        return {"spearman_ic": float("nan"), "quartile_return": float("nan")}
    ranks = ordered["prediction"].rank(method="first", pct=True)
    top = ordered.loc[ranks >= 0.75, "actual"]
    bottom = ordered.loc[ranks <= 0.25, "actual"]
    quartile_return = float(top.mean() - bottom.mean()) if len(top) and len(bottom) else float("nan")
    return {"spearman_ic": ic, "quartile_return": quartile_return}


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return float(drawdown.min())


def build_daily_metrics(group: pd.DataFrame, min_names_per_panel: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (actual_date, market), panel in group.groupby(["actual_date", "market"], sort=True):
        if panel["code"].nunique() < min_names_per_panel:
            continue
        metrics = compute_panel_metrics(panel)
        rows.append(
            {
                "actual_date": pd.Timestamp(actual_date),
                "market": str(market),
                "name_count": int(panel["code"].nunique()),
                "spearman_ic": metrics["spearman_ic"],
                "quartile_return": metrics["quartile_return"],
            }
        )

    if not rows:
        return pd.DataFrame(columns=["actual_date", "panel_count", "mean_spearman_ic", "quartile_return", "avg_name_count"])

    panel_df = pd.DataFrame(rows)
    daily = (
        panel_df.groupby("actual_date", as_index=False)
        .agg(
            panel_count=("market", "nunique"),
            mean_spearman_ic=("spearman_ic", "mean"),
            quartile_return=("quartile_return", "mean"),
            avg_name_count=("name_count", "mean"),
        )
        .sort_values("actual_date", kind="stable")
        .reset_index(drop=True)
    )
    return daily


def summarize_group(group: pd.DataFrame, min_names_per_panel: int) -> tuple[dict[str, object], pd.DataFrame]:
    error = group["error"].to_numpy(dtype=float)
    base = group["actual"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(np.square(error)))) if len(error) else float("nan")
    mae = float(np.mean(np.abs(error))) if len(error) else float("nan")
    base_loss = float(loss_fn(base)) if len(base) else float("nan")
    abs_loss = float(loss_fn(error)) if len(error) else float("nan")
    rel_score = float(1.0 - abs_loss / base_loss) if np.isfinite(base_loss) and base_loss > 0 else float("nan")

    daily = build_daily_metrics(group, min_names_per_panel)
    ic = daily["mean_spearman_ic"].dropna().to_numpy(dtype=float) if not daily.empty else np.asarray([], dtype=float)
    quartile_returns = daily["quartile_return"].dropna().to_numpy(dtype=float) if not daily.empty else np.asarray([], dtype=float)
    equity = np.cumprod(1.0 + quartile_returns) if len(quartile_returns) else np.asarray([], dtype=float)
    ic_t_stat = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 1 and ic.std(ddof=1) > 0 else float("nan")

    summary = {
        "row_count": int(len(group)),
        "day_count": int(group["actual_date"].nunique()),
        "code_count": int(group["code"].nunique()),
        "market_count": int(group["market"].nunique()),
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": float(group["direction_ok"].mean()) if len(group) else float("nan"),
        "base_loss": base_loss,
        "abs_loss": abs_loss,
        "rel_score": rel_score,
        "mean_spearman_ic": float(ic.mean()) if len(ic) else float("nan"),
        "ic_t_stat": ic_t_stat,
        "positive_ic_days": float(np.mean(ic > 0.0)) if len(ic) else float("nan"),
        "quartile_days": int(len(quartile_returns)),
        "quartile_mean_return": float(quartile_returns.mean()) if len(quartile_returns) else float("nan"),
        "quartile_hit_rate": float(np.mean(quartile_returns > 0.0)) if len(quartile_returns) else float("nan"),
        "quartile_equity": float(equity[-1]) if len(equity) else float("nan"),
        "quartile_max_drawdown": max_drawdown(equity),
    }
    return summary, daily


def build_group_summary(aligned: pd.DataFrame, min_names_per_panel: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    daily_rows: list[pd.DataFrame] = []

    for (model_name, split_name), model_split in aligned.groupby(["model", "split"], sort=True):
        markets = ["ALL", *sorted(model_split["market"].dropna().astype(str).unique())]
        for market_name in markets:
            market_df = model_split if market_name == "ALL" else model_split[model_split["market"] == market_name]
            if market_df.empty:
                continue

            regime_names = ["ALL", *sorted(market_df["regime"].dropna().astype(str).unique())]
            for regime_name in regime_names:
                group = market_df if regime_name == "ALL" else market_df[market_df["regime"] == regime_name]
                if group.empty:
                    continue
                summary, daily = summarize_group(group, min_names_per_panel)
                row = {
                    "model": model_name,
                    "split": split_name,
                    "market": market_name,
                    "regime": regime_name,
                    **summary,
                }
                summary_rows.append(row)
                if not daily.empty:
                    daily_rows.append(
                        daily.assign(
                            model=model_name,
                            split=split_name,
                            market=market_name,
                            regime=regime_name,
                        )
                    )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["split", "model", "market", "regime"],
        kind="stable",
    )
    daily_df = (
        pd.concat(daily_rows, ignore_index=True)
        .sort_values(["split", "model", "market", "regime", "actual_date"], kind="stable")
        if daily_rows
        else pd.DataFrame()
    )
    return summary_df, daily_df


def resolve_score_spec(aligned: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, ScoreSpec]:
    if args.score_column:
        if args.score_column not in aligned.columns:
            raise ValueError(f"Missing score column: {args.score_column}")
        series = pd.to_numeric(aligned[args.score_column], errors="coerce")
        if args.score_lower_is_better:
            series = -series
            return series, ScoreSpec(name=f"neg_{args.score_column}")
        return series, ScoreSpec(name=args.score_column)

    if "prediction_uncertainty" in aligned.columns:
        return -pd.to_numeric(aligned["prediction_uncertainty"], errors="coerce"), ScoreSpec(name="neg_prediction_uncertainty")

    normalizer_columns = [column for column in aligned.columns if column.startswith("__target_normalizer__")]
    if normalizer_columns:
        column = normalizer_columns[0]
        denom = pd.to_numeric(aligned[column], errors="coerce").abs().clip(lower=1e-6)
        score = aligned["prediction"].abs() / denom
        return score, ScoreSpec(name=f"abs_prediction_over_{column}")

    return aligned["prediction"].abs(), ScoreSpec(name="abs_prediction")


def select_top_coverage(group: pd.DataFrame, coverage: float) -> pd.DataFrame:
    if group.empty:
        return group
    keep_count = max(1, int(math.ceil(len(group) * coverage)))
    return group.nlargest(keep_count, "__score__", keep="all").head(keep_count).copy()


def build_selective_summary(
    aligned: pd.DataFrame,
    coverages: list[float],
    min_names_per_panel: int,
    score_spec: ScoreSpec,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, split_name), model_split in aligned.groupby(["model", "split"], sort=True):
        markets = ["ALL", *sorted(model_split["market"].dropna().astype(str).unique())]
        for market_name in markets:
            market_df = model_split if market_name == "ALL" else model_split[model_split["market"] == market_name]
            if market_df.empty:
                continue
            total_rows = int(len(market_df))
            for coverage in coverages:
                selected_parts: list[pd.DataFrame] = []
                for _, panel in market_df.groupby(["actual_date", "market"], sort=True):
                    selected_parts.append(select_top_coverage(panel, coverage))
                selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else market_df.iloc[0:0].copy()
                summary, _ = summarize_group(selected, min_names_per_panel)
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "market": market_name,
                        "score_name": score_spec.name,
                        "requested_coverage": float(coverage),
                        "actual_coverage": float(len(selected) / total_rows) if total_rows else float("nan"),
                        **summary,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["split", "model", "market", "requested_coverage"],
        kind="stable",
    )


def write_markdown(
    output_dir: Path,
    run_dir: Path,
    selected_models: list[str],
    selected_splits: list[str],
    summary_df: pd.DataFrame,
    selective_df: pd.DataFrame,
    score_spec: ScoreSpec,
) -> None:
    summary_path = output_dir / "summary.md"
    val_overall = summary_df[
        (summary_df["split"] == "val") & (summary_df["market"] == "ALL") & (summary_df["regime"] == "ALL")
    ].sort_values("rel_score", ascending=False, kind="stable")

    lines = [
        "# Regime Risk Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Models: `{', '.join(selected_models)}`",
        f"- Splits: `{', '.join(selected_splits)}`",
        f"- Selective score: `{score_spec.name}`",
        "- Regime labels are diagnostic only and are computed from realized cross-sectional market return and breadth on `actual_date`.",
        "",
    ]

    if not val_overall.empty:
        best_row = val_overall.iloc[0]
        lines.extend(
            [
                "## Best Validation Overall",
                "",
                f"- Model: `{best_row['model']}`",
                f"- rel_score: `{float(best_row['rel_score']):+.5f}`",
                f"- mean daily Spearman IC: `{float(best_row['mean_spearman_ic']):+.5f}`",
                f"- IC t-stat: `{float(best_row['ic_t_stat']):+.2f}`",
                f"- quartile equity: `{float(best_row['quartile_equity']):.3f}`",
                f"- quartile max drawdown: `{float(best_row['quartile_max_drawdown']):.1%}`",
                "",
            ]
        )

    per_market_val = summary_df[
        (summary_df["split"] == "val") & (summary_df["regime"] == "ALL") & (summary_df["market"] != "ALL")
    ].sort_values(["market", "rel_score"], ascending=[True, False], kind="stable")
    if not per_market_val.empty:
        lines.extend(
            [
                "## Validation By Market",
                "",
                "| Market | Model | rel_score | Mean IC | IC t-stat | Quartile equity | Max DD |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for market_name, market_df in per_market_val.groupby("market", sort=True):
            row = market_df.iloc[0]
            lines.append(
                "| "
                f"`{market_name}` | `{row['model']}` | {float(row['rel_score']):+.4f} | "
                f"{float(row['mean_spearman_ic']):+.4f} | {float(row['ic_t_stat']):+.2f} | "
                f"{float(row['quartile_equity']):.3f} | {float(row['quartile_max_drawdown']):.1%} |"
            )
        lines.append("")

    if not val_overall.empty and not selective_df.empty:
        best_model = str(val_overall.iloc[0]["model"])
        selective_best = selective_df[
            (selective_df["split"] == "val") & (selective_df["market"] == "ALL") & (selective_df["model"] == best_model)
        ].sort_values("requested_coverage", kind="stable")
        if not selective_best.empty:
            lines.extend(
                [
                    "## Validation Selective Curve",
                    "",
                    f"Best overall validation model: `{best_model}`",
                    "",
                    "| Coverage | rel_score | Mean IC | Quartile equity | Hit rate |",
                    "| ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for _, row in selective_best.iterrows():
                lines.append(
                    "| "
                    f"{float(row['actual_coverage']):.1%} | {float(row['rel_score']):+.4f} | "
                    f"{float(row['mean_spearman_ic']):+.4f} | {float(row['quartile_equity']):.3f} | "
                    f"{float(row['quartile_hit_rate']):.1%} |"
                )
            lines.append("")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    run_config = load_run_config(run_dir)
    default_market = str(run_config.get("market") or "VN")

    pred_df = load_predictions(run_dir)
    selected_models = parse_csv_list(args.models) or sorted(pred_df["model"].dropna().astype(str).unique().tolist())
    selected_splits = parse_csv_list(args.splits) or sorted(pred_df["split"].dropna().astype(str).unique().tolist())
    aligned = align_predictions(pred_df, set(selected_models), set(selected_splits))
    if aligned.empty:
        raise ValueError("No aligned predictions available for the requested models/splits.")

    aligned["market"] = aligned["code"].map(lambda value: infer_market_from_code(str(value), default_market))
    regime_df = build_daily_regimes(aligned, RegimeRuleConfig())
    aligned = aligned.merge(regime_df[["market", "actual_date", "regime"]], on=["market", "actual_date"], how="left")
    aligned["regime"] = aligned["regime"].fillna("sideways")

    summary_df, daily_df = build_group_summary(aligned, args.min_names_per_panel)
    score_values, score_spec = resolve_score_spec(aligned, args)
    aligned = aligned.assign(__score__=pd.to_numeric(score_values, errors="coerce").fillna(-np.inf))
    selective_df = build_selective_summary(aligned, parse_float_list(args.coverage_grid), args.min_names_per_panel, score_spec)

    output_dir = report_benchmark_path(run_dir, args.output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "summary_by_group.csv", index=False)
    if not daily_df.empty:
        daily_df.to_csv(output_dir / "daily_metrics.csv", index=False)
    selective_df.to_csv(output_dir / "selective_summary.csv", index=False)
    metadata = {
        "run_dir": str(run_dir),
        "models": selected_models,
        "splits": selected_splits,
        "score_name": score_spec.name,
        "coverage_grid": parse_float_list(args.coverage_grid),
        "min_names_per_panel": int(args.min_names_per_panel),
    }
    output_dir.joinpath("summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_markdown(output_dir, run_dir, selected_models, selected_splits, summary_df, selective_df, score_spec)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
