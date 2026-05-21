from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metric import directional_accuracy, evaluate
from src.reporting.layout import (
    report_backtest_path,
    report_core_path,
    report_metric_series_path,
    report_plot_path,
    report_diagnostic_path,
)
from src.reporting.feature_report import write_feature_formula_report
from src.reporting.standards import ReportingStandard, get_default_reporting_standard
from src.visualization.model_plots import (
    save_actual_vs_prediction_plot,
    save_equity_curve_plot,
    save_rel_score_hist_plot,
)


SPLIT_ORDER = ("train", "val", "test")


def _cleanup_hidden_split_artifacts(run_dir: Path, hidden_splits: set[str]) -> None:
    if not hidden_splits:
        return
    report_root = run_dir / "reports"
    for split_name in hidden_splits:
        patterns = (
            f"metric_series_*_{split_name}.csv",
            f"large_error_*_{split_name}.csv",
            f"error_hist_*_{split_name}.png",
            f"quartile_long_short_*_{split_name}.csv",
            f"quartile_long_short_equity_*_{split_name}.png",
        )
        for pattern in patterns:
            for path in run_dir.glob(pattern):
                if path.is_file():
                    path.unlink(missing_ok=True)
            if report_root.exists():
                for path in report_root.glob(f"**/{pattern}"):
                    if path.is_file():
                        path.unlink(missing_ok=True)


def _strip_hidden_split_payload(value: object, hidden_splits: set[str]) -> object:
    if isinstance(value, dict):
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in hidden_splits:
                continue
            if any(key_text == f"{split_name}_score" for split_name in hidden_splits):
                continue
            if any(key_text.startswith(f"{split_name}_") for split_name in hidden_splits):
                continue
            cleaned[key] = _strip_hidden_split_payload(item, hidden_splits)
        return cleaned
    if isinstance(value, list):
        return [_strip_hidden_split_payload(item, hidden_splits) for item in value]
    return value


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, quantile))


def _compute_split_summary(
    split_df: pd.DataFrame,
    standard: ReportingStandard,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    work = split_df.copy()
    if {"code", "Date"}.issubset(work.columns):
        work = work.sort_values(["code", "Date"], kind="stable")
    work["error"] = work["actual"] - work["prediction"]
    work["abs_error"] = work["error"].abs()
    group_ids = work["code"].to_numpy() if "code" in work.columns else None
    eval_result = evaluate(
        work["prediction"].to_numpy(dtype=float),
        work["actual"].to_numpy(dtype=float),
        group_ids=group_ids,
    )
    metric_series_df = pd.DataFrame(
        {
            "error": eval_result["error"],
            "base": eval_result["base"],
        }
    )
    rmse = float(np.sqrt(np.mean(np.square(work["error"].to_numpy(dtype=float))))) if not work.empty else float("nan")
    mae = float(work["abs_error"].mean()) if not work.empty else float("nan")
    q_low_label, q_high_label = standard.error_quantile_labels
    q_low, q_high = standard.error_quantiles
    large_error_threshold = _safe_quantile(work["abs_error"].to_numpy(dtype=float), standard.large_error_quantile)
    summary = {
        "row_count": int(len(work)),
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": float(directional_accuracy(work["prediction"].to_numpy(dtype=float), work["actual"].to_numpy(dtype=float), group_ids=group_ids)),
        "base_loss": float(eval_result["base_loss"]),
        "abs_loss": float(eval_result["abs_loss"]),
        "rel_score": float(eval_result["rel_score"]),
        f"error_{q_low_label}": _safe_quantile(work["error"].to_numpy(dtype=float), q_low),
        f"error_{q_high_label}": _safe_quantile(work["error"].to_numpy(dtype=float), q_high),
        "large_error_quantile": float(standard.large_error_quantile),
        "large_error_threshold_abs": large_error_threshold,
        "large_error_count": int((work["abs_error"] >= large_error_threshold).sum()) if np.isfinite(large_error_threshold) else 0,
        "large_error_share": float((work["abs_error"] >= large_error_threshold).mean()) if np.isfinite(large_error_threshold) and len(work) else 0.0,
    }
    return summary, work, metric_series_df


def _save_error_histogram(
    run_dir: Path,
    model_name: str,
    split_name: str,
    split_label: str,
    standard: ReportingStandard,
    split_df: pd.DataFrame,
    summary: dict[str, float | int | str],
) -> None:
    errors = split_df["error"].to_numpy(dtype=float)
    if len(errors) == 0:
        return
    q_low_label, q_high_label = standard.error_quantile_labels
    q_low_key = f"error_{q_low_label}"
    q_high_key = f"error_{q_high_label}"
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(errors, bins=60, color="#4e79a7", alpha=0.8)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.35)
    ax.axvline(float(summary[q_low_key]), color="#f28e2b", linewidth=1.4, linestyle="--", label=q_low_label)
    ax.axvline(float(summary[q_high_key]), color="#e15759", linewidth=1.4, linestyle="--", label=q_high_label)
    ax.set_title(f"{model_name} | {split_label} | Error histogram")
    ax.set_xlabel("error = actual - prediction")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    stats_text = (
        f"rmse={summary['rmse']:.4f}\n"
        f"rel_score={summary['rel_score']:.4f}\n"
        f"{q_low_label}={summary[q_low_key]:.4f}\n"
        f"{q_high_label}={summary[q_high_key]:.4f}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(report_plot_path(run_dir, f"error_hist_{model_name}_{split_name}.png"), dpi=200)
    plt.close(fig)


def _build_quartile_long_short_curve(split_df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, date_df in split_df.groupby("Date", sort=True):
        date_work = date_df.dropna(subset=["prediction", "actual"]).sort_values("prediction", kind="stable")
        count = len(date_work)
        selection_size = min(count // 2, max(1, int(np.floor(count * quantile))))
        if selection_size <= 0:
            continue
        longs = date_work.tail(selection_size)
        shorts = date_work.head(selection_size)
        long_return = float(longs["actual"].mean())
        short_return = float(shorts["actual"].mean())
        rows.append(
            {
                "Date": pd.Timestamp(date),
                "long_count": int(len(longs)),
                "short_count": int(len(shorts)),
                "long_mean_prediction": float(longs["prediction"].mean()),
                "short_mean_prediction": float(shorts["prediction"].mean()),
                "long_return": long_return,
                "short_return": short_return,
                "long_short_return": long_return - short_return,
            }
        )
    curve_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    if curve_df.empty:
        return curve_df
    curve_df["equity"] = (1.0 + curve_df["long_short_return"]).cumprod()
    return curve_df


def _summarize_curve(curve_df: pd.DataFrame, quantile: float) -> dict[str, float | int]:
    if curve_df.empty:
        return {
            "selection_quantile": float(quantile),
            "active_days": 0,
            "avg_daily_return": 0.0,
            "cumulative_return": 0.0,
            "final_equity": 1.0,
        }
    return {
        "selection_quantile": float(quantile),
        "active_days": int(len(curve_df)),
        "avg_daily_return": float(curve_df["long_short_return"].mean()),
        "cumulative_return": float(curve_df["long_short_return"].sum()),
        "final_equity": float(curve_df["equity"].iloc[-1]),
    }


def _save_large_error_report(run_dir: Path, model_name: str, split_name: str, split_df: pd.DataFrame, threshold: float) -> None:
    if not np.isfinite(threshold):
        return
    large_df = split_df.loc[split_df["abs_error"] >= threshold].copy()
    if large_df.empty:
        return
    keep_columns = [column for column in ("Date", "code", "prediction", "actual", "error", "abs_error", "split", "model") if column in large_df.columns]
    large_df = large_df.sort_values("abs_error", ascending=False, kind="stable")
    large_df.loc[:, keep_columns].to_csv(
        report_diagnostic_path(run_dir, f"large_error_{model_name}_{split_name}.csv"),
        index=False,
    )


def _model_family_name(model_name: str) -> str:
    for prefix in ("lstm_quantile", "lstm_signmag", "lstm_attention", "lstm_signal", "lstm_pcie_lite", "lstm_event", "lstm"):
        if model_name == prefix or model_name.startswith(f"{prefix}_"):
            return prefix
    return "baseline"


def _primary_split_name(visible_splits: list[str]) -> str:
    for split_name in ("test", "val", "train"):
        if split_name in visible_splits:
            return split_name
    return visible_splits[0]


def _build_report_leaderboard(
    metrics: dict[str, dict[str, dict[str, float]]],
    evaluation_summary: dict[str, dict[str, dict[str, float | int | str]]],
    report_model_names: set[str],
    visible_splits: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    primary_split = _primary_split_name(visible_splits)
    for model_name in sorted(report_model_names):
        if model_name not in metrics:
            continue
        row: dict[str, object] = {
            "model": model_name,
            "family": _model_family_name(model_name),
            "primary_split": primary_split,
        }
        for split_name in SPLIT_ORDER:
            split_metrics = metrics.get(model_name, {}).get(split_name, {})
            split_summary = evaluation_summary.get(model_name, {}).get(split_name, {})
            row[f"{split_name}_rel_score"] = split_metrics.get("rel_score")
            row[f"{split_name}_directional_accuracy"] = split_metrics.get("directional_accuracy")
            row[f"{split_name}_rmse"] = split_metrics.get("rmse")
            row[f"{split_name}_error_q2"] = split_summary.get("error_q2")
            row[f"{split_name}_error_q8"] = split_summary.get("error_q8")
            quartile = split_summary.get("quartile_long_short")
            if isinstance(quartile, dict):
                row[f"{split_name}_quartile_final_equity"] = quartile.get("final_equity")
            else:
                row[f"{split_name}_quartile_final_equity"] = None
        row["primary_rel_score"] = row.get(f"{primary_split}_rel_score")
        row["primary_directional_accuracy"] = row.get(f"{primary_split}_directional_accuracy")
        rows.append(row)
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard
    leaderboard = leaderboard.sort_values(
        ["primary_rel_score", "primary_directional_accuracy", "model"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    return leaderboard


def _write_run_overview(
    run_dir: Path,
    leaderboard: pd.DataFrame,
    visible_splits: list[str],
    feature_columns: tuple[str, ...] | None = None,
) -> None:
    config_path = report_core_path(run_dir, "config.json")
    config_payload = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    family_summary_path = report_core_path(run_dir, "family_selection_summary.json")
    family_summary = json.loads(family_summary_path.read_text(encoding="utf-8")) if family_summary_path.exists() else {}
    feature_count = len(feature_columns or ())
    stocks_raw = config_payload.get("stocks")
    if isinstance(stocks_raw, str):
        stock_count = len([item for item in stocks_raw.split(",") if item.strip()])
    elif isinstance(stocks_raw, list):
        stock_count = len(stocks_raw)
    else:
        stock_count = 0

    lines = [
        "# Run Overview",
        "",
        "Open this file first. Detailed diagnostic files now live under `reports/diagnostics/`.",
        "",
        "## Read Order",
        "",
        "1. `reports/core/run_overview.md`",
        "2. `reports/core/leaderboard.csv`",
        "3. `reports/core/family_selection_summary.json`",
        "4. top model plots under `reports/plots/`",
        "5. only then look at `reports/backtests/` and `reports/diagnostics/`",
        "",
        "## Run Setup",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| run_name | `{run_dir.name}` |",
        f"| target_mode | `{config_payload.get('target_mode', 'unknown')}` |",
        f"| sector | `{config_payload.get('sector', '-')}` |",
        f"| stock_count | `{stock_count}` |",
        f"| stocks | `{stocks_raw or '-'}` |",
        f"| window_size | `{config_payload.get('window_size', '-')}` |",
        f"| loss | `{config_payload.get('loss', '-')}` |",
        f"| sequence_normalization | `{config_payload.get('sequence_normalization', '-')}` |",
        f"| feature_phase | `{config_payload.get('feature_phase', '-')}` |",
        f"| feature_count | `{feature_count}` |",
        f"| visible_splits | `{', '.join(visible_splits)}` |",
        "",
    ]

    if not leaderboard.empty:
        primary_split = str(leaderboard.iloc[0]["primary_split"])
        lines.extend(
            [
                "## Leaderboard",
                "",
                f"Primary ranking split: `{primary_split}`",
                "",
                "| Rank | Model | Family | Primary rel_score | Primary dir_acc | Error q2 | Error q8 | Primary quartile equity |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in leaderboard.head(8).to_dict(orient="records"):
            split_name = str(row["primary_split"])
            rel_score = row.get("primary_rel_score")
            dir_acc = row.get("primary_directional_accuracy")
            final_equity = row.get(f"{split_name}_quartile_final_equity")
            error_q2 = row.get(f"{split_name}_error_q2")
            error_q8 = row.get(f"{split_name}_error_q8")
            rel_text = f"{float(rel_score):+.4f}" if isinstance(rel_score, (int, float)) else "-"
            dir_text = f"{float(dir_acc):.4f}" if isinstance(dir_acc, (int, float)) else "-"
            equity_text = f"{float(final_equity):.3f}" if isinstance(final_equity, (int, float)) else "-"
            q2_text = f"{float(error_q2):+.4f}" if isinstance(error_q2, (int, float)) else "-"
            q8_text = f"{float(error_q8):+.4f}" if isinstance(error_q8, (int, float)) else "-"
            lines.append(
                f"| {int(row['rank'])} | `{row['model']}` | `{row['family']}` | {rel_text} | {dir_text} | {q2_text} | {q8_text} | {equity_text} |"
            )
        top_model = str(leaderboard.iloc[0]["model"])
        lines.extend(
            [
                "",
                "## Open Next",
                "",
                f"- `reports/plots/actual_vs_prediction_{top_model}.png`",
                f"- `reports/plots/error_hist_{top_model}_{primary_split}.png`",
                f"- `reports/plots/rel_score_hist_{top_model}.png`",
                f"- `reports/backtests/quartile_long_short_{top_model}_{primary_split}.csv`",
                f"- `reports/backtests/quartile_long_short_equity_{top_model}_{primary_split}.png`",
                "",
            ]
        )

    if family_summary:
        lines.extend(
            [
                "## Family Selection",
                "",
                "| Family | best_by_val | top2_by_val |",
                "| --- | --- | --- |",
            ]
        )
        for family_name, payload in sorted(family_summary.items()):
            if not isinstance(payload, dict):
                continue
            best_by_val = payload.get("best_by_val", "-")
            top2 = payload.get("top2_by_val", [])
            top2_text = ", ".join(top2) if isinstance(top2, list) else "-"
            lines.append(f"| `{family_name}` | `{best_by_val}` | `{top2_text or '-'}` |")
        lines.append("")

    lines.extend(
        [
            "## Report Layout",
            "",
            "- `reports/core/`: summaries and small metadata",
            "- `reports/plots/`: shortlist plots, including actual-vs-prediction, error histograms, and rel_score histograms",
            "- `reports/backtests/`: shortlist long/short diagnostics",
            "- `reports/diagnostics/`: large-error dumps and training histories",
            "- `reports/metric_series/`: raw metric-series CSVs used by diagnostics",
            "",
        ]
    )

    report_core_path(run_dir, "run_overview.md").write_text("\n".join(lines), encoding="utf-8")


def refresh_run_report_artifacts(
    run_dir: Path,
    prediction_df: pd.DataFrame,
    metrics: dict[str, dict[str, dict[str, float]]],
    metric_details: dict[str, dict[str, dict[str, float | int]]],
    report_model_names: set[str],
    *,
    standard: ReportingStandard | None = None,
    feature_columns: tuple[str, ...] | None = None,
    reveal_out_sample: bool = False,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, dict[str, float | int]]]]:
    standard = standard or get_default_reporting_standard()
    visible_splits = list(standard.default_report_splits)
    if reveal_out_sample and standard.holdout_split not in visible_splits:
        visible_splits.append(standard.holdout_split)
    visible_split_set = set(visible_splits)
    hidden_splits = set(SPLIT_ORDER) - visible_split_set
    _cleanup_hidden_split_artifacts(run_dir, hidden_splits)
    private_dir = run_dir / "holdout_private"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_predictions_path = private_dir / "predictions_full.csv"
    if not private_predictions_path.exists() or reveal_out_sample:
        prediction_df.to_csv(private_predictions_path, index=False)
    prediction_df[prediction_df["split"].isin(visible_splits)].to_csv(
        report_core_path(run_dir, "predictions.csv"),
        index=False,
    )
    evaluation_summary: dict[str, dict[str, dict[str, float | int | str]]] = {}

    for model_name in sorted(prediction_df["model"].dropna().unique().tolist()):
        model_df = prediction_df[prediction_df["model"] == model_name].copy()
        if model_df.empty:
            continue
        evaluation_summary.setdefault(model_name, {})
        metrics.setdefault(model_name, {})
        metric_details.setdefault(model_name, {})

        for split_name in SPLIT_ORDER:
            if split_name not in visible_split_set:
                continue
            split_df = model_df[model_df["split"] == split_name].copy()
            if split_df.empty:
                continue
            split_label = standard.split_label(split_name)
            window = standard.window(split_name)
            summary, enriched_split_df, metric_series_df = _compute_split_summary(split_df, standard)
            summary["split"] = split_name
            summary["split_label"] = split_label
            summary["period_display"] = window.display
            evaluation_summary[model_name][split_name] = summary

            metrics[model_name].setdefault(split_name, {})
            metrics[model_name][split_name]["rmse"] = float(summary["rmse"])
            metrics[model_name][split_name]["mae"] = float(summary["mae"])
            metrics[model_name][split_name]["directional_accuracy"] = float(summary["directional_accuracy"])
            metrics[model_name][split_name]["base_loss"] = float(summary["base_loss"])
            metrics[model_name][split_name]["abs_loss"] = float(summary["abs_loss"])
            metrics[model_name][split_name]["rel_score"] = float(summary["rel_score"])

            metric_details[model_name][split_name] = {
                "split_label": split_label,
                "period_display": window.display,
                "row_count": int(summary["row_count"]),
                "rmse": float(summary["rmse"]),
                "mae": float(summary["mae"]),
                "directional_accuracy": float(summary["directional_accuracy"]),
                "base_loss": float(summary["base_loss"]),
                "abs_loss": float(summary["abs_loss"]),
                "rel_score": float(summary["rel_score"]),
                f"error_{standard.error_quantile_labels[0]}": float(summary[f"error_{standard.error_quantile_labels[0]}"]),
                f"error_{standard.error_quantile_labels[1]}": float(summary[f"error_{standard.error_quantile_labels[1]}"]),
                "large_error_quantile": float(summary["large_error_quantile"]),
                "large_error_threshold_abs": float(summary["large_error_threshold_abs"]),
                "large_error_count": int(summary["large_error_count"]),
                "large_error_share": float(summary["large_error_share"]),
            }

            if model_name not in report_model_names:
                continue

            metric_series_name = f"metric_series_{model_name}_{split_name}.csv"
            metric_series_df.to_csv(run_dir / metric_series_name, index=False)
            metric_series_df.to_csv(report_metric_series_path(run_dir, metric_series_name), index=False)

            _save_error_histogram(run_dir, model_name, split_name, split_label, standard, enriched_split_df, summary)
            _save_large_error_report(
                run_dir,
                model_name,
                split_name,
                enriched_split_df.assign(split=split_name, model=model_name),
                float(summary["large_error_threshold_abs"]),
            )

            curve_df = _build_quartile_long_short_curve(enriched_split_df, standard.long_short_quantile)
            if not curve_df.empty:
                curve_name = f"quartile_long_short_{model_name}_{split_name}.csv"
                curve_df.to_csv(report_backtest_path(run_dir, curve_name), index=False)
                curve_plot_df = curve_df[["Date", "equity"]].copy()
                curve_plot_df["label"] = f"{split_label} q-long/q-short"
                save_equity_curve_plot(
                    curve_plot_df[["Date", "label", "equity"]],
                    report_backtest_path(run_dir, f"quartile_long_short_equity_{model_name}_{split_name}.png"),
                    f"{model_name} | {split_label} | Long top 1/4, short bottom 1/4",
                )
                evaluation_summary[model_name][split_name]["quartile_long_short"] = _summarize_curve(
                    curve_df,
                    standard.long_short_quantile,
                )

        if model_name in report_model_names:
            report_prediction_df = prediction_df[prediction_df["split"].isin(visible_splits)].copy()
            save_actual_vs_prediction_plot(run_dir, report_prediction_df, model_name, split_names=visible_splits)
            save_rel_score_hist_plot(run_dir, model_name, split_names=tuple(visible_splits))

    standard_payload = standard.to_payload()
    standard_payload["holdout_policy"] = {
        "default_visible_splits": visible_splits,
        "holdout_split": standard.holdout_split,
        "out_sample_revealed": bool(reveal_out_sample),
    }
    for model_name in list(metrics.keys()):
        metrics[model_name] = {
            split_name: split_metrics
            for split_name, split_metrics in metrics[model_name].items()
            if split_name in visible_split_set
        }
    for model_name in list(metric_details.keys()):
        metric_details[model_name] = {
            split_name: split_details
            for split_name, split_details in metric_details[model_name].items()
            if split_name in visible_split_set
        }

    with report_core_path(run_dir, "reporting_standard.json").open("w", encoding="utf-8") as f:
        json.dump(standard_payload, f, indent=2)
    with report_core_path(run_dir, "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    family_summary_path = report_core_path(run_dir, "family_selection_summary.json")
    if family_summary_path.exists():
        with family_summary_path.open("r", encoding="utf-8") as f:
            family_summary = json.load(f)
        if hidden_splits:
            family_summary = _strip_hidden_split_payload(family_summary, hidden_splits)
            with family_summary_path.open("w", encoding="utf-8") as f:
                json.dump(family_summary, f, indent=2)

    active_features = tuple(feature_columns or ())
    if active_features:
        write_feature_formula_report(report_core_path(run_dir, "feature_formula_report.md"), active_features)

    leaderboard = _build_report_leaderboard(metrics, evaluation_summary, report_model_names, visible_splits)
    if not leaderboard.empty:
        leaderboard.to_csv(report_core_path(run_dir, "leaderboard.csv"), index=False)
    _write_run_overview(run_dir, leaderboard, visible_splits, active_features)

    return metrics, metric_details
