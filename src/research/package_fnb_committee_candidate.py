from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_RUNS = ROOT / "data/processed/assets/data_info_vn/history/training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a committee candidate with backtests and plots.")
    parser.add_argument("--standalone-run", type=Path, required=True)
    parser.add_argument("--standalone-model", default="lstm_best_by_val")
    parser.add_argument("--committee-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="F&B committee candidate")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_small_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_committee_packaged_run(target_dir: Path, committee_dir: Path, committee_summary: dict) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    core_dir = target_dir / "reports/core"
    core_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "reports/backtests").mkdir(parents=True, exist_ok=True)

    preds_path = committee_dir / "best_committee_predictions.csv"
    preds = pd.read_csv(preds_path)
    packaged = preds.rename(columns={"prediction_committee": "prediction"}).copy()
    packaged["model"] = "committee_best_by_val"
    if "target" not in packaged.columns:
        packaged["target"] = packaged["actual"]
    ordered = ["code", "Date", "target", "split", "model", "prediction", "actual"]
    extra = [col for col in packaged.columns if col not in ordered]
    packaged = packaged[ordered + extra]
    packaged.to_csv(core_dir / "predictions.csv", index=False)

    best_committee = committee_summary["best_committee"]
    stability = committee_summary.get("best_committee_stability") or {}
    config = {
        "target_mode": "return",
        "target_column": "target_next_return",
        "advisor_packaged": True,
        "source_type": "committee",
        "source_committee_summary": str(committee_dir / "best_committee_summary.json"),
        "source_predictions": str(preds_path),
        "expert_run": committee_summary["expert_run"],
        "market_run": committee_summary["market_run"],
        "expert_model": best_committee["expert_model"],
        "market_model": best_committee["market_model"],
        "selection_mode": committee_summary["selection_mode"],
        "committee_method": best_committee["method"],
        "committee_weight_expert": best_committee["weight_expert"],
        "committee_code_count": best_committee["code_count"],
        "committee_overlap_codes": best_committee["overlap_codes"].split(","),
        "committee_agreement_rate": best_committee["agreement_rate"],
        "committee_test_rel_score": best_committee["committee_test_rel_score"],
        "committee_val_rel_score": best_committee["committee_val_rel_score"],
        "stable_weight_min": stability.get("stable_weight_min"),
        "stable_weight_max": stability.get("stable_weight_max"),
        "stable_weight_count": stability.get("stable_weight_count"),
        "stable_test_rel_score_median": stability.get("stable_test_rel_score_median"),
        "stable_test_rel_score_mean": stability.get("stable_test_rel_score_mean"),
    }
    with (core_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    metrics = {
        "committee_best_by_val": {
            "val": {
                "rel_score": float(best_committee["committee_val_rel_score"]),
                "abs_loss": float(best_committee["committee_val_abs_loss"]),
            },
            "test": {
                "rel_score": float(best_committee["committee_test_rel_score"]),
                "abs_loss": float(best_committee["committee_test_abs_loss"]),
            },
        }
    }
    with (core_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return target_dir


def run_backtest(run_dir: Path, model: str, suffix: str) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "src/backtesting/threshold_backtest.py"),
        str(run_dir),
        "--models",
        model,
        "--output-suffix",
        suffix,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    summary_path = run_dir / f"reports/backtests/threshold_backtest_summary_{suffix}.json"
    return load_json(summary_path)


def compute_bias_stats(predictions_path: Path, model: str) -> dict[str, float]:
    df = pd.read_csv(predictions_path)
    test = df[(df["split"] == "test") & (df["model"] == model)].copy()
    pred_abs_mean = float(test["prediction"].abs().mean())
    actual_abs_mean = float(test["actual"].abs().mean())
    return {
        "test_rows": int(len(test)),
        "pred_pos_rate": float((test["prediction"] > 0.0).mean()),
        "actual_pos_rate": float((test["actual"] > 0.0).mean()),
        "pred_abs_mean": pred_abs_mean,
        "actual_abs_mean": actual_abs_mean,
        "pred_abs_over_actual_abs": float(pred_abs_mean / actual_abs_mean) if actual_abs_mean else 0.0,
    }


def load_test_predictions(predictions_path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    test = df[(df["split"] == "test") & (df["model"] == model)].copy()
    test["Date"] = pd.to_datetime(test["Date"])
    return test.sort_values(["Date", "code"]).reset_index(drop=True)


def build_trade_frame(test_df: pd.DataFrame) -> pd.DataFrame:
    active = test_df[test_df["prediction"] >= 0.0].copy()
    daily = active.groupby("Date", as_index=False)["actual"].mean().rename(columns={"actual": "strategy_return"})
    daily["equity"] = (1.0 + daily["strategy_return"]).cumprod()
    return daily


def compute_sign_agreement(test_df: pd.DataFrame) -> float:
    nonzero = test_df[(test_df["prediction"] != 0.0) & (test_df["actual"] != 0.0)].copy()
    if nonzero.empty:
        return 0.0
    return float((nonzero["prediction"] * nonzero["actual"] > 0.0).mean())


def build_signal_bucket_frame(test_df: pd.DataFrame, label: str) -> pd.DataFrame:
    ranked = test_df.copy()
    ranked["bucket"] = pd.qcut(
        ranked["prediction"].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    )
    summary = (
        ranked.groupby("bucket", as_index=False)
        .agg(
            prediction_mean=("prediction", "mean"),
            actual_mean=("actual", "mean"),
            actual_hit_rate=("actual", lambda x: float((x >= 0.0).mean())),
            count=("actual", "size"),
        )
        .sort_values("bucket")
    )
    summary["model_label"] = label
    return summary


def build_positive_precision_curve(test_df: pd.DataFrame, label: str) -> pd.DataFrame:
    positive = test_df[test_df["prediction"] > 0.0].copy()
    if positive.empty:
        return pd.DataFrame(columns=["quantile", "threshold", "avg_actual_return", "hit_rate", "trade_count", "model_label"])
    quantiles = [0.5, 0.6, 0.7, 0.8, 0.9]
    rows: list[dict[str, object]] = []
    for quantile in quantiles:
        threshold = float(positive["prediction"].quantile(quantile))
        active = positive[positive["prediction"] >= threshold].copy()
        rows.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "avg_actual_return": float(active["actual"].mean()) if not active.empty else 0.0,
                "hit_rate": float((active["actual"] >= 0.0).mean()) if not active.empty else 0.0,
                "trade_count": int(len(active)),
                "model_label": label,
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    target_dir: Path,
    standalone_predictions_path: Path,
    committee_predictions_path: Path,
    standalone_model: str,
    committee_model: str,
    standalone_metrics: dict,
    committee_summary: dict,
    standalone_bias: dict,
    committee_bias: dict,
    label: str,
) -> None:
    plots_dir = target_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    standalone_test = load_test_predictions(standalone_predictions_path, standalone_model)
    committee_test = load_test_predictions(committee_predictions_path, committee_model)
    overlap_codes = [
        code.strip()
        for code in committee_summary["best_committee"]["overlap_codes"].split(",")
        if code.strip()
    ]

    standalone_daily = (
        standalone_test.groupby("Date", as_index=False)[["actual", "prediction"]]
        .mean()
        .rename(columns={"prediction": "standalone_prediction"})
    )
    committee_daily = (
        committee_test.groupby("Date", as_index=False)[["actual", "prediction"]]
        .mean()
        .rename(columns={"actual": "committee_actual", "prediction": "committee_prediction"})
    )
    compare = standalone_daily.merge(
        committee_daily[["Date", "committee_prediction"]],
        on="Date",
        how="inner",
    ).sort_values("Date")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(compare["Date"], compare["actual"], label="Actual mean return", linewidth=1.4, color="#222222")
    ax.plot(compare["Date"], compare["standalone_prediction"], label="Standalone prediction", linewidth=1.2, color="#1f77b4")
    ax.plot(compare["Date"], compare["committee_prediction"], label="Committee prediction", linewidth=1.2, color="#d62728")
    ax.set_title(f"{label}: Actual vs Prediction Mean by Date")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_actual_vs_predictions_mean.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    scatter_specs = [
        (axes[0], standalone_test, "Standalone"),
        (axes[1], committee_test, "Committee"),
    ]
    for ax, df, title in scatter_specs:
        ax.scatter(df["actual"], df["prediction"], s=9, alpha=0.28, color="#1f77b4" if title == "Standalone" else "#d62728")
        lim = max(df["actual"].abs().max(), df["prediction"].abs().max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1.0, color="#555555")
        ax.axhline(0.0, linewidth=0.8, color="#888888")
        ax.axvline(0.0, linewidth=0.8, color="#888888")
        ax.set_title(f"{title}: Test Actual vs Prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Prediction")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_actual_vs_prediction_scatter.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    hist_specs = [
        (axes[0], standalone_test, "Standalone"),
        (axes[1], committee_test, "Committee"),
    ]
    for ax, df, title in hist_specs:
        ax.hist(df["actual"].abs(), bins=40, alpha=0.55, label="|actual|", color="#222222")
        ax.hist(df["prediction"].abs(), bins=40, alpha=0.65, label="|prediction|", color="#1f77b4" if title == "Standalone" else "#d62728")
        ax.set_title(f"{title}: |Actual| vs |Prediction|")
        ax.set_xlabel("Absolute return")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_abs_return_histogram.png", dpi=160)
    plt.close(fig)

    compare_roll = compare.copy()
    compare_roll["actual_roll20"] = compare_roll["actual"].rolling(20, min_periods=5).mean()
    compare_roll["standalone_roll20"] = compare_roll["standalone_prediction"].rolling(20, min_periods=5).mean()
    compare_roll["committee_roll20"] = compare_roll["committee_prediction"].rolling(20, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(compare_roll["Date"], compare_roll["actual_roll20"], label="Actual 20d mean", linewidth=1.6, color="#222222")
    ax.plot(compare_roll["Date"], compare_roll["standalone_roll20"], label="Standalone 20d mean", linewidth=1.3, color="#1f77b4")
    ax.plot(compare_roll["Date"], compare_roll["committee_roll20"], label="Committee 20d mean", linewidth=1.3, color="#d62728")
    ax.set_title(f"{label}: 20-Day Rolling Mean")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_rolling20_actual_vs_predictions.png", dpi=160)
    plt.close(fig)

    selected_codes = overlap_codes[:4] if overlap_codes else sorted(committee_test["code"].unique())[:4]
    if selected_codes:
        fig, axes = plt.subplots(len(selected_codes), 1, figsize=(12, 3.2 * len(selected_codes)), sharex=True)
        if len(selected_codes) == 1:
            axes = [axes]
        for ax, code in zip(axes, selected_codes):
            standalone_code = standalone_test[standalone_test["code"] == code].copy()
            committee_code = committee_test[committee_test["code"] == code].copy()
            code_df = (
                standalone_code[["Date", "actual", "prediction"]]
                .rename(columns={"prediction": "standalone_prediction"})
                .merge(
                    committee_code[["Date", "prediction"]].rename(columns={"prediction": "committee_prediction"}),
                    on="Date",
                    how="inner",
                )
                .sort_values("Date")
            )
            code_df["actual_roll20"] = code_df["actual"].rolling(20, min_periods=5).mean()
            code_df["standalone_roll20"] = code_df["standalone_prediction"].rolling(20, min_periods=5).mean()
            code_df["committee_roll20"] = code_df["committee_prediction"].rolling(20, min_periods=5).mean()
            ax.plot(code_df["Date"], code_df["actual_roll20"], label="Actual 20d mean", linewidth=1.5, color="#222222")
            ax.plot(code_df["Date"], code_df["standalone_roll20"], label="Standalone 20d mean", linewidth=1.2, color="#1f77b4")
            ax.plot(code_df["Date"], code_df["committee_roll20"], label="Committee 20d mean", linewidth=1.2, color="#d62728")
            ax.set_title(f"{code}: 20-Day Rolling Mean")
            ax.grid(alpha=0.2)
        axes[0].legend(ncol=3, loc="upper right")
        fig.tight_layout()
        fig.savefig(plots_dir / "test_per_stock_rolling20.png", dpi=160)
        plt.close(fig)

    standalone_trade = build_trade_frame(standalone_test)
    committee_trade = build_trade_frame(committee_test)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(standalone_trade["Date"], standalone_trade["equity"], label="Standalone threshold=0", linewidth=1.5, color="#1f77b4")
    ax.plot(committee_trade["Date"], committee_trade["equity"], label="Committee threshold=0", linewidth=1.5, color="#d62728")
    ax.set_title(f"{label}: Backtest Equity Curve")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_backtest_equity_curve.png", dpi=160)
    plt.close(fig)

    rel_scores = [
        standalone_metrics[standalone_model]["test"]["rel_score"],
        committee_summary["best_committee"]["committee_test_rel_score"],
        committee_summary["best_committee_stability"]["stable_test_rel_score_median"],
    ]
    labels = ["Standalone", "Committee point", "Committee stable median"]
    colors = ["#1f77b4", "#d62728", "#ff7f0e"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(labels, rel_scores, color=colors)
    axes[0].axhline(0.1, color="#555555", linestyle="--", linewidth=1.0, label="Target 0.1")
    axes[0].set_title("Test rel_score Comparison")
    axes[0].set_ylabel("rel_score")
    axes[0].legend()

    bias_labels = ["pred_pos_rate", "actual_pos_rate", "pred_abs/actual_abs"]
    standalone_bias_vals = [
        standalone_bias["pred_pos_rate"],
        standalone_bias["actual_pos_rate"],
        standalone_bias["pred_abs_over_actual_abs"],
    ]
    committee_bias_vals = [
        committee_bias["pred_pos_rate"],
        committee_bias["actual_pos_rate"],
        committee_bias["pred_abs_over_actual_abs"],
    ]
    x = range(len(bias_labels))
    width = 0.36
    axes[1].bar([i - width / 2 for i in x], standalone_bias_vals, width=width, label="Standalone", color="#1f77b4")
    axes[1].bar([i + width / 2 for i in x], committee_bias_vals, width=width, label="Committee", color="#d62728")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(bias_labels, rotation=10)
    axes[1].set_title("Bias Diagnostics on Test")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_relscore_and_bias_summary.png", dpi=160)
    plt.close(fig)

    sign_labels = ["pred_pos_rate", "actual_pos_rate", "sign_agreement"]
    standalone_sign_vals = [
        standalone_bias["pred_pos_rate"],
        standalone_bias["actual_pos_rate"],
        compute_sign_agreement(standalone_test),
    ]
    committee_sign_vals = [
        committee_bias["pred_pos_rate"],
        committee_bias["actual_pos_rate"],
        compute_sign_agreement(committee_test),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    x = range(len(sign_labels))
    width = 0.36
    ax.bar([i - width / 2 for i in x], standalone_sign_vals, width=width, label="Standalone", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], committee_sign_vals, width=width, label="Committee", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sign_labels, rotation=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Test Sign-Rate and Sign-Agreement")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_sign_rate_and_agreement.png", dpi=160)
    plt.close(fig)

    standalone_bucket = build_signal_bucket_frame(standalone_test, "Standalone")
    committee_bucket = build_signal_bucket_frame(committee_test, "Committee")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
    for ax, frame, title, color in (
        (axes[0], standalone_bucket, "Standalone", "#1f77b4"),
        (axes[1], committee_bucket, "Committee", "#d62728"),
    ):
        ax.bar(frame["bucket"].astype(str), frame["actual_mean"], color=color, alpha=0.75, label="Mean actual return")
        ax.plot(frame["bucket"].astype(str), frame["prediction_mean"], color="#222222", marker="o", linewidth=1.2, label="Mean prediction")
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_title(f"{title}: Signal Buckets")
        ax.set_xlabel("Prediction decile")
        ax.set_ylabel("Return")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "test_signal_bucket_actual_mean.png", dpi=160)
    plt.close(fig)

    standalone_curve = build_positive_precision_curve(standalone_test, "Standalone")
    committee_curve = build_positive_precision_curve(committee_test, "Committee")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, frame, metric, title in (
        (axes[0], pd.concat([standalone_curve, committee_curve], ignore_index=True), "avg_actual_return", "Positive-Signal Precision"),
        (axes[1], pd.concat([standalone_curve, committee_curve], ignore_index=True), "hit_rate", "Positive-Signal Hit Rate"),
    ):
        for model_label, color in (("Standalone", "#1f77b4"), ("Committee", "#d62728")):
            part = frame[frame["model_label"] == model_label]
            ax.plot(part["quantile"], part[metric], marker="o", linewidth=1.4, label=model_label, color=color)
        ax.set_title(title)
        ax.set_xlabel("Prediction quantile within positive signals")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Mean actual return")
    axes[1].set_ylabel("Hit rate")
    fig.tight_layout()
    fig.savefig(plots_dir / "test_positive_signal_precision_curve.png", dpi=160)
    plt.close(fig)

    standalone_mask = standalone_test["prediction"] >= 0.0
    committee_mask = committee_test["prediction"] >= 0.0
    mask_summary = pd.DataFrame(
        {
            "category": ["Standalone only", "Committee only", "Both active", "Both inactive"],
            "count": [
                int((standalone_mask & ~committee_mask).sum()),
                int((committee_mask & ~standalone_mask).sum()),
                int((standalone_mask & committee_mask).sum()),
                int((~standalone_mask & ~committee_mask).sum()),
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(mask_summary["category"], mask_summary["count"], color=["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"])
    ax.set_title("Trade-Mask Comparison at Threshold 0")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_trade_mask_comparison.png", dpi=160)
    plt.close(fig)


def write_report(
    target_dir: Path,
    standalone_metrics: dict,
    standalone_backtest: dict,
    standalone_bias: dict,
    committee_summary: dict,
    committee_backtest: dict,
    committee_bias: dict,
    label: str,
    standalone_model: str,
    committee_model: str,
) -> None:
    standalone_test = standalone_metrics[standalone_model]["test"]
    standalone_val = standalone_metrics[standalone_model]["val"]
    committee_best = committee_summary["best_committee"]
    committee_stability = committee_summary["best_committee_stability"]
    report = f"""# {label}

## Candidate shortlist
- Standalone expert: `{Path(committee_summary["expert_run"]).name}` with `{standalone_model}`
- Committee candidate: `{Path(committee_summary["market_run"]).name}` + expert
- Committee rule: `method={committee_best["method"]}`, `weight_expert={committee_best["weight_expert"]}`, overlap `{committee_best["overlap_codes"]}`

## Evaluation snapshot
- Standalone `val rel_score`: `{standalone_val["rel_score"]:.6f}`
- Standalone `test rel_score`: `{standalone_test["rel_score"]:.6f}`
- Committee `val rel_score`: `{committee_best["committee_val_rel_score"]:.6f}`
- Committee `test rel_score`: `{committee_best["committee_test_rel_score"]:.6f}`
- Committee stable-band test median: `{committee_stability["stable_test_rel_score_median"]:.6f}`
- Committee stable-band test mean: `{committee_stability["stable_test_rel_score_mean"]:.6f}`

## Backtest snapshot
- Standalone best threshold: `{standalone_backtest[standalone_model]["threshold"]}`
- Standalone final equity: `{standalone_backtest[standalone_model]["final_equity"]:.6f}`
- Standalone avg strategy return: `{standalone_backtest[standalone_model]["avg_strategy_return"]:.6f}`
- Committee best threshold: `{committee_backtest[committee_model]["threshold"]}`
- Committee final equity: `{committee_backtest[committee_model]["final_equity"]:.6f}`
- Committee avg strategy return: `{committee_backtest[committee_model]["avg_strategy_return"]:.6f}`

## Bias diagnostics
- Standalone `pred_pos_rate`: `{standalone_bias["pred_pos_rate"]:.4f}` vs `actual_pos_rate`: `{standalone_bias["actual_pos_rate"]:.4f}`
- Standalone `pred_abs_over_actual_abs`: `{standalone_bias["pred_abs_over_actual_abs"]:.4f}`
- Committee `pred_pos_rate`: `{committee_bias["pred_pos_rate"]:.4f}` vs `actual_pos_rate`: `{committee_bias["actual_pos_rate"]:.4f}`
- Committee `pred_abs_over_actual_abs`: `{committee_bias["pred_abs_over_actual_abs"]:.4f}`
"""
    (target_dir / "advisor_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    ensure_clean_dir(output_dir)
    committee_summary = load_json(args.committee_dir / "best_committee_summary.json")

    standalone_core = args.standalone_run / "reports/core"
    standalone_metrics = load_json(standalone_core / "metrics.json")

    export_dir = output_dir / "artifacts"
    export_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("metrics.json", "family_selection_summary.json", "config.json"):
        copy_small_artifact(standalone_core / filename, export_dir / f"standalone_{filename}")
    for filename in ("best_committee_summary.json", "committee_grid_results.csv", "committee_stability_summary.csv"):
        copy_small_artifact(args.committee_dir / filename, export_dir / filename)

    committee_run = build_committee_packaged_run(output_dir / "committee_packaged_run", args.committee_dir, committee_summary)
    standalone_backtest = run_backtest(args.standalone_run, args.standalone_model, "candidate_report")
    committee_backtest = run_backtest(committee_run, "committee_best_by_val", "candidate_report")

    copy_small_artifact(
        args.standalone_run / "reports/backtests/threshold_backtest_candidate_report.csv",
        export_dir / "standalone_threshold_backtest_candidate_report.csv",
    )
    copy_small_artifact(
        args.standalone_run / "reports/backtests/threshold_backtest_summary_candidate_report.json",
        export_dir / "standalone_threshold_backtest_summary_candidate_report.json",
    )
    copy_small_artifact(
        committee_run / "reports/backtests/threshold_backtest_candidate_report.csv",
        export_dir / "committee_threshold_backtest_candidate_report.csv",
    )
    copy_small_artifact(
        committee_run / "reports/backtests/threshold_backtest_summary_candidate_report.json",
        export_dir / "committee_threshold_backtest_summary_candidate_report.json",
    )

    standalone_bias = compute_bias_stats(standalone_core / "predictions.csv", args.standalone_model)
    committee_bias = compute_bias_stats(committee_run / "reports/core/predictions.csv", "committee_best_by_val")
    save_plots(
        output_dir,
        standalone_core / "predictions.csv",
        committee_run / "reports/core/predictions.csv",
        args.standalone_model,
        "committee_best_by_val",
        standalone_metrics,
        committee_summary,
        standalone_bias,
        committee_bias,
        args.label,
    )

    manifest = {
        "label": args.label,
        "output_dir": str(output_dir),
        "standalone_run": str(args.standalone_run),
        "committee_source_dir": str(args.committee_dir),
        "committee_packaged_run": str(committee_run),
        "standalone_model": args.standalone_model,
        "committee_model": "committee_best_by_val",
        "standalone_backtest_summary": standalone_backtest,
        "committee_backtest_summary": committee_backtest,
        "standalone_bias": standalone_bias,
        "committee_bias": committee_bias,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_report(
        output_dir,
        standalone_metrics,
        standalone_backtest,
        standalone_bias,
        committee_summary,
        committee_backtest,
        committee_bias,
        args.label,
        args.standalone_model,
        "committee_best_by_val",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
