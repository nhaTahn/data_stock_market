from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_RUNS = ROOT / "data/processed/assets/data_info_vn/history/training_runs"
ADVISOR_ROOT = TRAINING_RUNS / "reports/advisor_shortlist/fnb_committee_best_20260409"
STANDALONE_RUN = TRAINING_RUNS / "overnight_fnb_w5_mag_base_20260409_101741"
COMMITTEE_DIR = (
    TRAINING_RUNS
    / "reports/committee_experiments/confirm_vn100_fnb_committee_20260408_235445_r01__committee__fnb"
)


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


def build_committee_packaged_run(target_dir: Path, committee_summary: dict) -> tuple[Path, dict]:
    target_dir.mkdir(parents=True, exist_ok=True)
    core_dir = target_dir / "reports/core"
    core_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "reports/backtests").mkdir(parents=True, exist_ok=True)

    preds_path = COMMITTEE_DIR / "best_committee_predictions.csv"
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
    stability = committee_summary["best_committee_stability"]
    config = {
        "target_mode": "return",
        "target_column": "target_next_return",
        "advisor_packaged": True,
        "source_type": "committee",
        "source_committee_summary": str(COMMITTEE_DIR / "best_committee_summary.json"),
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
        "stable_weight_min": stability["stable_weight_min"],
        "stable_weight_max": stability["stable_weight_max"],
        "stable_weight_count": stability["stable_weight_count"],
        "stable_test_rel_score_median": stability["stable_test_rel_score_median"],
        "stable_test_rel_score_mean": stability["stable_test_rel_score_mean"],
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
    return target_dir, config


def run_backtest(run_dir: Path, model: str) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "src/backtesting/threshold_backtest.py"),
        str(run_dir),
        "--models",
        model,
        "--output-suffix",
        "advisor_report",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    summary_path = run_dir / "reports/backtests/threshold_backtest_summary_advisor_report.json"
    return load_json(summary_path)


def compute_bias_stats(predictions_path: Path, model: str) -> dict[str, float]:
    df = pd.read_csv(predictions_path)
    test = df[(df["split"] == "test") & (df["model"] == model)].copy()
    if test.empty:
        return {}
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
    if active.empty:
        return active
    daily = active.groupby("Date", as_index=False)["actual"].mean().rename(columns={"actual": "strategy_return"})
    daily["equity"] = (1.0 + daily["strategy_return"]).cumprod()
    return daily


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
) -> None:
    plots_dir = target_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    standalone_test = load_test_predictions(standalone_predictions_path, standalone_model)
    committee_test = load_test_predictions(committee_predictions_path, committee_model)

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
    ax.set_title("F&B Test: Actual vs Prediction Mean by Date")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_actual_vs_predictions_mean.png", dpi=160)
    plt.close(fig)

    compare_roll = compare.copy()
    compare_roll["actual_roll20"] = compare_roll["actual"].rolling(20, min_periods=5).mean()
    compare_roll["standalone_roll20"] = compare_roll["standalone_prediction"].rolling(20, min_periods=5).mean()
    compare_roll["committee_roll20"] = compare_roll["committee_prediction"].rolling(20, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(compare_roll["Date"], compare_roll["actual_roll20"], label="Actual 20d mean", linewidth=1.6, color="#222222")
    ax.plot(compare_roll["Date"], compare_roll["standalone_roll20"], label="Standalone 20d mean", linewidth=1.3, color="#1f77b4")
    ax.plot(compare_roll["Date"], compare_roll["committee_roll20"], label="Committee 20d mean", linewidth=1.3, color="#d62728")
    ax.set_title("F&B Test: 20-Day Rolling Mean Shows Underreaction")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_rolling20_actual_vs_predictions.png", dpi=160)
    plt.close(fig)

    standalone_trade = build_trade_frame(standalone_test)
    committee_trade = build_trade_frame(committee_test)
    fig, ax = plt.subplots(figsize=(12, 5))
    if not standalone_trade.empty:
        ax.plot(standalone_trade["Date"], standalone_trade["equity"], label="Standalone threshold=0", linewidth=1.5, color="#1f77b4")
    if not committee_trade.empty:
        ax.plot(committee_trade["Date"], committee_trade["equity"], label="Committee threshold=0", linewidth=1.5, color="#d62728")
    ax.set_title("F&B Test Backtest Equity Curve")
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


def write_report(
    target_dir: Path,
    standalone_metrics: dict,
    standalone_backtest: dict,
    standalone_bias: dict,
    committee_summary: dict,
    committee_backtest: dict,
    committee_bias: dict,
) -> None:
    standalone_model = "lstm_best_by_val"
    committee_model = "committee_best_by_val"
    standalone_test = standalone_metrics[standalone_model]["test"]
    standalone_val = standalone_metrics[standalone_model]["val"]
    committee_best = committee_summary["best_committee"]
    committee_stability = committee_summary["best_committee_stability"]
    report = f"""# F&B Best Candidate Report

## Candidate shortlist
- Standalone expert: `{STANDALONE_RUN.name}` with `{standalone_model}` on `KDC,SAB,SBT,VNM`
- Committee candidate: stable-band committee from `VN100 context + F&B expert`
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
- Standalone directional accuracy: `{standalone_backtest[standalone_model]["directional_accuracy"]:.6f}`
- Committee best threshold: `{committee_backtest[committee_model]["threshold"]}`
- Committee final equity: `{committee_backtest[committee_model]["final_equity"]:.6f}`
- Committee avg strategy return: `{committee_backtest[committee_model]["avg_strategy_return"]:.6f}`
- Committee directional accuracy: `{committee_backtest[committee_model]["directional_accuracy"]:.6f}`

## Bias diagnostics
- Standalone `pred_pos_rate`: `{standalone_bias["pred_pos_rate"]:.4f}` vs `actual_pos_rate`: `{standalone_bias["actual_pos_rate"]:.4f}`
- Standalone `pred_abs_over_actual_abs`: `{standalone_bias["pred_abs_over_actual_abs"]:.4f}`
- Committee `pred_pos_rate`: `{committee_bias["pred_pos_rate"]:.4f}` vs `actual_pos_rate`: `{committee_bias["actual_pos_rate"]:.4f}`
- Committee `pred_abs_over_actual_abs`: `{committee_bias["pred_abs_over_actual_abs"]:.4f}`

## Reading notes
- Current best direction is still `F&B mini-group + committee`, not whole-market LSTM.
- Committee improves `rel_score` over standalone on the same 4-code overlap, but still remains far below the target `0.1`.
- The main failure mode is underreaction: prediction amplitude is still much smaller than actual return amplitude.
- This package is suitable for advisor review because it isolates the strongest path and removes the noise from failed whole-market and residual experiments.
"""
    (target_dir / "advisor_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_clean_dir(ADVISOR_ROOT)

    standalone_core = STANDALONE_RUN / "reports/core"
    standalone_metrics_path = standalone_core / "metrics.json"
    standalone_family_path = standalone_core / "family_selection_summary.json"
    standalone_config_path = standalone_core / "config.json"
    committee_summary_path = COMMITTEE_DIR / "best_committee_summary.json"
    committee_stability_path = COMMITTEE_DIR / "committee_stability_summary.csv"
    committee_grid_path = COMMITTEE_DIR / "committee_grid_results.csv"

    standalone_metrics = load_json(standalone_metrics_path)
    committee_summary = load_json(committee_summary_path)

    export_dir = ADVISOR_ROOT / "artifacts"
    export_dir.mkdir(parents=True, exist_ok=True)
    copy_small_artifact(standalone_metrics_path, export_dir / "standalone_metrics.json")
    copy_small_artifact(standalone_family_path, export_dir / "standalone_family_selection_summary.json")
    copy_small_artifact(standalone_config_path, export_dir / "standalone_config.json")
    copy_small_artifact(committee_summary_path, export_dir / "committee_best_summary.json")
    copy_small_artifact(committee_stability_path, export_dir / "committee_stability_summary.csv")
    copy_small_artifact(committee_grid_path, export_dir / "committee_grid_results.csv")

    committee_run, _ = build_committee_packaged_run(ADVISOR_ROOT / "committee_packaged_run", committee_summary)
    standalone_backtest = run_backtest(STANDALONE_RUN, "lstm_best_by_val")
    committee_backtest = run_backtest(committee_run, "committee_best_by_val")

    copy_small_artifact(
        STANDALONE_RUN / "reports/backtests/threshold_backtest_advisor_report.csv",
        export_dir / "standalone_threshold_backtest_advisor_report.csv",
    )
    copy_small_artifact(
        STANDALONE_RUN / "reports/backtests/threshold_backtest_summary_advisor_report.json",
        export_dir / "standalone_threshold_backtest_summary_advisor_report.json",
    )
    copy_small_artifact(
        committee_run / "reports/backtests/threshold_backtest_advisor_report.csv",
        export_dir / "committee_threshold_backtest_advisor_report.csv",
    )
    copy_small_artifact(
        committee_run / "reports/backtests/threshold_backtest_summary_advisor_report.json",
        export_dir / "committee_threshold_backtest_summary_advisor_report.json",
    )

    standalone_bias = compute_bias_stats(standalone_core / "predictions.csv", "lstm_best_by_val")
    committee_bias = compute_bias_stats(committee_run / "reports/core/predictions.csv", "committee_best_by_val")
    save_plots(
        ADVISOR_ROOT,
        standalone_core / "predictions.csv",
        committee_run / "reports/core/predictions.csv",
        "lstm_best_by_val",
        "committee_best_by_val",
        standalone_metrics,
        committee_summary,
        standalone_bias,
        committee_bias,
    )

    manifest = {
        "advisor_dir": str(ADVISOR_ROOT),
        "standalone_run": str(STANDALONE_RUN),
        "committee_source_dir": str(COMMITTEE_DIR),
        "committee_packaged_run": str(committee_run),
        "standalone_model": "lstm_best_by_val",
        "committee_model": "committee_best_by_val",
        "standalone_backtest_summary": standalone_backtest,
        "committee_backtest_summary": committee_backtest,
        "standalone_bias": standalone_bias,
        "committee_bias": committee_bias,
    }
    with (ADVISOR_ROOT / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    write_report(
        ADVISOR_ROOT,
        standalone_metrics,
        standalone_backtest,
        standalone_bias,
        committee_summary,
        committee_backtest,
        committee_bias,
    )
    print(ADVISOR_ROOT)


if __name__ == "__main__":
    main()
