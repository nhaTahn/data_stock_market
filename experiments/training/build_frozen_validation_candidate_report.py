"""Build frozen validation candidate advisor report.

Combines the fixed long-train ensemble prediction diagnostics with the
validation-only portfolio/risk overlay diagnostics. Holdout/test is not used.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ENSEMBLE_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/fixed_train_ensemble_calibration_20260524"
FOLD_SERIES_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/fixed_train_fold_relscore_series_20260524"
SINGLE_SEED_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/fixed_train_relscore_calibration_20260524"
PORTFOLIO_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_market_gate_overlays_20260524"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/frozen_validation_candidate_20260524"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/frozen_validation_candidate_20260524"

PREDICTION_VARIANT = "ensemble_mean_cal_each_traincal_clip"
PORTFOLIO_POLICY = "daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5"
PORTFOLIO_GATE = "wyck040"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def copy_outputs() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.glob("*"):
        if path.is_file():
            path.unlink()


def load_prediction_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.read_csv(ENSEMBLE_DIR / "overall_metrics.csv")
    folds = pd.read_csv(ENSEMBLE_DIR / "fold_metrics.csv")
    daily = pd.read_csv(ENSEMBLE_DIR / "daily_metrics.csv")
    fold_series_summary = pd.read_csv(FOLD_SERIES_DIR / "summary.csv")
    single_seed_summary = pd.read_csv(SINGLE_SEED_DIR / "summary_metrics.csv")
    return overall, folds, daily, fold_series_summary, single_seed_summary


def build_year_month_tables(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = daily[daily["variant"] == PREDICTION_VARIANT].copy()
    selected["Date"] = pd.to_datetime(selected["Date"])
    selected["year"] = selected["Date"].dt.year
    selected["month"] = selected["Date"].dt.to_period("M").astype(str)
    year = selected.groupby("year").agg(
        days=("rel_score", "size"),
        mean_rel_score=("rel_score", "mean"),
        median_rel_score=("rel_score", "median"),
        positive_days=("rel_score", lambda series: int((series > 0).sum())),
        mean_absE_robust=("absE_robust", "mean"),
        p90_absE_robust=("absE_robust", lambda series: float(series.quantile(0.9))),
        mean_absE_q90=("absE_q90", "mean"),
        mean_DA=("DA", "mean"),
    ).reset_index()
    month = selected.groupby("month").agg(
        days=("rel_score", "size"),
        mean_rel_score=("rel_score", "mean"),
        median_rel_score=("rel_score", "median"),
        positive_days=("rel_score", lambda series: int((series > 0).sum())),
        mean_absE_robust=("absE_robust", "mean"),
        p90_absE_robust=("absE_robust", lambda series: float(series.quantile(0.9))),
        mean_absE_q90=("absE_q90", "mean"),
        mean_DA=("DA", "mean"),
    ).reset_index()
    return year, month


def build_worst_tables(folds: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_folds = folds[folds["variant"] == PREDICTION_VARIANT].copy()
    selected_daily = daily[daily["variant"] == PREDICTION_VARIANT].copy()
    worst_folds = selected_folds.sort_values("rel_score").head(12)
    worst_days = selected_daily.sort_values("rel_score").head(20)
    return worst_folds, worst_days


def build_portfolio_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overlay = pd.read_csv(PORTFOLIO_DIR / "gate_overlay_summary.csv")
    seed = pd.read_csv(PORTFOLIO_DIR / "seed_gate_metrics.csv")
    fold = pd.read_csv(PORTFOLIO_DIR / "fold_gate_metrics.csv")
    selected_overlay = overlay[(overlay["policy"] == PORTFOLIO_POLICY) & (overlay["gate"] == PORTFOLIO_GATE)]
    selected_seed = seed[(seed["policy"] == PORTFOLIO_POLICY) & (seed["gate"] == PORTFOLIO_GATE)]
    selected_fold = fold[(fold["policy"] == PORTFOLIO_POLICY) & (fold["gate"] == PORTFOLIO_GATE)]
    return selected_overlay, selected_seed, selected_fold


def write_tables(
    overall: pd.DataFrame,
    fold_summary: pd.DataFrame,
    year: pd.DataFrame,
    month: pd.DataFrame,
    worst_folds: pd.DataFrame,
    worst_days: pd.DataFrame,
    portfolio: pd.DataFrame,
    portfolio_seed: pd.DataFrame,
    portfolio_fold: pd.DataFrame,
) -> None:
    tables = {
        "prediction_overall_metrics.csv": overall,
        "prediction_fixed_fold_summary.csv": fold_summary,
        "prediction_year_metrics.csv": year,
        "prediction_month_metrics.csv": month,
        "prediction_worst_folds.csv": worst_folds,
        "prediction_worst_days.csv": worst_days,
        "portfolio_overlay_summary.csv": portfolio,
        "portfolio_seed_metrics.csv": portfolio_seed,
        "portfolio_fold_metrics.csv": portfolio_fold,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT / name, index=False)
        table.to_csv(GOLD / name, index=False)


def render_report(
    overall: pd.DataFrame,
    fold_summary: pd.DataFrame,
    fixed_fold_summary: pd.DataFrame,
    single_seed_summary: pd.DataFrame,
    year: pd.DataFrame,
    month: pd.DataFrame,
    worst_folds: pd.DataFrame,
    worst_days: pd.DataFrame,
    portfolio: pd.DataFrame,
    portfolio_seed: pd.DataFrame,
    portfolio_fold: pd.DataFrame,
) -> str:
    selected = overall[overall["variant"] == PREDICTION_VARIANT].iloc[0]
    single = single_seed_summary[
        (single_seed_summary["variant"] == "train_cal") & (single_seed_summary["gate"] == "none")
    ].iloc[0]
    portfolio_row = portfolio.iloc[0]
    fold_row = fold_summary[fold_summary["variant"] == PREDICTION_VARIANT].iloc[0]
    lines = [
        "# Frozen Validation Candidate Advisor Report",
        "",
        "Protocol: train `<= 2020-03-31`, validation/in-sample `2020-04-01..2022-11-15`. Holdout/test not used.",
        "",
        "## Recommendation",
        "",
        f"- Prediction candidate: `{PREDICTION_VARIANT}` from cached `hetero_combined_full5_20260521`.",
        f"- Portfolio/risk overlay: `{PORTFOLIO_POLICY}` with gate `{PORTFOLIO_GATE}`.",
        "- Keep holdout closed until these choices are frozen and no further validation tuning is planned.",
        "",
        "## Prediction Target Metrics",
        "",
        "| Metric | Frozen ensemble | Prior fixed train_cal reference |",
        "| --- | ---: | ---: |",
        f"| rel_score | **{selected['rel_score']:.5f}** | {single['mean_rel_score']:.5f} |",
        f"| absE_robust | **{pct(selected['absE_robust'])}** | {pct(single['mean_absE_robust'])} |",
        f"| absE_q90 | **{pct(selected['absE_q90'])}** | {pct(single['mean_absE_q90'])} |",
        f"| DA | **{pct(selected['DA'])}** | {pct(single['mean_DA'])} |",
        f"| pred/actual q90 ratio | **{selected['pred_actual_q90_ratio']:.3f}** | {single['mean_ratio']:.3f} |",
        "",
        "## 21-Day Fold Stability",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| mean fold rel_score | {fold_row['mean_fold_rel']:.5f} |",
        f"| median fold rel_score | {fold_row['median_fold_rel']:.5f} |",
        f"| minimum fold rel_score | {fold_row['min_fold_rel']:.5f} |",
        f"| positive folds | {int(fold_row['positive_folds'])}/{int(fold_row['folds'])} |",
        f"| mean absE_robust | {pct(fold_row['mean_absE_robust'])} |",
        f"| p90 absE_robust | {pct(fold_row['p90_absE_robust'])} |",
        "",
        "## Yearly Prediction Series",
        "",
        year.round(5).to_markdown(index=False),
        "",
        "## Worst Prediction Folds",
        "",
        worst_folds[
            ["test_start", "test_end", "n", "rel_score", "absE_robust", "absE_q90", "DA", "pred_actual_q90_ratio"]
        ].round(5).to_markdown(index=False),
        "",
        "## Portfolio Overlay",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| mean equity | {portfolio_row['mean_equity']:.4f} |",
        f"| min seed equity | {portfolio_row['min_equity']:.4f} |",
        f"| mean annual return | {pct(portfolio_row['mean_ann_ret'])} |",
        f"| mean Sharpe | {portfolio_row['mean_sharpe']:.4f} |",
        f"| worst max drawdown | {pct(portfolio_row['worst_max_dd'])} |",
        f"| worst fold equity | {portfolio_row['worst_fold_equity']:.4f} |",
        f"| positive folds | {int(portfolio_row['positive_folds'])}/{int(portfolio_row['n_folds'])} |",
        f"| gate active days | {pct(portfolio_row['gate_active_global'])} |",
        "",
        "## Portfolio Seed Metrics",
        "",
        portfolio_seed.round(5).to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- Full-universe rel_score should be judged on the ungated prediction candidate.",
        "- Wyckoff/pressure gates are execution overlays; they reduce crash exposure but should not be used to score full-universe prediction quality.",
        "- Short-window rolling retrain (`w126`, and even `w504`) remains a robustness stress-test, not the main objective protocol.",
        "- The next step before holdout is to freeze this candidate and generate the final run manifest/config from these exact artifacts.",
    ]
    return "\n".join(lines)


def mirror_report_assets() -> None:
    for source in [ENSEMBLE_DIR / "summary.md", PORTFOLIO_DIR / "summary.md"]:
        if source.exists():
            shutil.copy2(source, OUTPUT / f"source_{source.parent.name}_{source.name}")
            shutil.copy2(source, GOLD / f"source_{source.parent.name}_{source.name}")


def main() -> None:
    copy_outputs()
    overall, folds, daily, fixed_fold_summary, single_seed_summary = load_prediction_tables()
    fold_summary = pd.read_csv(ENSEMBLE_DIR / "fold_summary.csv")
    year, month = build_year_month_tables(daily)
    worst_folds, worst_days = build_worst_tables(folds, daily)
    portfolio, portfolio_seed, portfolio_fold = build_portfolio_tables()
    write_tables(overall, fixed_fold_summary, year, month, worst_folds, worst_days, portfolio, portfolio_seed, portfolio_fold)
    report = render_report(
        overall,
        fold_summary,
        fixed_fold_summary,
        single_seed_summary,
        year,
        month,
        worst_folds,
        worst_days,
        portfolio,
        portfolio_seed,
        portfolio_fold,
    )
    (OUTPUT / "advisor_report.md").write_text(report, encoding="utf-8")
    (GOLD / "advisor_report.md").write_text(report, encoding="utf-8")
    mirror_report_assets()
    print(report)


if __name__ == "__main__":
    main()
