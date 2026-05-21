from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.evaluate_selective_error_control import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_REPORT_ROOT,
    RISK_FEATURES,
    add_model_scores,
    add_risk_features,
    daily_q90_for_selection,
    fit_risk_models,
    load_predictions,
    load_raw_features,
    parse_seeds,
    split_train_calibration,
    summarize_selection,
)


DEFAULT_OUTPUT = DEFAULT_REPORT_ROOT / "day_guard_error_control_20260520"
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "day_guard_error_control_20260520"
DEFAULT_MARKET_DATA = DEFAULT_DATA
DEFAULT_VN100_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn100_symbols.csv"


@dataclass(frozen=True)
class SamplePolicy:
    score: str
    quantile: float

    @property
    def name(self) -> str:
        return f"{self.score}_q{int(round(100.0 * self.quantile))}"


@dataclass(frozen=True)
class DayGuardPolicy:
    sample_policy: SamplePolicy
    guard_column: str
    guard_quantile: float

    @property
    def name(self) -> str:
        return f"{self.sample_policy.name}_day_{self.guard_column}_q{int(round(100.0 * self.guard_quantile))}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated day-level guards on top of LSTM selective error-control.",
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--market-data", type=Path, default=DEFAULT_MARKET_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--target-error", type=float, default=0.035)
    parser.add_argument("--target-max-error", type=float, default=0.050)
    parser.add_argument("--min-daily-n", type=int, default=20)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    return parser.parse_args(argv)


def read_symbol_file(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    column = "symbol" if "symbol" in frame.columns else "code"
    return set(frame[column].dropna().astype(str).str.upper())


def build_index_proxy(data_path: Path, symbols: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=["Date", "code", "adjust"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = raw[raw["code"].isin(symbols)].copy()
    raw = raw.sort_values(["code", "Date"], kind="stable")
    raw["stock_return"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    daily = raw.groupby("Date", sort=True)["stock_return"].mean().rename("index_proxy_return").reset_index()
    daily["index_proxy"] = (1.0 + daily["index_proxy_return"].fillna(0.0)).cumprod()
    return daily


def rebase_to_100(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    first = clean.dropna().iloc[0]
    return clean / first * 100.0


def candidate_day_guards(frame: pd.DataFrame, selected: pd.Series, score_column: str) -> pd.DataFrame:
    selected_frame = frame.loc[selected].copy()
    if selected_frame.empty:
        return pd.DataFrame(columns=["Date", "selected_count"])
    selected_score = (
        selected_frame.groupby("Date", sort=True)[score_column]
        .quantile(0.90)
        .rename("selected_score_q90")
        .reset_index()
    )
    selected_count = selected_frame.groupby("Date", sort=True)["abs_error"].count().rename("selected_count").reset_index()
    daily_context = (
        frame.groupby("Date", sort=True)
        .agg(
            daily_input_noise_q90=("daily_input_noise_q90", "last"),
            daily_disagreement_q90=("daily_disagreement_q90", "last"),
            daily_volatility_q90=("daily_volatility_q90", "last"),
            daily_bb_width_q90=("daily_bb_width_q90", "last"),
        )
        .reset_index()
    )
    return selected_count.merge(selected_score, on="Date", how="left").merge(daily_context, on="Date", how="left")


def choose_day_guard(
    calibration: pd.DataFrame,
    selected: pd.Series,
    *,
    sample_policy: SamplePolicy,
    guard_column: str,
    target_error: float,
    target_max_error: float,
    min_daily_n: int,
) -> tuple[float, str, pd.DataFrame]:
    day_values = candidate_day_guards(calibration, selected, sample_policy.score)
    day_values = day_values.dropna(subset=[guard_column]).copy()
    if day_values.empty:
        return float("nan"), "missing_day_guard", day_values

    best_threshold = float(day_values[guard_column].quantile(0.10))
    best_quantile = 0.10
    best_coverage = -np.inf
    for quantile in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
        threshold = float(day_values[guard_column].quantile(quantile))
        allowed_dates = set(day_values.loc[day_values[guard_column].le(threshold), "Date"])
        guarded_selected = selected & calibration["Date"].isin(allowed_dates)
        daily = daily_q90_for_selection(calibration.loc[guarded_selected], min_daily_n)
        if daily.empty:
            continue
        daily_p90 = float(daily.quantile(0.90))
        daily_max = float(daily.max())
        coverage = float(guarded_selected.mean())
        if daily_p90 <= target_error and daily_max <= target_max_error and coverage > best_coverage:
            best_threshold = threshold
            best_quantile = quantile
            best_coverage = coverage
    return best_threshold, f"{guard_column}_q{int(round(100.0 * best_quantile))}", day_values


def apply_day_guard(
    frame: pd.DataFrame,
    selected: pd.Series,
    *,
    score_column: str,
    guard_column: str,
    threshold: float,
) -> pd.Series:
    if not np.isfinite(threshold):
        return pd.Series(False, index=frame.index)
    day_values = candidate_day_guards(frame, selected, score_column)
    allowed_dates = set(day_values.loc[day_values[guard_column].le(threshold), "Date"])
    return selected & frame["Date"].isin(allowed_dates)


def evaluate_seed(seed: int, args: argparse.Namespace, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_predictions(args.report_root, seed)
    frame = add_risk_features(frame, raw)
    train = frame[frame["split"].eq("train")].copy()
    val = frame[frame["split"].eq("val")].copy()
    fit, calibration = split_train_calibration(train, args.calibration_fraction)
    models = fit_risk_models(fit, args.target_error)
    calibration = add_model_scores(calibration, models)
    val = add_model_scores(val, models)

    sample_policies = [
        SamplePolicy("risk_hgb", 0.40),
        SamplePolicy("risk_logistic", 0.25),
        SamplePolicy("risk_input_noise", 0.30),
    ]
    guard_columns = [
        "selected_score_q90",
        "daily_input_noise_q90",
        "daily_disagreement_q90",
        "daily_volatility_q90",
        "daily_bb_width_q90",
    ]
    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    accepted_parts: list[pd.DataFrame] = []

    for sample_policy in sample_policies:
        sample_threshold = float(np.quantile(calibration[sample_policy.score].dropna().to_numpy(dtype=float), sample_policy.quantile))
        cal_selected = calibration[sample_policy.score].le(sample_threshold)
        val_selected = val[sample_policy.score].le(sample_threshold)
        rows.append(
            summarize_selection(
                val,
                val_selected,
                seed=seed,
                score_name=sample_policy.score,
                policy=sample_policy.name,
                threshold=sample_threshold,
                min_daily_n=args.min_daily_n,
            )
        )
        for guard_column in guard_columns:
            guard_threshold, guard_policy, _ = choose_day_guard(
                calibration,
                cal_selected,
                sample_policy=sample_policy,
                guard_column=guard_column,
                target_error=args.target_error,
                target_max_error=args.target_max_error,
                min_daily_n=args.min_daily_n,
            )
            final_selected = apply_day_guard(
                val,
                val_selected,
                score_column=sample_policy.score,
                guard_column=guard_column,
                threshold=guard_threshold,
            )
            policy_name = f"{sample_policy.name}__day_guard_{guard_column}_calibrated"
            row = summarize_selection(
                val,
                final_selected,
                seed=seed,
                score_name=sample_policy.score,
                policy=policy_name,
                threshold=sample_threshold,
                min_daily_n=args.min_daily_n,
            )
            row["day_guard_column"] = guard_column
            row["day_guard_threshold"] = guard_threshold
            rows.append(row)
            threshold_rows.append(
                {
                    "seed": seed,
                    "sample_score": sample_policy.score,
                    "sample_policy": sample_policy.name,
                    "sample_threshold": sample_threshold,
                    "day_guard_column": guard_column,
                    "day_guard_threshold": guard_threshold,
                    "day_guard_policy": guard_policy,
                }
            )
            accepted = val.loc[final_selected, ["code", "Date", "actual", "prediction", "abs_error", sample_policy.score]].copy()
            accepted["seed"] = seed
            accepted["policy"] = policy_name
            accepted["score"] = sample_policy.score
            accepted_parts.append(accepted)

    accepted_frame = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else pd.DataFrame()
    return pd.DataFrame(rows), pd.DataFrame(threshold_rows), accepted_frame


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["score", "policy"], sort=True)
        .agg(
            seeds=("seed", "nunique"),
            obs_coverage_mean=("obs_coverage", "mean"),
            day_coverage_mean=("day_coverage", "mean"),
            rel_score_mean=("rel_score", "mean"),
            q90_abs_error_mean=("q90_abs_error", "mean"),
            daily_q90_median_mean=("daily_q90_median", "mean"),
            daily_q90_p90_mean=("daily_q90_p90", "mean"),
            daily_q90_max_mean=("daily_q90_max", "mean"),
            days_ge_3p5_mean=("days_ge_3p5", "mean"),
            days_ge_5_mean=("days_ge_5", "mean"),
            days_ge_7_mean=("days_ge_7", "mean"),
            days_ge_8_mean=("days_ge_8", "mean"),
        )
        .reset_index()
    )


def write_summary(gold_dir: Path, aggregate: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    display = aggregate.sort_values(["daily_q90_max_mean", "daily_q90_p90_mean", "obs_coverage_mean"], ascending=[True, True, False])
    lines = [
        "# Day-Guard Error-Control Readout",
        "",
        "Base forecast: `stressaux_w20`. Sample gate and day guard are calibrated on train/calibration, then evaluated on validation.",
        "",
        "| score | policy | obs coverage | day coverage | rel_score | daily median | daily p90 | daily max | days >=3.5% | days >=5% | days >=8% |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.head(24).iterrows():
        lines.append(
            f"| `{row.score}` | `{row.policy}` | {100*float(row.obs_coverage_mean):.2f}% | {100*float(row.day_coverage_mean):.2f}% | "
            f"{float(row.rel_score_mean):.5f} | {100*float(row.daily_q90_median_mean):.2f}% | "
            f"{100*float(row.daily_q90_p90_mean):.2f}% | {100*float(row.daily_q90_max_mean):.2f}% | "
            f"{float(row.days_ge_3p5_mean):.1f} | {float(row.days_ge_5_mean):.1f} | {float(row.days_ge_8_mean):.1f} |"
        )
    lines += [
        "",
        "## Thresholds",
        "",
        thresholds.to_markdown(index=False),
    ]
    (gold_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_frontier(gold_dir: Path, aggregate: pd.DataFrame) -> None:
    plot = aggregate.sort_values(["daily_q90_max_mean", "daily_q90_p90_mean"]).head(18).copy()
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.scatter(100.0 * plot["obs_coverage_mean"], 100.0 * plot["daily_q90_max_mean"], s=72, alpha=0.85)
    for _, row in plot.iterrows():
        label = str(row["policy"]).replace("__day_guard_", "\n")
        ax.annotate(label, (100.0 * row["obs_coverage_mean"], 100.0 * row["daily_q90_max_mean"]), fontsize=7)
    ax.axhline(5.0, color="#dc2626", linestyle="--", linewidth=1.0, label="5% max target")
    ax.set_xlabel("Observation coverage (%)")
    ax.set_ylabel("Validation daily q90(|E|) max (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(gold_dir / "day_guard_max_error_frontier.png", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    raw = load_raw_features(args.data)

    result_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    accepted_parts: list[pd.DataFrame] = []
    for seed in seeds:
        result, thresholds, accepted = evaluate_seed(seed, args, raw)
        result_parts.append(result)
        threshold_parts.append(thresholds)
        accepted_parts.append(accepted)

    results = pd.concat(result_parts, ignore_index=True)
    thresholds = pd.concat(threshold_parts, ignore_index=True)
    accepted = pd.concat(accepted_parts, ignore_index=True)
    aggregate = aggregate_results(results)
    results.to_csv(args.output_dir / "day_guard_by_seed.csv", index=False)
    thresholds.to_csv(args.output_dir / "day_guard_thresholds.csv", index=False)
    aggregate.to_csv(args.output_dir / "day_guard_aggregate.csv", index=False)
    accepted.to_csv(args.output_dir / "day_guard_accepted_rows.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "target_error": args.target_error,
                "target_max_error": args.target_max_error,
                "min_daily_n": args.min_daily_n,
                "risk_features": list(RISK_FEATURES),
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for file_name in [
        "day_guard_by_seed.csv",
        "day_guard_thresholds.csv",
        "day_guard_aggregate.csv",
        "day_guard_accepted_rows.csv",
        "manifest.json",
    ]:
        (args.gold_dir / file_name).write_bytes((args.output_dir / file_name).read_bytes())
    write_summary(args.gold_dir, aggregate, thresholds)
    plot_frontier(args.gold_dir, aggregate)
    print((args.gold_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
