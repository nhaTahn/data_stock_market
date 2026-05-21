from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs" / "reports"
DEFAULT_SOURCE = DEFAULT_REPORT_ROOT / "selective_error_control_target3p0_20260520"
DEFAULT_OUTPUT = DEFAULT_REPORT_ROOT / "standard_error_control_20260520"
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "standard_error_control_20260520"


@dataclass(frozen=True)
class PolicySpec:
    score: str
    policy: str

    @property
    def label(self) -> str:
        if self.score == "none":
            return "full_coverage"
        return f"{self.score}_{self.policy}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build standard multi-seed report for LSTM + post-hoc selective error-control.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--policies",
        default=(
            "risk_logistic:coverage_q40,"
            "risk_hgb:coverage_q40,"
            "risk_input_noise:coverage_q30,"
            "risk_input_noise:calibrated_p90_3p5"
        ),
    )
    parser.add_argument("--min-daily-n", type=int, default=20)
    parser.add_argument("--target-error", type=float, default=0.035)
    return parser.parse_args(argv)


def parse_policies(value: str) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    for item in value.split(","):
        if not item.strip():
            continue
        score, policy = item.split(":", maxsplit=1)
        specs.append(PolicySpec(score.strip(), policy.strip()))
    return specs


def pct(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def daily_q90(frame: pd.DataFrame, min_daily_n: int) -> pd.Series:
    counts = frame.groupby("Date", sort=True)["abs_error"].count()
    daily = frame.groupby("Date", sort=True)["abs_error"].quantile(0.90)
    return daily[counts >= min_daily_n]


def threshold_for_policy(by_seed: pd.DataFrame, seed: int, spec: PolicySpec) -> float:
    row = by_seed[
        by_seed["seed"].eq(seed)
        & by_seed["score"].eq(spec.score)
        & by_seed["policy"].eq(spec.policy)
    ]
    if row.empty:
        raise ValueError(f"Missing threshold row for seed={seed}, score={spec.score}, policy={spec.policy}")
    return float(row.iloc[0]["threshold"])


def select_policy(frame: pd.DataFrame, spec: PolicySpec, threshold: float) -> pd.Series:
    if spec.score == "none" and spec.policy == "full_coverage":
        return pd.Series(True, index=frame.index)
    if spec.score not in frame.columns:
        raise ValueError(f"Missing score column in validation frame: {spec.score}")
    return frame[spec.score].le(threshold)


def summarize_year(
    frame: pd.DataFrame,
    selected: pd.Series,
    *,
    seed: int,
    spec: PolicySpec,
    year: int,
    min_daily_n: int,
) -> dict[str, object]:
    year_frame = frame[frame["Date"].dt.year.eq(year)]
    selected_frame = year_frame.loc[selected.reindex(year_frame.index).fillna(False)]
    daily = daily_q90(selected_frame, min_daily_n)
    return {
        "seed": seed,
        "score": spec.score,
        "policy": spec.policy,
        "label": spec.label,
        "year": year,
        "n_obs": int(len(selected_frame)),
        "obs_coverage": float(len(selected_frame) / max(len(year_frame), 1)),
        "n_days": int(daily.shape[0]),
        "day_coverage": float(daily.shape[0] / max(year_frame["Date"].nunique(), 1)),
        "daily_q90_median": float(daily.median()) if not daily.empty else float("nan"),
        "daily_q90_p90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "daily_q90_max": float(daily.max()) if not daily.empty else float("nan"),
        "days_ge_3p5": int(daily.ge(0.035).sum()) if not daily.empty else 0,
        "days_ge_5": int(daily.ge(0.05).sum()) if not daily.empty else 0,
        "days_ge_8": int(daily.ge(0.08).sum()) if not daily.empty else 0,
    }


def daily_rows(
    frame: pd.DataFrame,
    selected: pd.Series,
    *,
    seed: int,
    spec: PolicySpec,
    min_daily_n: int,
) -> pd.DataFrame:
    selected_frame = frame.loc[selected].copy()
    daily = daily_q90(selected_frame, min_daily_n)
    coverage = selected_frame.groupby("Date", sort=True)["abs_error"].count() / frame.groupby("Date", sort=True)["abs_error"].count()
    out = pd.DataFrame(
        {
            "Date": daily.index,
            "seed": seed,
            "score": spec.score,
            "policy": spec.policy,
            "label": spec.label,
            "daily_q90": daily.to_numpy(dtype=float),
            "obs_coverage": coverage.reindex(daily.index).to_numpy(dtype=float),
        }
    )
    out["year"] = pd.to_datetime(out["Date"]).dt.year
    return out


def aggregate_yearly(yearly: pd.DataFrame) -> pd.DataFrame:
    return (
        yearly.groupby(["score", "policy", "label", "year"], sort=True)
        .agg(
            seeds=("seed", "nunique"),
            obs_coverage_mean=("obs_coverage", "mean"),
            day_coverage_mean=("day_coverage", "mean"),
            daily_q90_median_mean=("daily_q90_median", "mean"),
            daily_q90_p90_mean=("daily_q90_p90", "mean"),
            daily_q90_max_mean=("daily_q90_max", "mean"),
            days_ge_3p5_mean=("days_ge_3p5", "mean"),
            days_ge_5_mean=("days_ge_5", "mean"),
            days_ge_8_mean=("days_ge_8", "mean"),
        )
        .reset_index()
    )


def write_yearly_plot(gold_dir: Path, daily: pd.DataFrame, full_daily: pd.DataFrame, spec: PolicySpec, target_error: float) -> None:
    selected_daily = daily[daily["label"].eq(spec.label)].copy()
    years = sorted(selected_daily["year"].dropna().unique())
    if not years:
        return
    fig, axes = plt.subplots(len(years), 1, figsize=(11, 3.2 * len(years)), sharey=True)
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        full = (
            full_daily[full_daily["year"].eq(year)]
            .groupby("Date", sort=True)["daily_q90"]
            .mean()
            .reset_index()
        )
        selected = (
            selected_daily[selected_daily["year"].eq(year)]
            .groupby("Date", sort=True)["daily_q90"]
            .mean()
            .reset_index()
        )
        ax.plot(full["Date"], 100.0 * full["daily_q90"], color="#475569", linewidth=1.3, label="full coverage")
        ax.plot(selected["Date"], 100.0 * selected["daily_q90"], color="#2563eb", linewidth=1.5, label="accepted")
        ax.axhline(100.0 * target_error, color="#dc2626", linestyle="--", linewidth=1.0, label="3.5% target")
        ax.set_title(f"{year}")
        ax.set_ylabel("daily q90(|E|), %")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Full vs accepted error: {spec.score}/{spec.policy}", y=0.995)
    fig.tight_layout()
    fig.savefig(gold_dir / f"yearly_error_{spec.label}.png", dpi=180)
    plt.close(fig)


def write_frontier_plot(gold_dir: Path, aggregate: pd.DataFrame, specs: list[PolicySpec], target_error: float) -> None:
    keep = pd.DataFrame(
        [{"score": spec.score, "policy": spec.policy, "label": spec.label} for spec in [PolicySpec("none", "full_coverage"), *specs]]
    )
    plot = aggregate.merge(keep, on=["score", "policy"], how="inner", suffixes=("", "_spec"))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(100.0 * plot["obs_coverage_mean"], 100.0 * plot["daily_q90_p90_mean"], s=76, alpha=0.88)
    for _, row in plot.iterrows():
        ax.annotate(str(row["label"]), (100.0 * row["obs_coverage_mean"], 100.0 * row["daily_q90_p90_mean"]), fontsize=8)
    ax.axhline(100.0 * target_error, color="#dc2626", linestyle="--", linewidth=1.0, label="3.5% target")
    ax.set_xlabel("Observation coverage (%)")
    ax.set_ylabel("Validation daily q90(|E|) p90 (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(gold_dir / "standard_error_control_frontier.png", dpi=180)
    plt.close(fig)


def format_policy_table(aggregate: pd.DataFrame, specs: list[PolicySpec]) -> str:
    keep = [(spec.score, spec.policy) for spec in [PolicySpec("none", "full_coverage"), *specs]]
    display = aggregate[
        aggregate.apply(lambda row: (row["score"], row["policy"]) in keep, axis=1)
    ].copy()
    order = {pair: idx for idx, pair in enumerate(keep)}
    display["order"] = display.apply(lambda row: order[(row["score"], row["policy"])], axis=1)
    display = display.sort_values("order")
    rows = [
        "| score | policy | obs coverage | day coverage | rel_score | daily median | daily p90 | daily max | days >=3.5% | days >=8% |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.iterrows():
        rows.append(
            f"| `{row.score}` | `{row.policy}` | {pct(row.obs_coverage_mean)} | {pct(row.day_coverage_mean)} | "
            f"{float(row.rel_score_mean):.5f} | {pct(row.daily_q90_median_mean)} | {pct(row.daily_q90_p90_mean)} | "
            f"{pct(row.daily_q90_max_mean)} | {float(row.days_ge_3p5_mean):.1f} | {float(row.days_ge_8_mean):.1f} |"
        )
    return "\n".join(rows)


def format_yearly_table(yearly_agg: pd.DataFrame, specs: list[PolicySpec]) -> str:
    keep_labels = {spec.label for spec in [PolicySpec("none", "full_coverage"), *specs]}
    display = yearly_agg[yearly_agg["label"].isin(keep_labels)].copy()
    display = display.sort_values(["year", "label"])
    rows = [
        "| year | label | obs coverage | daily median | daily p90 | daily max | days >=3.5% | days >=8% |",
        "|--:|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.iterrows():
        rows.append(
            f"| {int(row.year)} | `{row.label}` | {pct(row.obs_coverage_mean)} | "
            f"{pct(row.daily_q90_median_mean)} | {pct(row.daily_q90_p90_mean)} | {pct(row.daily_q90_max_mean)} | "
            f"{float(row.days_ge_3p5_mean):.1f} | {float(row.days_ge_8_mean):.1f} |"
        )
    return "\n".join(rows)


def write_summary(
    gold_dir: Path,
    aggregate: pd.DataFrame,
    yearly_agg: pd.DataFrame,
    specs: list[PolicySpec],
    seeds: list[int],
    target_error: float,
) -> None:
    lines = [
        "# Standard LSTM Error-Control Report",
        "",
        "Base forecast: `stressaux_w20` LSTM. Risk scores are fitted on early train, calibrated on late train, and reported on validation only.",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
        "",
        "## Multi-Seed Policy Table",
        "",
        format_policy_table(aggregate, specs),
        "",
        "## Yearly Stability",
        "",
        format_yearly_table(yearly_agg, specs),
        "",
        "## Read",
        "",
        f"- Full coverage keeps the best return-forecast score, but daily p90 stays above `{target_error:.1%}`.",
        "- Selective policies remove abnormal `>=8%` spike days in validation, but only on accepted samples.",
        "- `risk_input_noise` is the cleanest evidence that input quality/noise controls error spikes.",
        "- This supports a two-layer framing: LSTM for expected return, calibrated error-control for confidence.",
        "",
        "## Plot Files",
        "",
        "- `standard_error_control_frontier.png`",
    ]
    for spec in specs:
        lines.append(f"- `yearly_error_{spec.label}.png`")
    (gold_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    specs = parse_policies(args.policies)

    aggregate = pd.read_csv(args.source_dir / "selective_error_aggregate.csv")
    by_seed = pd.read_csv(args.source_dir / "selective_error_by_seed.csv")
    seeds = sorted(int(seed) for seed in by_seed["seed"].dropna().unique())

    yearly_rows: list[dict[str, object]] = []
    daily_parts: list[pd.DataFrame] = []
    full_spec = PolicySpec("none", "full_coverage")
    for seed in seeds:
        frame = pd.read_csv(args.source_dir / f"val_selective_scores_seed_{seed}.csv", parse_dates=["Date"])
        years = sorted(frame["Date"].dt.year.dropna().unique())
        full_selected = pd.Series(True, index=frame.index)
        for year in years:
            yearly_rows.append(
                summarize_year(
                    frame,
                    full_selected,
                    seed=seed,
                    spec=full_spec,
                    year=int(year),
                    min_daily_n=args.min_daily_n,
                )
            )
        daily_parts.append(daily_rows(frame, full_selected, seed=seed, spec=full_spec, min_daily_n=args.min_daily_n))
        for spec in specs:
            threshold = threshold_for_policy(by_seed, seed, spec)
            selected = select_policy(frame, spec, threshold)
            for year in years:
                yearly_rows.append(
                    summarize_year(
                        frame,
                        selected,
                        seed=seed,
                        spec=spec,
                        year=int(year),
                        min_daily_n=args.min_daily_n,
                    )
                )
            daily_parts.append(daily_rows(frame, selected, seed=seed, spec=spec, min_daily_n=args.min_daily_n))

    yearly = pd.DataFrame(yearly_rows)
    yearly_agg = aggregate_yearly(yearly)
    daily = pd.concat(daily_parts, ignore_index=True)
    full_daily = daily[daily["label"].eq(full_spec.label)].copy()

    yearly.to_csv(args.output_dir / "standard_error_control_yearly_by_seed.csv", index=False)
    yearly_agg.to_csv(args.output_dir / "standard_error_control_yearly_aggregate.csv", index=False)
    daily.to_csv(args.output_dir / "standard_error_control_daily_series.csv", index=False)
    aggregate.to_csv(args.output_dir / "selective_error_aggregate_source.csv", index=False)
    manifest = {
        "source_dir": str(args.source_dir),
        "seeds": seeds,
        "policies": [{"score": spec.score, "policy": spec.policy, "label": spec.label} for spec in specs],
        "base_prediction": "stressaux_w20",
        "target_error": args.target_error,
        "min_daily_n": args.min_daily_n,
        "holdout_test_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for file_name in [
        "standard_error_control_yearly_by_seed.csv",
        "standard_error_control_yearly_aggregate.csv",
        "standard_error_control_daily_series.csv",
        "selective_error_aggregate_source.csv",
        "manifest.json",
    ]:
        (args.gold_dir / file_name).write_bytes((args.output_dir / file_name).read_bytes())
    write_frontier_plot(args.gold_dir, aggregate, specs, args.target_error)
    for spec in specs:
        write_yearly_plot(args.gold_dir, daily, full_daily, spec, args.target_error)
    write_summary(args.gold_dir, aggregate, yearly_agg, specs, seeds, args.target_error)
    print((args.gold_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
