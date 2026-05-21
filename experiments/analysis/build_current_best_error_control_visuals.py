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
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "selective_error_control_target3p0_20260520"
)
DEFAULT_OUTPUT = (
    ROOT
    / "gold"
    / "vn_transition_pressure_20260512"
    / "plots"
    / "current_best_error_control_visuals_20260520"
)


@dataclass(frozen=True)
class PolicySpec:
    score: str
    policy: str
    title: str

    @property
    def label(self) -> str:
        return f"{self.score}_{self.policy}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current-best LSTM error-control visuals.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score", default="risk_hgb")
    parser.add_argument("--policy", default="coverage_q40")
    parser.add_argument("--title", default="LSTM stressaux_w20 + HGB error-control q40")
    parser.add_argument("--min-daily-n", type=int, default=20)
    parser.add_argument("--target-error", type=float, default=0.035)
    return parser.parse_args(argv)


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def pct(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * float(value):.2f}%"


def threshold_for_seed(by_seed: pd.DataFrame, seed: int, spec: PolicySpec) -> float:
    row = by_seed[
        by_seed["seed"].eq(seed)
        & by_seed["score"].eq(spec.score)
        & by_seed["policy"].eq(spec.policy)
    ]
    if row.empty:
        raise ValueError(f"Missing threshold for seed={seed}, score={spec.score}, policy={spec.policy}")
    return float(row.iloc[0]["threshold"])


def load_policy_frames(source_dir: Path, spec: PolicySpec) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    by_seed = pd.read_csv(source_dir / "selective_error_by_seed.csv")
    seeds = sorted(int(seed) for seed in by_seed["seed"].dropna().unique())
    full_parts: list[pd.DataFrame] = []
    accepted_parts: list[pd.DataFrame] = []
    for seed in seeds:
        frame = pd.read_csv(source_dir / f"val_selective_scores_seed_{seed}.csv", parse_dates=["Date"])
        frame["seed"] = seed
        frame["error"] = frame["prediction"].astype(float) - frame["actual"].astype(float)
        frame["abs_error"] = frame["error"].abs()
        full_parts.append(frame)
        threshold = threshold_for_seed(by_seed, seed, spec)
        accepted = frame[frame[spec.score].astype(float).le(threshold)].copy()
        accepted["threshold"] = threshold
        accepted_parts.append(accepted)
    return pd.concat(full_parts, ignore_index=True), pd.concat(accepted_parts, ignore_index=True), seeds


def summarize_frame(frame: pd.DataFrame, *, label: str) -> dict[str, object]:
    actual = frame["actual"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    error = prediction - actual
    return {
        "label": label,
        "rows": int(len(frame)),
        "stocks": int(frame["code"].astype(str).nunique()),
        "days": int(frame["Date"].nunique()),
        "obs_coverage": float("nan"),
        "rel_score": rel_score(actual, prediction),
        "base_score": robust_loss(actual),
        "abs_score": robust_loss(actual - prediction),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(prediction))) if len(frame) else float("nan"),
        "error_min": float(np.min(error)) if len(frame) else float("nan"),
        "error_q20": float(np.quantile(error, 0.20)) if len(frame) else float("nan"),
        "error_q25": float(np.quantile(error, 0.25)) if len(frame) else float("nan"),
        "error_median": float(np.quantile(error, 0.50)) if len(frame) else float("nan"),
        "error_q75": float(np.quantile(error, 0.75)) if len(frame) else float("nan"),
        "error_q80": float(np.quantile(error, 0.80)) if len(frame) else float("nan"),
        "error_max": float(np.max(error)) if len(frame) else float("nan"),
        "error_mean": float(np.mean(error)) if len(frame) else float("nan"),
        "error_std": float(np.std(error)) if len(frame) else float("nan"),
        "abs_error_q90": float(np.quantile(np.abs(error), 0.90)) if len(frame) else float("nan"),
    }


def save_error_histogram(frame: pd.DataFrame, summary: dict[str, object], title: str, output_path: Path) -> None:
    errors = frame["error"].to_numpy(dtype=float)
    x_low = float(np.quantile(errors, 0.005))
    x_high = float(np.quantile(errors, 0.995))
    visible = errors[(errors >= x_low) & (errors <= x_high)]
    clipped_share = 1.0 - len(visible) / max(len(errors), 1)

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    bins = np.linspace(x_low, x_high, 52)
    ax.hist(visible, bins=bins, color="#1f77b4", alpha=0.86, edgecolor="white", linewidth=0.7)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="E = 0")
    ax.axvline(float(summary["error_mean"]), color="#d62728", linewidth=1.5, label=f"mean={summary['error_mean']:.5f}")
    ax.axvline(float(summary["error_q20"]), color="#17becf", linestyle=":", linewidth=1.35, label=f"q20={summary['error_q20']:.5f}")
    ax.axvline(float(summary["error_q25"]), color="#2ca02c", linestyle=":", linewidth=1.35, label=f"q25={summary['error_q25']:.5f}")
    ax.axvline(float(summary["error_q75"]), color="#9467bd", linestyle=":", linewidth=1.35, label=f"q75={summary['error_q75']:.5f}")
    ax.axvline(float(summary["error_q80"]), color="#bcbd22", linestyle=":", linewidth=1.35, label=f"q80={summary['error_q80']:.5f}")
    ax.set_title(f"Histogram of E = prediction - actual (validation, seed-pooled)\n{title}")
    ax.set_xlabel("E = prediction - actual")
    ax.set_ylabel("Frequency")
    ax.set_xlim(x_low, x_high)
    ax.grid(axis="y", alpha=0.18)
    stats_text = (
        f"rel_score={summary['rel_score']:.5f}\n"
        f"base_score={summary['base_score']:.5f}\n"
        f"abs_score={summary['abs_score']:.5f}\n"
        f"dir_acc={summary['directional_accuracy']:.2%}\n"
        f"stocks={summary['stocks']} rows={summary['rows']}\n"
        f"xlim=q0.5..q99.5 clipped={clipped_share:.1%}\n"
        f"min={summary['error_min']:.5f}  max={summary['error_max']:.5f}\n"
        f"q20={summary['error_q20']:.5f}  q25={summary['error_q25']:.5f}\n"
        f"q75={summary['error_q75']:.5f}  q80={summary['error_q80']:.5f}\n"
        f"mean={summary['error_mean']:.5f}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        fontsize=10.5,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_rel_score_proxy_histogram(frame: pd.DataFrame, summary: dict[str, object], title: str, output_path: Path) -> None:
    actual_abs = np.abs(frame["actual"].to_numpy(dtype=float))
    error_abs = frame["abs_error"].to_numpy(dtype=float)
    denom = np.maximum(actual_abs, max(float(summary["base_score"]), 1e-4))
    proxy = np.clip(1.0 - error_abs / denom, -1.5, 1.0)
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    ax.hist(proxy, bins=np.linspace(-1.5, 1.0, 50), color="#2563eb", alpha=0.84, edgecolor="white", linewidth=0.6)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="proxy = 0")
    ax.axvline(float(summary["rel_score"]), color="#16a34a", linestyle="--", linewidth=1.4, label=f"aggregate rel_score={summary['rel_score']:.5f}")
    ax.axvline(float(np.mean(proxy)), color="#dc2626", linewidth=1.4, label=f"mean proxy={np.mean(proxy):.5f}")
    ax.set_title(f"Stabilized local rel_score proxy histogram (validation, seed-pooled)\n{title}")
    ax.set_xlabel("1 - |E| / max(|actual|, base_score)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.18)
    stats_text = (
        f"aggregate rel_score={summary['rel_score']:.5f}\n"
        f"mean proxy={np.mean(proxy):.5f}\n"
        f"median proxy={np.median(proxy):.5f}\n"
        f"share(proxy>0)={np.mean(proxy > 0.0):.2%}\n"
        f"share(proxy<-0.5)={np.mean(proxy < -0.5):.2%}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        fontsize=10.5,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def daily_q90(frame: pd.DataFrame, min_daily_n: int) -> pd.Series:
    counts = frame.groupby(["seed", "Date"], sort=True)["abs_error"].count()
    daily = frame.groupby(["seed", "Date"], sort=True)["abs_error"].quantile(0.90)
    return daily[counts >= min_daily_n]


def daily_series(full: pd.DataFrame, accepted: pd.DataFrame, min_daily_n: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label, frame in [("full coverage", full), ("accepted", accepted)]:
        daily = daily_q90(frame, min_daily_n).rename("daily_q90").reset_index()
        daily["label"] = label
        rows.append(daily)
    out = pd.concat(rows, ignore_index=True)
    out["year"] = pd.to_datetime(out["Date"]).dt.year
    return out


def save_yearly_abs_error_plot(daily: pd.DataFrame, title: str, target_error: float, output_path: Path) -> None:
    years = sorted(daily["year"].dropna().unique())
    fig, axes = plt.subplots(len(years), 1, figsize=(11.5, 3.2 * len(years)), sharey=True)
    if len(years) == 1:
        axes = [axes]
    colors = {"full coverage": "#475569", "accepted": "#2563eb"}
    for ax, year in zip(axes, years):
        year_df = daily[daily["year"].eq(year)]
        for label in ("full coverage", "accepted"):
            line = year_df[year_df["label"].eq(label)].groupby("Date", sort=True)["daily_q90"].mean().reset_index()
            ax.plot(line["Date"], 100.0 * line["daily_q90"], label=label, color=colors[label], linewidth=1.5)
        ax.axhline(100.0 * target_error, color="#dc2626", linestyle="--", linewidth=1.0, label="3.5% target")
        ax.set_title(str(year))
        ax.set_ylabel("daily q90(|E|), %")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    full_summary: dict[str, object],
    accepted_summary: dict[str, object],
    daily: pd.DataFrame,
    spec: PolicySpec,
    seeds: list[int],
) -> None:
    daily_summary = (
        daily.groupby(["label", "year"], sort=True)
        .agg(
            daily_q90_median=("daily_q90", "median"),
            daily_q90_p90=("daily_q90", lambda values: float(np.quantile(values, 0.90))),
            daily_q90_max=("daily_q90", "max"),
            days_ge_3p5=("daily_q90", lambda values: int(np.sum(np.asarray(values) >= 0.035))),
            days_ge_8=("daily_q90", lambda values: int(np.sum(np.asarray(values) >= 0.08))),
        )
        .reset_index()
    )
    daily_summary.to_csv(output_dir / "yearly_abs_error_summary.csv", index=False)
    rows = [
        "# Current Best Error-Control Visuals",
        "",
        f"Policy: `{spec.score}/{spec.policy}`.",
        f"Seeds pooled for visualization: `{', '.join(str(seed) for seed in seeds)}`.",
        "",
        "## Histogram Summary",
        "",
        "| sample | rows | rel_score | base_score | abs_score | q90(|E|) | dir_acc | q20(E) | q80(E) | mean(E) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for sample, summary in [("full coverage", full_summary), ("accepted", accepted_summary)]:
        rows.append(
            f"| `{sample}` | {int(summary['rows'])} | {float(summary['rel_score']):+.5f} | "
            f"{float(summary['base_score']):.5f} | {float(summary['abs_score']):.5f} | "
            f"{pct(summary['abs_error_q90'])} | {float(summary['directional_accuracy']):.2%} | "
            f"{float(summary['error_q20']):+.5f} | {float(summary['error_q80']):+.5f} | "
            f"{float(summary['error_mean']):+.5f} |"
        )
    rows += [
        "",
        "## Yearly q90(|E|)",
        "",
        "| label | year | median | p90 | max | days >=3.5% | days >=8% |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in daily_summary.iterrows():
        rows.append(
            f"| `{row.label}` | {int(row.year)} | {pct(row.daily_q90_median)} | {pct(row.daily_q90_p90)} | "
            f"{pct(row.daily_q90_max)} | {int(row.days_ge_3p5)} | {int(row.days_ge_8)} |"
        )
    rows += [
        "",
        "## Plot Files",
        "",
        "- `error_hist_full_coverage_seed_pooled.png`",
        "- `error_hist_accepted_seed_pooled.png`",
        "- `rel_score_proxy_hist_accepted_seed_pooled.png`",
        "- `yearly_q90_abs_error_best_policy.png`",
    ]
    (output_dir / "summary.md").write_text("\n".join(rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = PolicySpec(args.score, args.policy, args.title)
    full, accepted, seeds = load_policy_frames(args.source_dir, spec)
    full_summary = summarize_frame(full, label="full_coverage")
    accepted_summary = summarize_frame(accepted, label=spec.label)
    save_error_histogram(full, full_summary, "stressaux_w20 full coverage", args.output_dir / "error_hist_full_coverage_seed_pooled.png")
    save_error_histogram(accepted, accepted_summary, spec.title, args.output_dir / "error_hist_accepted_seed_pooled.png")
    save_rel_score_proxy_histogram(accepted, accepted_summary, spec.title, args.output_dir / "rel_score_proxy_hist_accepted_seed_pooled.png")
    daily = daily_series(full, accepted, args.min_daily_n)
    daily.to_csv(args.output_dir / "daily_q90_abs_error_series.csv", index=False)
    save_yearly_abs_error_plot(
        daily,
        f"Validation daily q90(|E|) by year\n{spec.title}",
        args.target_error,
        args.output_dir / "yearly_q90_abs_error_best_policy.png",
    )
    pd.DataFrame([full_summary, accepted_summary]).to_csv(args.output_dir / "error_hist_summary.csv", index=False)
    write_summary(args.output_dir, full_summary, accepted_summary, daily, spec, seeds)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(args.source_dir),
                "score": args.score,
                "policy": args.policy,
                "title": args.title,
                "seeds": seeds,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
