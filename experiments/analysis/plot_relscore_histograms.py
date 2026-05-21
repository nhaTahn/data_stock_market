"""Plot rel_score and error distribution diagnostics for the current best
candidate (stressaux_w20).

Outputs:
  gold/vn_transition_pressure_20260512/plots/relscore_histograms_20260521/
    error_histogram_full_coverage.png
    error_histogram_per_seed.png
    relscore_summary.csv
    summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROBE_DIR = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history"
    / "training_runs" / "reports" / "stressaux_lstm_probe_20260519"
)
DEFAULT_OUTPUT = (
    ROOT / "gold" / "vn_transition_pressure_20260512" / "plots"
    / "relscore_histograms_20260521"
)
DEFAULT_VARIANT = "plain_global_weighted_mild_tail35_stressaux_w20"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--seeds", default="43,52,71")
    return parser.parse_args(argv)


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score_fn(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if base <= 0 or not np.isfinite(base):
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def load_predictions(probe_dir: Path, variant: str, seeds: list[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for seed in seeds:
        path = probe_dir / f"seed_{seed}" / f"predictions_{variant}.csv"
        if not path.exists():
            print(f"Warning: missing {path}")
            continue
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df["seed"] = seed
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No predictions found for variant={variant} in {probe_dir}")
    return pd.concat(parts, ignore_index=True)


def compute_per_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, split), group in df.groupby(["seed", "split"]):
        actual = group["actual"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        error = actual - pred
        abs_error = np.abs(error)
        daily = group.copy()
        daily["abs_error"] = abs_error
        daily_q90 = daily.groupby("Date")["abs_error"].quantile(0.90)
        rows.append({
            "seed": int(seed),
            "split": split,
            "n_obs": int(len(group)),
            "rel_score": rel_score_fn(actual, pred),
            "median_abs_error": float(np.quantile(abs_error, 0.5)),
            "q90_abs_error": float(np.quantile(abs_error, 0.9)),
            "q95_abs_error": float(np.quantile(abs_error, 0.95)),
            "daily_q90_p90": float(daily_q90.quantile(0.90)),
            "daily_q90_max": float(daily_q90.max()),
            "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))),
            "spike_days_ge_5pct": int(daily_q90.ge(0.05).sum()),
            "spike_days_ge_7pct": int(daily_q90.ge(0.07).sum()),
            "spike_days_ge_8pct": int(daily_q90.ge(0.08).sum()),
            "actual_q90": float(np.quantile(np.abs(actual), 0.9)),
            "pred_q90": float(np.quantile(np.abs(pred), 0.9)),
            "pred_actual_q90_ratio": float(np.quantile(np.abs(pred), 0.9) / max(np.quantile(np.abs(actual), 0.9), 1e-8)),
        })
    return pd.DataFrame(rows)


def plot_error_histogram_pooled(df: pd.DataFrame, output_path: Path, variant: str) -> None:
    """Plot histogram of signed error pooled across seeds, train vs val."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, split, color in zip(axes, ["train", "val"], ["#4c72b0", "#c44e52"]):
        subset = df[df["split"] == split]
        if subset.empty:
            ax.text(0.5, 0.5, f"No {split} data", ha="center", va="center", transform=ax.transAxes)
            continue
        actual = subset["actual"].to_numpy(dtype=float)
        pred = subset["prediction"].to_numpy(dtype=float)
        error = (actual - pred) * 100  # signed error in %
        abs_error = np.abs(error)
        rs = rel_score_fn(actual, pred)
        mean_e = float(np.mean(error))
        std_e = float(np.std(error))
        q20 = float(np.quantile(abs_error, 0.20))
        q25 = float(np.quantile(abs_error, 0.25))
        median_abs = float(np.quantile(abs_error, 0.5))
        q75 = float(np.quantile(abs_error, 0.75))
        q80 = float(np.quantile(abs_error, 0.80))
        q90_abs = float(np.quantile(abs_error, 0.9))
        da = float(np.mean(np.sign(actual) == np.sign(pred)))
        # Clip for display
        clip_range = min(max(q90_abs * 1.5, 8.0), 15.0)
        clipped = np.clip(error, -clip_range, clip_range)
        ax.hist(clipped, bins=100, color=color, alpha=0.75, edgecolor="black", linewidth=0.2)
        # Vertical lines — quantiles of |E| shown symmetrically
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(mean_e, color="orange", linestyle="--", linewidth=1.2, label=f"mean = {mean_e:+.3f}%")
        for q_val, q_label, q_color, q_alpha in [
            (q25, "Q25", "#2ca02c", 0.5),
            (q75, "Q75", "#9467bd", 0.6),
            (q80, "Q80", "#8c564b", 0.6),
            (q90_abs, "Q90", "#d62728", 0.8),
        ]:
            ax.axvline(-q_val, color=q_color, linestyle=":", linewidth=0.9, alpha=q_alpha)
            ax.axvline(+q_val, color=q_color, linestyle=":", linewidth=0.9, alpha=q_alpha,
                       label=f"±{q_label} = ±{q_val:.2f}%")
        ax.axvline(-3.5, color="darkred", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axvline(+3.5, color="darkred", linestyle="--", linewidth=1.0, alpha=0.6, label="±3.5% violation")
        ax.set_xlim(-clip_range, clip_range)
        ax.set_xlabel(r"$E = r_{\mathrm{actual}} - \hat{r}_{\mathrm{prediction}}$ [%]", fontsize=10)
        ax.set_ylabel("count")
        title_line1 = f"{split.capitalize()}  |  rel_score = {rs:+.4f}  |  DA = {da:.1%}"
        title_line2 = (
            f"Q20={q20:.2f}%  Q25={q25:.2f}%  median={median_abs:.2f}%  "
            f"Q75={q75:.2f}%  Q80={q80:.2f}%  Q90={q90_abs:.2f}%"
        )
        ax.set_title(f"{title_line1}\n{title_line2}", fontsize=9)
        ax.legend(loc="upper left", fontsize=7.5, ncol=1)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    fig.suptitle(
        f"Signed error distribution — `{variant}` (pooled across seeds)\n"
        f"std(E) shown in title; vertical lines = quantiles of |E|",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram_per_seed(df: pd.DataFrame, output_path: Path, variant: str) -> None:
    """Per-seed histogram on val split with rel_score annotations."""
    seeds = sorted(df["seed"].unique())
    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(5.2 * n_seeds, 4.6), sharey=True)
    if n_seeds == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds):
        sub = df[(df["seed"] == seed) & (df["split"] == "val")]
        actual = sub["actual"].to_numpy(dtype=float)
        pred = sub["prediction"].to_numpy(dtype=float)
        abs_error = np.abs(actual - pred) * 100
        rs = rel_score_fn(actual, pred)
        median_e = float(np.quantile(abs_error, 0.5))
        q90_e = float(np.quantile(abs_error, 0.9))
        ax.hist(abs_error.clip(0, 15.0), bins=70, color="#c44e52", alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.axvline(median_e, color="black", linestyle="--", linewidth=1.0, label=f"median={median_e:.2f}%")
        ax.axvline(q90_e, color="orange", linestyle="--", linewidth=1.2, label=f"Q90={q90_e:.2f}%")
        ax.axvline(3.0, color="green", linestyle=":", linewidth=1.0, label="target 3.0%")
        ax.axvline(3.5, color="red", linestyle=":", linewidth=1.0, label="violation 3.5%")
        ax.set_xlim(0, 15)
        ax.set_xlabel(r"$|\mathrm{actual}-\mathrm{prediction}|$ [%]")
        ax.set_title(f"seed={seed}  rel_score={rs:.4f}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    axes[0].set_ylabel("count")
    fig.suptitle(
        f"Per-seed validation absolute-error distribution — variant `{variant}`",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_signed_error_histogram(df: pd.DataFrame, output_path: Path, variant: str) -> None:
    """Plot signed error distribution to show direction bias and shape."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, split, color in zip(axes, ["train", "val"], ["#4c72b0", "#c44e52"]):
        subset = df[df["split"] == split]
        if subset.empty:
            continue
        actual = subset["actual"].to_numpy(dtype=float)
        pred = subset["prediction"].to_numpy(dtype=float)
        error = (actual - pred) * 100
        clipped = np.clip(error, -10, 10)
        mean_e = float(np.mean(error))
        std_e = float(np.std(error))
        skew = float(np.mean(((error - mean_e) / max(std_e, 1e-8)) ** 3))
        ax.hist(clipped, bins=80, color=color, alpha=0.75, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
        ax.axvline(mean_e, color="orange", linestyle="--", linewidth=1.0, label=f"mean={mean_e:.3f}%")
        ax.set_xlim(-10, 10)
        ax.set_xlabel("error = actual - prediction [%]")
        ax.set_ylabel("count")
        ax.set_title(
            f"{split.capitalize()} split  |  std={std_e:.3f}%  |  skew={skew:+.2f}  |  n={len(subset):,}",
            fontsize=11,
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    fig.suptitle(
        f"Signed error distribution — variant `{variant}` (pooled across seeds)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_relscore_bar(per_seed: pd.DataFrame, output_path: Path, variant: str) -> None:
    """Bar chart of rel_score per seed for train vs val."""
    pivot = per_seed.pivot(index="seed", columns="split", values="rel_score").sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.35
    seeds = pivot.index.tolist()
    x = np.arange(len(seeds))
    for offset, split, color in [(-bar_width / 2, "train", "#4c72b0"), (bar_width / 2, "val", "#c44e52")]:
        if split not in pivot.columns:
            continue
        values = pivot[split].values
        bars = ax.bar(x + offset, values, bar_width, label=split, color=color, edgecolor="black", linewidth=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:+.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("rel_score")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"rel_score per seed — variant `{variant}`", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_summary(output_dir: Path, per_seed: pd.DataFrame, variant: str) -> None:
    val = per_seed[per_seed["split"] == "val"].copy()
    train = per_seed[per_seed["split"] == "train"].copy()
    rows: list[dict[str, object]] = []
    for split, sub in [("train", train), ("val", val)]:
        for col in ["rel_score", "median_abs_error", "q90_abs_error", "q95_abs_error",
                    "daily_q90_p90", "daily_q90_max", "directional_accuracy",
                    "spike_days_ge_5pct", "spike_days_ge_7pct", "spike_days_ge_8pct",
                    "actual_q90", "pred_q90", "pred_actual_q90_ratio"]:
            if col not in sub.columns or sub.empty:
                continue
            vals = sub[col].astype(float)
            rows.append({
                "split": split,
                "metric": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            })
    agg = pd.DataFrame(rows)
    agg.to_csv(output_dir / "relscore_summary.csv", index=False)

    def fmt_pct(v: float) -> str:
        return f"{v*100:.2f}%"

    def fmt_score(v: float) -> str:
        return f"{v:+.5f}"

    lines: list[str] = []
    lines.append(f"# rel_score & Error Distribution — `{variant}`")
    lines.append("")
    lines.append("Scope: VN train + validation pooled over seeds 43, 52, 71. Holdout/test not used.")
    lines.append("")
    lines.append("## Per-Seed Metrics")
    lines.append("")
    disp = per_seed.copy().sort_values(["split", "seed"]).reset_index(drop=True)
    pct_cols = ["median_abs_error", "q90_abs_error", "q95_abs_error", "daily_q90_p90",
                "daily_q90_max", "actual_q90", "pred_q90", "directional_accuracy", "pred_actual_q90_ratio"]
    for col in pct_cols:
        if col in disp.columns:
            if col in ("directional_accuracy", "pred_actual_q90_ratio"):
                disp[col] = disp[col].astype(float).map(lambda v: f"{v:.4f}")
            else:
                disp[col] = disp[col].astype(float).map(fmt_pct)
    if "rel_score" in disp.columns:
        disp["rel_score"] = disp["rel_score"].astype(float).map(fmt_score)
    show_cols = ["split", "seed", "n_obs", "rel_score", "median_abs_error", "q90_abs_error",
                 "daily_q90_p90", "daily_q90_max", "directional_accuracy", "pred_actual_q90_ratio",
                 "spike_days_ge_5pct", "spike_days_ge_7pct", "spike_days_ge_8pct"]
    show_cols = [c for c in show_cols if c in disp.columns]
    lines.append(disp[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Aggregate (mean ± std across seeds)")
    lines.append("")
    val_rs = per_seed.loc[per_seed["split"] == "val", "rel_score"]
    train_rs = per_seed.loc[per_seed["split"] == "train", "rel_score"]
    lines.append(f"- **Train rel_score**: {train_rs.mean():+.4f} ± {train_rs.std(ddof=1):.4f}")
    lines.append(f"- **Validation rel_score**: {val_rs.mean():+.4f} ± {val_rs.std(ddof=1):.4f}")
    val_q90 = per_seed.loc[per_seed["split"] == "val", "q90_abs_error"]
    val_dailymax = per_seed.loc[per_seed["split"] == "val", "daily_q90_max"]
    val_da = per_seed.loc[per_seed["split"] == "val", "directional_accuracy"]
    val_spike8 = per_seed.loc[per_seed["split"] == "val", "spike_days_ge_8pct"]
    val_ratio = per_seed.loc[per_seed["split"] == "val", "pred_actual_q90_ratio"]
    lines.append(f"- **Validation q90(|E|)**: {val_q90.mean()*100:.2f}% ± {val_q90.std(ddof=1)*100:.2f}%")
    lines.append(f"- **Validation daily_q90 max**: {val_dailymax.mean()*100:.2f}% ± {val_dailymax.std(ddof=1)*100:.2f}%")
    lines.append(f"- **Validation directional accuracy**: {val_da.mean()*100:.2f}% ± {val_da.std(ddof=1)*100:.2f}%")
    lines.append(f"- **Validation spike days ≥8%**: {val_spike8.mean():.1f} ± {val_spike8.std(ddof=1):.1f}")
    lines.append(f"- **Validation pred/actual q90 ratio**: {val_ratio.mean():.3f} ± {val_ratio.std(ddof=1):.3f}")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("- `error_histogram_pooled.png` — absolute error distribution, train vs val, pooled across seeds.")
    lines.append("- `error_histogram_per_seed.png` — per-seed validation absolute-error histograms.")
    lines.append("- `signed_error_histogram.png` — signed error showing direction bias and skew.")
    lines.append("- `relscore_per_seed.png` — bar chart of rel_score per seed (train vs val).")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    df = load_predictions(args.probe_dir, args.variant, seeds)
    print(f"Loaded {len(df):,} prediction rows for variant {args.variant}, seeds {seeds}")
    per_seed = compute_per_seed_metrics(df)
    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    plot_error_histogram_pooled(df, args.output_dir / "error_histogram_pooled.png", args.variant)
    plot_error_histogram_per_seed(df, args.output_dir / "error_histogram_per_seed.png", args.variant)
    plot_signed_error_histogram(df, args.output_dir / "signed_error_histogram.png", args.variant)
    plot_relscore_bar(per_seed, args.output_dir / "relscore_per_seed.png", args.variant)
    write_summary(args.output_dir, per_seed, args.variant)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
