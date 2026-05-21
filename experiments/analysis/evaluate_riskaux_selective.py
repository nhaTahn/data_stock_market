from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "riskaux_lstm_probe_20260520"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "riskaux_lstm_probe_20260520"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selective prediction using LSTM risk auxiliary head.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seeds", default="52")
    parser.add_argument(
        "--variants",
        default="plain_global_weighted_mild_tail35_riskaux_w10,plain_global_weighted_mild_tail35_riskaux_w20",
    )
    parser.add_argument("--target-error", type=float, default=0.035)
    parser.add_argument("--min-daily-n", type=int, default=20)
    return parser.parse_args(argv)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def daily_q90(frame: pd.DataFrame, min_daily_n: int) -> pd.Series:
    counts = frame.groupby("Date", sort=True)["abs_error"].count()
    daily = frame.groupby("Date", sort=True)["abs_error"].quantile(0.90)
    return daily[counts >= min_daily_n]


def summarize(
    frame: pd.DataFrame,
    selected: pd.Series,
    *,
    seed: int,
    variant: str,
    policy: str,
    threshold: float,
    min_daily_n: int,
) -> dict[str, object]:
    part = frame.loc[selected].copy()
    actual = part["actual"].to_numpy(dtype=float)
    pred = part["prediction"].to_numpy(dtype=float)
    daily = daily_q90(part, min_daily_n)
    return {
        "seed": seed,
        "variant": variant,
        "policy": policy,
        "threshold": threshold,
        "n_obs": int(len(part)),
        "obs_coverage": float(len(part) / max(len(frame), 1)),
        "n_days": int(part["Date"].nunique()),
        "day_coverage": float(part["Date"].nunique() / max(frame["Date"].nunique(), 1)),
        "rel_score": rel_score(actual, pred) if len(part) else float("nan"),
        "q90_abs_error": float(np.quantile(np.abs(actual - pred), 0.90)) if len(part) else float("nan"),
        "daily_q90_median": float(daily.median()) if not daily.empty else float("nan"),
        "daily_q90_p90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "daily_q90_max": float(daily.max()) if not daily.empty else float("nan"),
        "days_ge_3p5": int(daily.ge(0.035).sum()) if not daily.empty else 0,
        "days_ge_5": int(daily.ge(0.05).sum()) if not daily.empty else 0,
        "days_ge_7": int(daily.ge(0.07).sum()) if not daily.empty else 0,
        "days_ge_8": int(daily.ge(0.08).sum()) if not daily.empty else 0,
    }


def choose_threshold(train: pd.DataFrame, target_error: float, min_daily_n: int) -> tuple[float, str]:
    scores = train["risk_probability"].dropna().to_numpy(dtype=float)
    if scores.size == 0:
        return float("nan"), "missing_risk"
    best_threshold = float(np.quantile(scores, 0.20))
    best_policy = "lowest_risk_available"
    best_coverage = -np.inf
    for quantile in np.linspace(0.20, 1.0, 17):
        threshold = float(np.quantile(scores, quantile))
        selected = train["risk_probability"].le(threshold)
        daily = daily_q90(train.loc[selected], min_daily_n)
        if daily.empty:
            continue
        if float(daily.quantile(0.90)) <= target_error:
            coverage = float(selected.mean())
            if coverage > best_coverage:
                best_threshold = threshold
                best_policy = "calibrated_train_p90"
                best_coverage = coverage
    return best_threshold, best_policy


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"])
    if "risk_probability" not in frame.columns:
        raise ValueError(f"Missing risk_probability in {path}")
    frame["abs_error"] = (frame["actual"].astype(float) - frame["prediction"].astype(float)).abs()
    return frame.dropna(subset=["Date", "actual", "prediction", "risk_probability"])


def evaluate_variant(run_root: Path, seed: int, variant: str, target_error: float, min_daily_n: int) -> pd.DataFrame:
    path = run_root / f"seed_{seed}" / f"predictions_{variant}.csv"
    frame = read_prediction(path)
    train = frame[frame["split"].eq("train")].copy()
    val = frame[frame["split"].eq("val")].copy()
    rows: list[dict[str, object]] = []
    threshold, policy = choose_threshold(train, target_error, min_daily_n)
    for quantile in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0):
        q_threshold = float(np.quantile(train["risk_probability"].to_numpy(dtype=float), quantile))
        rows.append(
            summarize(
                val,
                val["risk_probability"].le(q_threshold),
                seed=seed,
                variant=variant,
                policy=f"risk_q{int(quantile * 100)}",
                threshold=q_threshold,
                min_daily_n=min_daily_n,
            )
        )
    rows.append(
        summarize(
            val,
            val["risk_probability"].le(threshold),
            seed=seed,
            variant=variant,
            policy=policy,
            threshold=threshold,
            min_daily_n=min_daily_n,
        )
    )
    rows.append(
        summarize(
            val,
            pd.Series(True, index=val.index),
            seed=seed,
            variant=variant,
            policy="full_coverage",
            threshold=float("inf"),
            min_daily_n=min_daily_n,
        )
    )
    return pd.DataFrame(rows)


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["variant", "policy"], sort=True)
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


def write_plot(gold_dir: Path, agg: pd.DataFrame) -> None:
    plot = agg[agg["policy"].isin(["full_coverage", "calibrated_train_p90", "risk_q20", "risk_q30", "risk_q40", "risk_q50"])].copy()
    plot["label"] = plot["variant"].str.replace("plain_global_weighted_mild_tail35_", "", regex=False) + "/" + plot["policy"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(100.0 * plot["obs_coverage_mean"], 100.0 * plot["daily_q90_p90_mean"], s=72, alpha=0.85)
    for _, row in plot.iterrows():
        ax.annotate(str(row["label"]), (100.0 * row["obs_coverage_mean"], 100.0 * row["daily_q90_p90_mean"]), fontsize=8)
    ax.axhline(3.5, color="#dc2626", linestyle="--", linewidth=1, label="3.5% target")
    ax.set_xlabel("Observation coverage (%)")
    ax.set_ylabel("Validation daily q90(|E|) p90 (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(gold_dir / "riskaux_selective_frontier.png", dpi=180)
    plt.close(fig)


def pct(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * float(value):.2f}%"


def write_summary(gold_dir: Path, agg: pd.DataFrame) -> None:
    display = agg.sort_values(["daily_q90_p90_mean", "obs_coverage_mean"], ascending=[True, False]).copy()
    lines = [
        "# RiskAux Selective Readout",
        "",
        "Risk score comes from an LSTM auxiliary head. Detached variants stop risk-head gradients from updating the return backbone.",
        "",
        "| variant | policy | obs coverage | day coverage | rel_score | q90 error | daily median | daily p90 | daily max | days >=3.5% | days >=8% |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.head(24).iterrows():
        lines.append(
            f"| `{row.variant}` | `{row.policy}` | {pct(row.obs_coverage_mean)} | {pct(row.day_coverage_mean)} | "
            f"{float(row.rel_score_mean):.5f} | {pct(row.q90_abs_error_mean)} | {pct(row.daily_q90_median_mean)} | "
            f"{pct(row.daily_q90_p90_mean)} | {pct(row.daily_q90_max_mean)} | {float(row.days_ge_3p5_mean):.1f} | "
            f"{float(row.days_ge_8_mean):.1f} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "- If RiskAux beats the post-hoc input-noise/disagreement scores at similar coverage, the learned confidence head is worth promoting.",
        "- If it only matches post-hoc filters, the next improvement should focus on cleaner input features and stronger calibration rather than more heads.",
    ]
    (gold_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    variants = parse_csv(args.variants)
    frames = [
        evaluate_variant(args.run_root, seed, variant, args.target_error, args.min_daily_n)
        for seed in seeds
        for variant in variants
    ]
    rows = pd.concat(frames, ignore_index=True)
    agg = aggregate(rows)
    rows.to_csv(args.gold_dir / "riskaux_selective_by_seed.csv", index=False)
    agg.to_csv(args.gold_dir / "riskaux_selective_aggregate.csv", index=False)
    (args.gold_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_root": str(args.run_root),
                "seeds": seeds,
                "variants": variants,
                "target_error": args.target_error,
                "min_daily_n": args.min_daily_n,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_plot(args.gold_dir, agg)
    write_summary(args.gold_dir, agg)
    print((args.gold_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
