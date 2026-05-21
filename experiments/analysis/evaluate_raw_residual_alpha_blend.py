from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.evaluate_residual_market_component_processing import pct, summarize
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "residual_target_probe_20260519"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "raw_residual_alpha_blend_20260520"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "raw_residual_alpha_blend_20260520"


BLEND_WEIGHTS = (-0.50, -0.25, 0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 1.0, 1.25, 1.50)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether residual-alpha target predictions add useful signal to raw LSTM predictions."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--min-daily-n", type=int, default=20)
    parser.add_argument("--target-error", type=float, default=0.035)
    return parser.parse_args(argv)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def split_late_train(train: pd.DataFrame, calibration_fraction: float) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(sorted(train["Date"].dropna().unique())))
    cutoff_idx = max(1, int(len(dates) * (1.0 - calibration_fraction)))
    cutoff = dates.iloc[cutoff_idx - 1]
    return train[train["Date"].gt(cutoff)].copy()


def objective(row: dict[str, object], target_error: float) -> float:
    return (
        float(row["rel_score"])
        - 1.5 * max(0.0, float(row["daily_q90_p90"]) - target_error)
        - 1.2 * max(0.0, float(row["daily_q90_max"]) - 0.060)
        - 0.003 * float(row["days_ge_5"])
        - 0.006 * float(row["days_ge_7"])
    )


def load_seed_frame(source_dir: Path, seed: int) -> pd.DataFrame:
    raw_path = source_dir / f"predictions_raw_baseline_seed_{seed}.csv"
    residual_path = source_dir / f"predictions_residual_lagged_ar1_seed_{seed}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not residual_path.exists():
        raise FileNotFoundError(residual_path)

    usecols = ["code", "Date", "split", "seed", "prediction_raw", "prediction_target_space", "actual"]
    raw = pd.read_csv(raw_path, usecols=usecols, parse_dates=["Date"])
    residual = pd.read_csv(residual_path, usecols=usecols, parse_dates=["Date"])
    for frame in (raw, residual):
        frame["code"] = frame["code"].astype(str).str.upper()

    merged = raw.rename(
        columns={
            "prediction_raw": "raw_pred",
            "prediction_target_space": "raw_target_space",
        }
    ).merge(
        residual.rename(
            columns={
                "prediction_raw": "residual_lagged_pred",
                "prediction_target_space": "alpha_pred",
                "split": "split_residual",
                "seed": "seed_residual",
                "actual": "actual_residual",
            }
        ),
        on=["code", "Date"],
        how="inner",
    )
    if not merged["split"].equals(merged["split_residual"]):
        raise ValueError(f"Split mismatch for seed {seed}")
    if not np.allclose(merged["actual"].to_numpy(float), merged["actual_residual"].to_numpy(float), equal_nan=True):
        raise ValueError(f"Actual mismatch for seed {seed}")
    return merged.drop(columns=["split_residual", "seed_residual", "actual_residual"])


def add_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pred_raw_baseline"] = out["raw_pred"].astype(float)
    out["pred_alpha_only"] = out["alpha_pred"].astype(float)
    out["pred_existing_residual_lagged_ar1"] = out["residual_lagged_pred"].astype(float)
    for weight in BLEND_WEIGHTS:
        col = f"pred_blend_w_{weight:g}".replace("-", "neg").replace(".", "p")
        out[col] = (1.0 - weight) * out["raw_pred"].astype(float) + weight * out["alpha_pred"].astype(float)
    return out


def evaluate_seed(seed: int, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_predictions(load_seed_frame(args.source_dir, seed))
    late_train = split_late_train(frame[frame["split"].eq("train")], args.calibration_fraction)
    rows: list[dict[str, object]] = []
    best_col = "pred_raw_baseline"
    best_score = -np.inf

    for weight in BLEND_WEIGHTS:
        col = f"pred_blend_w_{weight:g}".replace("-", "neg").replace(".", "p")
        row = summarize(late_train, col, seed=seed, policy=f"blend_w_{weight:g}", min_daily_n=args.min_daily_n)
        row["selection_split"] = "late_train"
        row["blend_weight"] = weight
        row["objective"] = objective(row, args.target_error)
        rows.append(row)
        if float(row["objective"]) > best_score:
            best_score = float(row["objective"])
            best_col = col

    val = frame[frame["split"].eq("val")].copy()
    fixed_policies = {
        "raw_baseline": "pred_raw_baseline",
        "alpha_only_no_market": "pred_alpha_only",
        "existing_residual_lagged_ar1": "pred_existing_residual_lagged_ar1",
    }
    for policy, col in fixed_policies.items():
        row = summarize(val, col, seed=seed, policy=policy, min_daily_n=args.min_daily_n)
        row["selection_split"] = "val"
        row["blend_weight"] = np.nan
        rows.append(row)

    selected_weight = float(best_col.removeprefix("pred_blend_w_").replace("neg", "-").replace("p", "."))
    selected = summarize(val, best_col, seed=seed, policy=f"selected_blend_w_{selected_weight:g}", min_daily_n=args.min_daily_n)
    selected["selection_split"] = "val"
    selected["blend_weight"] = selected_weight
    rows.append(selected)

    val_keep = val.loc[
        :,
        [
            "seed",
            "code",
            "Date",
            "actual",
            "pred_raw_baseline",
            "pred_alpha_only",
            "pred_existing_residual_lagged_ar1",
            best_col,
        ],
    ].copy()
    val_keep = val_keep.rename(columns={best_col: "pred_selected_blend"})
    val_keep["selected_blend_weight"] = selected_weight
    return pd.DataFrame(rows), val_keep


def aggregate_modes(rows: pd.DataFrame) -> pd.DataFrame:
    val = rows[rows["selection_split"].eq("val")].copy()
    val["mode"] = val["policy"].map(lambda value: "selected_blend" if str(value).startswith("selected_blend") else str(value))
    return (
        val.groupby("mode", sort=True)
        .agg(
            seeds=("seed", "nunique"),
            policies=("policy", lambda values: ", ".join(sorted(set(map(str, values))))),
            rel_score_mean=("rel_score", "mean"),
            rel_score_std=("rel_score", "std"),
            q90_abs_error_mean=("q90_abs_error", "mean"),
            daily_q90_median_mean=("daily_q90_median", "mean"),
            daily_q90_p90_mean=("daily_q90_p90", "mean"),
            daily_q90_max_mean=("daily_q90_max", "mean"),
            days_ge_3p5_mean=("days_ge_3p5", "mean"),
            days_ge_5_mean=("days_ge_5", "mean"),
            days_ge_7_mean=("days_ge_7", "mean"),
            days_ge_8_mean=("days_ge_8", "mean"),
            prediction_abs_q90_mean=("prediction_abs_q90", "mean"),
            actual_abs_q90_mean=("actual_abs_q90", "mean"),
        )
        .reset_index()
    )


def write_frontier(gold_dir: Path, mode_agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = {
        "raw_baseline": "#2563eb",
        "alpha_only_no_market": "#16a34a",
        "existing_residual_lagged_ar1": "#9333ea",
        "selected_blend": "#dc2626",
    }
    for _, row in mode_agg.iterrows():
        mode = str(row["mode"])
        ax.scatter(
            100 * float(row["daily_q90_p90_mean"]),
            100 * float(row["daily_q90_max_mean"]),
            s=95,
            color=colors.get(mode, "#64748b"),
            alpha=0.85,
        )
        ax.annotate(mode, (100 * float(row["daily_q90_p90_mean"]), 100 * float(row["daily_q90_max_mean"])), fontsize=9)
    ax.axvline(3.5, color="#dc2626", linestyle="--", linewidth=1.0)
    ax.axhline(8.0, color="#f97316", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Daily q90(|E|) p90 (%)")
    ax.set_ylabel("Daily q90(|E|) max (%)")
    ax.set_title("Raw LSTM vs residual-alpha blend")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(gold_dir / "raw_residual_alpha_blend_frontier.png", dpi=180)
    plt.close(fig)


def write_summary(gold_dir: Path, mode_agg: pd.DataFrame) -> None:
    display = mode_agg.sort_values(["daily_q90_p90_mean", "daily_q90_max_mean"], ascending=[True, True]).copy()
    lines = [
        "# Raw + Residual-Alpha Blend",
        "",
        "Scope: reuse existing 3-seed predictions from residual target probe. Blend weight is selected on late-train only, then evaluated on validation.",
        "",
        "| mode | policies | seeds | rel_score | q90 abs E | daily median | daily p90 | daily max | days >=5% | days >=7% | days >=8% | pred abs q90 |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"| `{row['mode']}` | `{row.policies}` | {int(row.seeds)} | {float(row.rel_score_mean):.5f} | "
            f"{pct(row.q90_abs_error_mean)} | {pct(row.daily_q90_median_mean)} | {pct(row.daily_q90_p90_mean)} | "
            f"{pct(row.daily_q90_max_mean)} | {float(row.days_ge_5_mean):.1f} | {float(row.days_ge_7_mean):.1f} | "
            f"{float(row.days_ge_8_mean):.1f} | {pct(row.prediction_abs_q90_mean)} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "- `alpha_only_no_market` tests residual target without adding any market component back.",
        "- `selected_blend` tests whether raw-return prediction and residual-alpha prediction are complementary under a late-train selection rule.",
        "- If selected blend fails to beat `raw_baseline`, residual target is not yet a practical input/target processing improvement for full-coverage next-day return.",
    ]
    (gold_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    row_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    seeds = parse_seeds(args.seeds)
    for seed in seeds:
        rows, predictions = evaluate_seed(seed, args)
        row_parts.append(rows)
        prediction_parts.append(predictions)
    rows_all = pd.concat(row_parts, ignore_index=True)
    predictions_all = pd.concat(prediction_parts, ignore_index=True)
    mode_agg = aggregate_modes(rows_all)

    rows_all.to_csv(args.output_dir / "raw_residual_alpha_blend_by_seed.csv", index=False)
    predictions_all.to_csv(args.output_dir / "raw_residual_alpha_blend_val_predictions.csv", index=False)
    mode_agg.to_csv(args.output_dir / "raw_residual_alpha_blend_mode_aggregate.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(args.source_dir),
                "seeds": seeds,
                "blend_weights": list(BLEND_WEIGHTS),
                "calibration_fraction": args.calibration_fraction,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name in (
        "raw_residual_alpha_blend_by_seed.csv",
        "raw_residual_alpha_blend_mode_aggregate.csv",
        "manifest.json",
    ):
        (args.gold_dir / name).write_bytes((args.output_dir / name).read_bytes())
    write_frontier(args.gold_dir, mode_agg)
    write_summary(args.gold_dir, mode_agg)
    print((args.gold_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
