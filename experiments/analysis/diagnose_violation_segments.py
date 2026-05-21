"""Diagnose top violation segments: identify per-stock errors and form hypotheses.

For each top segment found by replot_teacher_style_with_thresholds.py, we look up
prediction CSVs from the all-VN portable run, filter to the segment date range,
and rank stocks by abs error contribution. Outputs a diagnostic markdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VIOLATIONS = (
    ROOT / "gold" / "vn_transition_pressure_20260512" / "plots"
    / "teacher_style_threshold_replot_20260521" / "violation_segments.csv"
)
DEFAULT_PREDICTIONS_DIR = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
    / "reports" / "residual_target_probe_20260519"
)
DEFAULT_OUTPUT = (
    ROOT / "gold" / "vn_transition_pressure_20260512" / "plots"
    / "teacher_style_threshold_replot_20260521"
)
DEFAULT_GOLD_CSV = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--violations", type=Path, default=DEFAULT_VIOLATIONS)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-segments", type=int, default=8)
    parser.add_argument("--top-stocks", type=int, default=10)
    return parser.parse_args(argv)


def load_predictions(predictions_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """Load raw_baseline predictions from residual probe (3 seeds, train+val)."""
    parts: list[pd.DataFrame] = []
    for seed in seeds:
        path = predictions_dir / f"predictions_raw_baseline_seed_{seed}.csv"
        if not path.exists():
            print(f"Warning: missing {path.name}")
            continue
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df["seed"] = seed
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No predictions found in {predictions_dir}")
    return pd.concat(parts, ignore_index=True)


def load_sector_map(gold_csv: Path) -> dict[str, str]:
    """Load sector mapping from gold dataset."""
    df = pd.read_csv(gold_csv, usecols=["code", "sector"])
    df = df.drop_duplicates(subset=["code"])
    return dict(zip(df["code"].astype(str).str.upper(), df["sector"].fillna("Unknown")))


def diagnose_segment(
    pred_df: pd.DataFrame,
    sector_map: dict[str, str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    top_stocks: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """For one segment, identify top error stocks and compute diagnostics."""
    seg = pred_df.loc[(pred_df["Date"] >= start_date) & (pred_df["Date"] <= end_date)].copy()
    if seg.empty:
        return pd.DataFrame(), {}
    seg["abs_error"] = (seg["actual"] - seg["prediction_raw"]).abs()
    seg["sign_match"] = (np.sign(seg["actual"]) == np.sign(seg["prediction_raw"])).astype(int)
    seg["abs_actual"] = seg["actual"].abs()
    seg["abs_pred"] = seg["prediction_raw"].abs()
    # Average across seeds for each (code, Date)
    by_code = seg.groupby("code").agg(
        n_obs=("abs_error", "size"),
        mean_abs_error=("abs_error", "mean"),
        max_abs_error=("abs_error", "max"),
        mean_abs_actual=("abs_actual", "mean"),
        mean_abs_pred=("abs_pred", "mean"),
        sign_match_rate=("sign_match", "mean"),
    ).reset_index()
    by_code["sector"] = by_code["code"].map(sector_map).fillna("Unknown")
    by_code["pred_actual_ratio"] = by_code["mean_abs_pred"] / by_code["mean_abs_actual"].clip(lower=1e-6)
    by_code = by_code.sort_values("mean_abs_error", ascending=False).head(top_stocks).reset_index(drop=True)
    # Segment-level stats
    diag = {
        "median_q90_actual": float(seg.groupby("Date")["abs_actual"].quantile(0.90).median()),
        "median_q90_pred": float(seg.groupby("Date")["abs_pred"].quantile(0.90).median()),
        "median_pred_actual_q90_ratio": float(
            (seg.groupby("Date")["abs_pred"].quantile(0.90)
             / seg.groupby("Date")["abs_actual"].quantile(0.90).clip(lower=1e-6)).median()
        ),
        "median_sign_mismatch_rate": float(1.0 - seg.groupby("Date")["sign_match"].mean().median()),
        "n_unique_stocks": int(seg["code"].nunique()),
        "n_days": int(seg["Date"].nunique()),
    }
    return by_code, diag


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    violations = pd.read_csv(args.violations)
    violations["start_date"] = pd.to_datetime(violations["start_date"])
    violations["end_date"] = pd.to_datetime(violations["end_date"])
    print(f"Loaded {len(violations)} violation segments")
    pred_df = load_predictions(args.predictions_dir, seeds=[43, 52, 71])
    print(f"Loaded {len(pred_df)} prediction rows from {pred_df['Date'].min().date()} to {pred_df['Date'].max().date()}")
    sector_map = load_sector_map(args.gold_csv)

    # Filter only segments with predictions available
    pred_min, pred_max = pred_df["Date"].min(), pred_df["Date"].max()
    in_range = (violations["start_date"] >= pred_min) & (violations["end_date"] <= pred_max)
    available = violations.loc[in_range].head(args.top_segments)
    print(f"{len(available)} segments overlap with prediction date range")

    lines: list[str] = []
    lines.append("# Violation Segment Diagnostics")
    lines.append("")
    lines.append(f"Source predictions: `{args.predictions_dir.name}` (raw_baseline, seeds 43/52/71).")
    lines.append(f"Prediction date range: {pred_min.date()} to {pred_max.date()}.")
    lines.append("")
    lines.append("## Top Violation Segments In Prediction Range")
    lines.append("")
    if available.empty:
        lines.append("_Không có segment nào trong khoảng prediction (predictions thuộc validation 2020-04 → 2022-11)._")
        lines.append("")
        lines.append("## Diagnostic All Validation Days With High Q90")
        lines.append("")
        # Fallback: identify top error days within prediction range
        pred_df["abs_error"] = (pred_df["actual"] - pred_df["prediction_raw"]).abs()
        daily_q90 = pred_df.groupby("Date")["abs_error"].quantile(0.90).reset_index(name="q90_abs_error")
        top_days = daily_q90.sort_values("q90_abs_error", ascending=False).head(20)
        lines.append("Top 20 ngày có Q90(|E|) cao nhất trong validation:")
        lines.append("")
        top_days["q90_abs_error"] = (top_days["q90_abs_error"] * 100).map(lambda v: f"{v:.2f}%")
        lines.append(top_days.to_markdown(index=False))
        lines.append("")
    else:
        all_diag_rows: list[dict[str, object]] = []
        all_top_codes: dict[str, dict] = {}
        for _, seg_row in available.iterrows():
            top_codes, diag = diagnose_segment(
                pred_df, sector_map,
                seg_row["start_date"], seg_row["end_date"], args.top_stocks,
            )
            label = f"{seg_row['start_date'].date()} → {seg_row['end_date'].date()}"
            lines.append(f"### Segment {label} (year {int(seg_row['year'])}, {int(seg_row['n_days'])} days)")
            lines.append("")
            lines.append(f"- Max error in segment: {seg_row['max_error']*100:.2f}%")
            lines.append(f"- Median segment error: {seg_row['median_error']*100:.2f}%")
            lines.append(f"- Index change: {seg_row['index_change']*100:+.1f}%")
            if diag:
                lines.append(f"- Median Q90 of |actual| return: {diag['median_q90_actual']*100:.2f}%")
                lines.append(f"- Median Q90 of |prediction|: {diag['median_q90_pred']*100:.2f}%")
                lines.append(f"- Median pred/actual Q90 ratio: {diag['median_pred_actual_q90_ratio']:.3f}")
                lines.append(f"- Median sign-mismatch rate: {diag['median_sign_mismatch_rate']*100:.1f}%")
            lines.append("")
            if not top_codes.empty:
                disp = top_codes.copy()
                for col in ["mean_abs_error", "max_abs_error", "mean_abs_actual", "mean_abs_pred", "sign_match_rate"]:
                    disp[col] = (disp[col] * 100).map(lambda v: f"{v:.2f}%")
                disp["pred_actual_ratio"] = disp["pred_actual_ratio"].map(lambda v: f"{v:.3f}")
                lines.append(disp.to_markdown(index=False))
                lines.append("")
            all_diag_rows.append({"label": label, **diag})
        # Save all_diag
        if all_diag_rows:
            pd.DataFrame(all_diag_rows).to_csv(args.output_dir / "segment_diagnostics.csv", index=False)
    (args.output_dir / "violation_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {args.output_dir / 'violation_diagnostics.md'}")


if __name__ == "__main__":
    main()
