from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_prediction_router import add_candidate_predictions  # noqa: E402
from experiments.analysis.analyze_router_rolling_validation import DEFAULT_ROUTER_REPORT  # noqa: E402


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "cross_sectional_ic"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze daily cross-sectional IC for router candidates.")
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--stamp", default="20260426_r01")
    parser.add_argument("--output-name", default="anchor_sector19_cross_sectional_ic")
    parser.add_argument("--min-names-per-day", type=int, default=8)
    return parser.parse_args(argv)


def load_candidate_frame(router_report: Path) -> pd.DataFrame:
    df = pd.read_csv(router_report / "candidate_predictions.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["actual_date"] = pd.to_datetime(df["actual_date"])
    if not any(column.startswith("candidate__") for column in df.columns):
        df = add_candidate_predictions(df)
    return df[df["split"].isin({"train", "val"})].copy()


def build_long_candidates(df: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [column for column in df.columns if column.startswith("candidate__")]
    frames: list[pd.DataFrame] = []
    for column in candidate_columns:
        candidate = column.removeprefix("candidate__")
        part = df[["code", "split", "signal_date", "actual_date", "actual", "regime", column]].copy()
        part = part.rename(columns={column: "prediction"})
        part["candidate"] = candidate
        part["year"] = part["actual_date"].dt.year
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def daily_ic(long_df: pd.DataFrame, min_names_per_day: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["candidate", "split", "actual_date", "regime", "year"]
    for keys, group in long_df.groupby(group_cols, sort=True):
        if len(group) < min_names_per_day:
            continue
        ic = group["prediction"].corr(group["actual"], method="spearman")
        if not np.isfinite(ic):
            continue
        rows.append(
            {
                **dict(zip(group_cols, keys, strict=True)),
                "ic": float(ic),
                "name_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(daily: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in daily.groupby(group_cols, sort=True):
        values = group["ic"].to_numpy(dtype=float)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        mean = float(np.mean(values)) if len(values) else float("nan")
        t_stat = mean / (std / np.sqrt(len(values))) if len(values) > 1 and std > 0 else float("nan")
        rows.append(
            {
                **dict(zip(group_cols, keys, strict=True)),
                "days": int(len(values)),
                "mean_ic": mean,
                "ic_std": std,
                "t_stat": float(t_stat),
                "pct_positive": float(np.mean(values > 0.0)) if len(values) else float("nan"),
                "avg_names": float(group["name_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_plot(output_dir: Path, summary: pd.DataFrame) -> None:
    val = summary[summary["split"] == "val"].sort_values("mean_ic", ascending=True, kind="stable")
    fig, axis = plt.subplots(figsize=(10, 5))
    colors = ["#386641" if value > 0 else "#bc4749" for value in val["mean_ic"]]
    axis.barh(val["candidate"], val["mean_ic"], color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Validation mean daily Spearman IC")
    axis.set_title("Cross-Sectional IC By Candidate")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_mean_ic.png", dpi=160)
    plt.close(fig)


def write_markdown(
    output_dir: Path,
    overall: pd.DataFrame,
    by_regime: pd.DataFrame,
    by_year: pd.DataFrame,
) -> None:
    val = overall[overall["split"] == "val"].sort_values("mean_ic", ascending=False, kind="stable")
    lines = [
        "# Cross-Sectional IC",
        "",
        "Scope: train/validation predictions only. No test/out-sample data is used.",
        "",
        "IC is daily Spearman correlation between cross-sectional predictions and next-day realized returns.",
        "",
        "![Validation mean IC](validation_mean_ic.png)",
        "",
        "## Validation Ranking",
        "",
        "| Candidate | Mean IC | t-stat | Positive days | Days | Avg names |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in val.iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {float(row['mean_ic']):+.4f} | {float(row['t_stat']):+.2f} | "
            f"{float(row['pct_positive']):.1%} | {int(row['days'])} | {float(row['avg_names']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Validation By Regime",
            "",
            "| Candidate | Regime | Mean IC | t-stat | Positive days | Days |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    val_regime = by_regime[by_regime["split"] == "val"].sort_values(["candidate", "mean_ic"], ascending=[True, False], kind="stable")
    for _, row in val_regime.iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | `{row['regime']}` | {float(row['mean_ic']):+.4f} | {float(row['t_stat']):+.2f} | "
            f"{float(row['pct_positive']):.1%} | {int(row['days'])} |"
        )

    lines.extend(
        [
            "",
            "## Validation By Year",
            "",
            "| Candidate | Year | Mean IC | t-stat | Positive days | Days |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    val_year = by_year[by_year["split"] == "val"].sort_values(["candidate", "year"], kind="stable")
    for _, row in val_year.iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {int(row['year'])} | {float(row['mean_ic']):+.4f} | {float(row['t_stat']):+.2f} | "
            f"{float(row['pct_positive']):.1%} | {int(row['days'])} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_frame(args.router_report)
    long_df = build_long_candidates(candidates)
    daily = daily_ic(long_df, args.min_names_per_day)
    overall = summarize_ic(daily, ["candidate", "split"])
    by_regime = summarize_ic(daily, ["candidate", "split", "regime"])
    by_year = summarize_ic(daily, ["candidate", "split", "year"])

    daily.to_csv(output_dir / "daily_ic.csv", index=False)
    overall.to_csv(output_dir / "summary_overall.csv", index=False)
    by_regime.to_csv(output_dir / "summary_by_regime.csv", index=False)
    by_year.to_csv(output_dir / "summary_by_year.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "router_report": str(args.router_report),
                "min_names_per_day": args.min_names_per_day,
                "overall": overall.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_plot(output_dir, overall)
    write_markdown(output_dir, overall, by_regime, by_year)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
