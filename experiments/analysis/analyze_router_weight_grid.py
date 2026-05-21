from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_regime_performance import (  # noqa: E402
    build_daily_quartile_returns,
    build_regime_filter_summary,
    rel_score,
)
from experiments.analysis.analyze_router_rolling_validation import (  # noqa: E402
    DEFAULT_ROUTER_REPORT,
    build_windows,
    max_drawdown,
)


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "router_weight_grid"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search anchor/challenger ensemble weights on validation only.")
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--stamp", default="20260426_r01")
    parser.add_argument("--output-name", default="anchor_sector19_weight_grid")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Write large per-row prediction and daily return CSVs for debugging.",
    )
    return parser.parse_args(argv)


def load_candidate_base(router_report: Path) -> pd.DataFrame:
    df = pd.read_csv(router_report / "candidate_predictions.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["actual_date"] = pd.to_datetime(df["actual_date"])
    return df[df["split"].isin({"train", "val"})].copy()


def build_weighted_long_frame(base: pd.DataFrame, weights: list[float]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    anchor = base["prediction__anchor"].to_numpy(dtype=float)
    challenger = base["prediction__challenger"].to_numpy(dtype=float)
    for weight in weights:
        candidate = f"w_challenger_{int(round(weight * 100)):03d}"
        part = base[["code", "split", "signal_date", "actual_date", "actual", "regime"]].copy()
        part["prediction"] = (1.0 - weight) * anchor + weight * challenger
        part["model"] = candidate
        part["run_name"] = candidate
        part["weight_challenger"] = weight
        part["error"] = part["actual"] - part["prediction"]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def summarize_prediction(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, split), group in long_df.groupby(["run_name", "split"], sort=True):
        weight = float(group["weight_challenger"].iloc[0])
        rows.append(
            {
                "candidate": candidate,
                "weight_challenger": weight,
                "split": split,
                "n_obs": int(len(group)),
                "rel_score": rel_score(group["error"], group["actual"]),
                "directional_accuracy": float((np.sign(group["prediction"]) == np.sign(group["actual"])).mean()),
                "error_q2": float(group["error"].quantile(0.2)),
                "error_q8": float(group["error"].quantile(0.8)),
            }
        )
    return pd.DataFrame(rows)


def summarize_year_stability(daily_quartile: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    def candidate_weight(candidate: str) -> float:
        prefix = "w_challenger_"
        if not candidate.startswith(prefix):
            return float("nan")
        return float(candidate.removeprefix(prefix)) / 100.0

    rows: list[dict[str, object]] = []
    val_years = windows[windows["scope"] == "val_year"].copy()
    val_daily = daily_quartile[daily_quartile["split"] == "val"].copy()
    for candidate, candidate_df in val_daily.groupby("run_name", sort=True):
        equities: list[float] = []
        drawdowns: list[float] = []
        hit_rates: list[float] = []
        for _, window in val_years.iterrows():
            segment = candidate_df[
                (candidate_df["actual_date"] >= pd.Timestamp(window["start_date"]))
                & (candidate_df["actual_date"] <= pd.Timestamp(window["end_date"]))
            ].copy()
            returns = segment["long_short_return"].to_numpy(dtype=float)
            equity = np.cumprod(1.0 + returns) if len(returns) else np.asarray([], dtype=float)
            equities.append(float(equity[-1]) if len(equity) else float("nan"))
            drawdowns.append(max_drawdown(equity))
            hit_rates.append(float(np.mean(returns > 0.0)) if len(returns) else float("nan"))
        clean_equities = np.asarray([value for value in equities if np.isfinite(value)], dtype=float)
        rows.append(
            {
                "candidate": candidate,
                "weight_challenger": candidate_weight(candidate),
                "years": int(len(clean_equities)),
                "avg_year_equity": float(np.mean(clean_equities)) if len(clean_equities) else float("nan"),
                "worst_year_equity": float(np.min(clean_equities)) if len(clean_equities) else float("nan"),
                "profitable_years": int(np.sum(clean_equities > 1.0)) if len(clean_equities) else 0,
                "avg_year_hit_rate": float(np.nanmean(hit_rates)),
                "worst_year_max_drawdown": float(np.nanmin(drawdowns)),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    output_dir: Path,
    prediction_summary: pd.DataFrame,
    filter_summary: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    def fmt_float(value: object, *, signed: bool = False) -> str:
        if pd.isna(value):
            return "n/a"
        prefix = "+" if signed else ""
        return f"{float(value):{prefix}.4f}" if signed else f"{float(value):.3f}"

    def fmt_int(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        return str(int(value))

    val_pred = prediction_summary[prediction_summary["split"] == "val"].copy()
    val_filters = filter_summary[(filter_summary["split"] == "val") & (filter_summary["filter_name"] == "all_regimes")].copy()
    merged = val_pred.merge(
        val_filters[["run_name", "final_equity", "hit_rate", "mean_return"]],
        left_on="candidate",
        right_on="run_name",
        how="left",
    ).merge(stability, on=["candidate", "weight_challenger"], how="left")

    lines = [
        "# Router Weight Grid",
        "",
        "Scope: train/validation predictions only. No test/out-sample data is used.",
        "",
        "Weight is the challenger share in `(1 - w) * anchor + w * challenger`.",
        "",
        "![Weight grid trade and rel_score](weight_grid_trade_relscore.png)",
        "",
        "## Validation Ranking By Trade Equity",
        "",
        "| w challenger | rel_score | Equity | Hit rate | Worst year equity | Profitable years |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in merged.sort_values("final_equity", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"{float(row['weight_challenger']):.2f} | {fmt_float(row['rel_score'], signed=True)} | "
            f"{fmt_float(row['final_equity'])} | {float(row['hit_rate']):.1%} | "
            f"{fmt_float(row['worst_year_equity'])} | {fmt_int(row['profitable_years'])} |"
        )

    lines.extend(
        [
            "",
            "## Validation Ranking By rel_score",
            "",
            "| w challenger | rel_score | Equity | Worst year equity |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in merged.sort_values("rel_score", ascending=False, kind="stable").head(12).iterrows():
        lines.append(
            f"| {float(row['weight_challenger']):.2f} | {fmt_float(row['rel_score'], signed=True)} | {fmt_float(row['final_equity'])} | {fmt_float(row['worst_year_equity'])} |"
        )

    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(output_dir: Path, prediction_summary: pd.DataFrame, filter_summary: pd.DataFrame, stability: pd.DataFrame) -> None:
    val_pred = prediction_summary[prediction_summary["split"] == "val"].copy()
    val_filters = filter_summary[(filter_summary["split"] == "val") & (filter_summary["filter_name"] == "all_regimes")].copy()
    merged = (
        val_pred.merge(
            val_filters[["run_name", "final_equity", "hit_rate"]],
            left_on="candidate",
            right_on="run_name",
            how="left",
        )
        .merge(stability[["candidate", "worst_year_equity"]], on="candidate", how="left")
        .sort_values("weight_challenger", kind="stable")
    )

    fig, left_axis = plt.subplots(figsize=(10, 5))
    right_axis = left_axis.twinx()
    x_values = merged["weight_challenger"].to_numpy(dtype=float)

    left_axis.plot(x_values, merged["final_equity"], marker="o", color="#1f77b4", label="Validation trade equity")
    left_axis.plot(x_values, merged["worst_year_equity"], marker="s", color="#2ca02c", label="Worst year equity")
    right_axis.plot(x_values, merged["rel_score"], marker="^", color="#d62728", label="Validation rel_score")

    left_axis.set_xlabel("Challenger weight")
    left_axis.set_ylabel("Quartile long-short equity")
    right_axis.set_ylabel("rel_score")
    left_axis.grid(True, alpha=0.25)
    left_axis.set_title("Anchor/Sector19 Weighted Router Grid")

    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "weight_grid_trade_relscore.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step <= 0.0 or args.step > 1.0:
        raise ValueError("--step must be in (0, 1].")
    weights = [round(value, 4) for value in np.arange(0.0, 1.0 + args.step / 2.0, args.step)]
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_candidate_base(args.router_report)
    long_df = build_weighted_long_frame(base, weights)
    prediction_summary = summarize_prediction(long_df)
    daily_quartile = build_daily_quartile_returns(long_df)
    filter_summary = build_regime_filter_summary(daily_quartile)
    stability = summarize_year_stability(daily_quartile, build_windows(base))

    prediction_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    filter_summary.to_csv(output_dir / "regime_filter_summary.csv", index=False)
    stability.to_csv(output_dir / "year_stability.csv", index=False)
    if args.keep_intermediate:
        long_df.to_csv(output_dir / "weighted_candidate_predictions.csv", index=False)
        daily_quartile.to_csv(output_dir / "daily_quartile_returns.csv", index=False)
    else:
        for stale_intermediate in ["weighted_candidate_predictions.csv", "daily_quartile_returns.csv"]:
            output_dir.joinpath(stale_intermediate).unlink(missing_ok=True)
    write_plot(output_dir, prediction_summary, filter_summary, stability)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "router_report": str(args.router_report),
                "weights": weights,
                "prediction_summary": prediction_summary.to_dict(orient="records"),
                "year_stability": stability.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, prediction_summary, filter_summary, stability)
    print(json.dumps({"output_dir": str(output_dir), "weights": weights}, indent=2))


if __name__ == "__main__":
    main()
