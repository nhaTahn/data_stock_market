from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_regime_performance import (  # noqa: E402
    RegimeRuleConfig,
    build_daily_quartile_returns,
    build_daily_regimes,
    build_regime_filter_summary,
    load_predictions,
    rel_score,
)


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "router_analysis"


DEFAULT_ANCHOR_RUN = "broad_signmag_prune_general_sector_full_20260424_r04"
DEFAULT_CHALLENGER_RUN = "broad_signmag_prune_phase_ic_sector19_20260425_r09"
DEFAULT_MODEL = "lstm_signmag_best_by_val"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction ensemble/router candidates from existing runs."
    )
    parser.add_argument("--anchor-run", default=DEFAULT_ANCHOR_RUN)
    parser.add_argument("--challenger-run", default=DEFAULT_CHALLENGER_RUN)
    parser.add_argument("--anchor-model", default=DEFAULT_MODEL)
    parser.add_argument("--challenger-model", default=DEFAULT_MODEL)
    parser.add_argument("--stamp", default="20260425_r01")
    parser.add_argument("--output-name", default="anchor_sector19_router")
    return parser.parse_args(argv)


def _align_single_prediction_frame(df: pd.DataFrame, run_name: str, model_name: str) -> pd.DataFrame:
    model_df = df[(df["model"] == model_name) & (df["split"].isin({"train", "val"}))].copy()
    parts: list[pd.DataFrame] = []
    for (code, split), group in model_df.sort_values(["code", "split", "Date"], kind="stable").groupby(["code", "split"], sort=False):
        if len(group) < 3:
            continue
        signal_rows = group.iloc[1:-1].reset_index(drop=True)
        actual_rows = group.iloc[2:].reset_index(drop=True)
        parts.append(
            pd.DataFrame(
                {
                    "code": code,
                    "split": split,
                    "signal_date": signal_rows["Date"],
                    "actual_date": actual_rows["Date"],
                    f"prediction__{run_name}": signal_rows["prediction"].to_numpy(dtype=float),
                    "actual": actual_rows["actual"].to_numpy(dtype=float),
                }
            )
        )
    if not parts:
        raise RuntimeError(f"No aligned predictions for {run_name}:{model_name}")
    return pd.concat(parts, ignore_index=True)


def build_aligned_pair(anchor_run: str, anchor_model: str, challenger_run: str, challenger_model: str) -> pd.DataFrame:
    anchor_predictions = load_predictions(anchor_run)
    challenger_predictions = load_predictions(challenger_run)
    regimes = build_daily_regimes(anchor_predictions, RegimeRuleConfig())
    regime_lookup = regimes[["Date", "regime"]].rename(columns={"Date": "signal_date"})

    anchor = _align_single_prediction_frame(anchor_predictions, "anchor", anchor_model)
    challenger = _align_single_prediction_frame(challenger_predictions, "challenger", challenger_model).drop(columns=["actual"])
    merged = anchor.merge(challenger, on=["code", "split", "signal_date", "actual_date"], how="inner")
    merged = merged.merge(regime_lookup, on="signal_date", how="left")
    merged["regime"] = merged["regime"].fillna("unknown")
    return merged


def add_candidate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    anchor = out["prediction__anchor"].to_numpy(dtype=float)
    challenger = out["prediction__challenger"].to_numpy(dtype=float)
    regime = out["regime"].astype(str)

    out["candidate__anchor"] = anchor
    out["candidate__challenger"] = challenger
    out["candidate__avg_50_50"] = 0.5 * anchor + 0.5 * challenger
    out["candidate__avg_70_anchor"] = 0.7 * anchor + 0.3 * challenger
    out["candidate__avg_70_challenger"] = 0.3 * anchor + 0.7 * challenger
    out["candidate__sector19_down_up_anchor_else"] = np.where(
        regime.isin(["downtrend", "uptrend"]),
        challenger,
        anchor,
    )
    out["candidate__sector19_down_anchor_else"] = np.where(regime == "downtrend", challenger, anchor)
    out["candidate__sector19_up_anchor_else"] = np.where(regime == "uptrend", challenger, anchor)
    out["candidate__anchor_distribution_sideways_sector19_else"] = np.where(
        regime.isin(["distribution", "sideways"]),
        anchor,
        challenger,
    )
    return out


def summarize_candidates(candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_columns = [column for column in candidate_df.columns if column.startswith("candidate__")]
    summary_rows: list[dict[str, object]] = []
    long_frames: list[pd.DataFrame] = []
    for column in candidate_columns:
        candidate_name = column.removeprefix("candidate__")
        frame = candidate_df[
            ["code", "split", "signal_date", "actual_date", "actual", "regime", column]
        ].rename(columns={column: "prediction"}).copy()
        frame["model"] = candidate_name
        frame["run_name"] = candidate_name
        frame["error"] = frame["actual"] - frame["prediction"]
        long_frames.append(frame)
        for split, group in frame.groupby("split", sort=True):
            summary_rows.append(
                {
                    "candidate": candidate_name,
                    "split": split,
                    "n_obs": int(len(group)),
                    "rel_score": rel_score(group["error"], group["actual"]),
                    "directional_accuracy": float((np.sign(group["prediction"]) == np.sign(group["actual"])).mean()),
                    "error_q2": float(group["error"].quantile(0.2)),
                    "error_q8": float(group["error"].quantile(0.8)),
                }
            )
    long_df = pd.concat(long_frames, ignore_index=True)
    daily_quartile = build_daily_quartile_returns(long_df)
    filter_summary = build_regime_filter_summary(daily_quartile)
    return pd.DataFrame(summary_rows), daily_quartile, filter_summary


def write_markdown(summary: pd.DataFrame, filter_summary: pd.DataFrame, output_dir: Path) -> None:
    val_summary = summary[summary["split"] == "val"].copy()
    val_filters = filter_summary[filter_summary["split"] == "val"].copy()
    best_filters = (
        val_filters.sort_values(["run_name", "final_equity"], ascending=[True, False], kind="stable")
        .groupby("run_name", as_index=False)
        .head(1)
    )
    lines = [
        "# Prediction Router Analysis",
        "",
        "Scope: existing train/validation predictions only. No test/out-sample data is used.",
        "",
        "## Validation Prediction Metrics",
        "",
        "| Candidate | rel_score | Direction | Error q2/q8 | Obs |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for _, row in val_summary.sort_values("rel_score", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {float(row['rel_score']):+.4f} | {float(row['directional_accuracy']):.1%} | "
            f"{float(row['error_q2']):+.4f} / {float(row['error_q8']):+.4f} | {int(row['n_obs'])} |"
        )

    lines.extend(
        [
            "",
            "## Best Validation Trade Filter Per Candidate",
            "",
            "| Candidate | Best filter | Trade days | Equity | Hit rate | Mean return |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in best_filters.sort_values("final_equity", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['run_name']}` | `{row['filter_name']}` | {int(row['trade_days'])} | "
            f"{float(row['final_equity']):.3f} | {float(row['hit_rate']):.1%} | {float(row['mean_return']):+.4f} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned = build_aligned_pair(args.anchor_run, args.anchor_model, args.challenger_run, args.challenger_model)
    candidates = add_candidate_predictions(aligned)
    summary, daily_quartile, filter_summary = summarize_candidates(candidates)
    aligned.to_csv(output_dir / "aligned_anchor_challenger.csv", index=False)
    candidates.to_csv(output_dir / "candidate_predictions.csv", index=False)
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    daily_quartile.to_csv(output_dir / "daily_quartile_returns.csv", index=False)
    filter_summary.to_csv(output_dir / "regime_filter_summary.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "anchor_run": args.anchor_run,
                "challenger_run": args.challenger_run,
                "anchor_model": args.anchor_model,
                "challenger_model": args.challenger_model,
                "summary": summary.to_dict(orient="records"),
                "regime_filter_summary": filter_summary.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(summary, filter_summary, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "candidates": len(summary["candidate"].unique())}, indent=2))


if __name__ == "__main__":
    main()
