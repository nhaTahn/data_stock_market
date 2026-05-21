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

from experiments.analysis.analyze_prediction_router import add_candidate_predictions  # noqa: E402
from experiments.analysis.analyze_rank_objective_offline import compute_daily_rank_metrics  # noqa: E402
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
REPORT_ROOT = RUN_ROOT / "reports" / "rank_router_train_selected"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select anchor/challenger router weights on train rank objectives, then validate."
    )
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--stamp", default="20260427_r01")
    parser.add_argument("--output-name", default="anchor_sector19_rank_router")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--min-group-days", type=int, default=120)
    return parser.parse_args(argv)


def load_base(router_report: Path) -> pd.DataFrame:
    df = pd.read_csv(router_report / "candidate_predictions.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["actual_date"] = pd.to_datetime(df["actual_date"])
    if not any(column.startswith("candidate__") for column in df.columns):
        df = add_candidate_predictions(df)
    return df[df["split"].isin({"train", "val"})].copy()


def weighted_prediction(base: pd.DataFrame, weight_challenger: float) -> np.ndarray:
    anchor = base["prediction__anchor"].to_numpy(dtype=float)
    challenger = base["prediction__challenger"].to_numpy(dtype=float)
    return (1.0 - weight_challenger) * anchor + weight_challenger * challenger


def frame_with_prediction(base: pd.DataFrame, candidate: str, prediction: np.ndarray) -> pd.DataFrame:
    out = base[["code", "split", "signal_date", "actual_date", "actual", "regime"]].copy()
    out["prediction"] = prediction
    out["model"] = candidate
    out["run_name"] = candidate
    out["error"] = out["actual"] - out["prediction"]
    return out


def candidate_frame(base: pd.DataFrame, candidate: str) -> pd.DataFrame:
    column = f"candidate__{candidate}"
    if column not in base.columns:
        raise ValueError(f"Missing candidate column: {column}")
    return frame_with_prediction(base, candidate, base[column].to_numpy(dtype=float))


def daily_rank_metrics(frame: pd.DataFrame, split: str, regime: str | None = None) -> pd.DataFrame:
    selected = frame[frame["split"] == split].copy()
    if regime is not None:
        selected = selected[selected["regime"] == regime].copy()
    rows: list[dict[str, object]] = []
    for actual_date, group in selected.groupby("actual_date", sort=True):
        rows.append({"actual_date": actual_date, **compute_daily_rank_metrics(group)})
    return pd.DataFrame(rows)


def rank_objective_score(frame: pd.DataFrame, objective: str, split: str = "train", regime: str | None = None) -> float:
    daily = daily_rank_metrics(frame, split=split, regime=regime)
    if daily.empty:
        return float("nan")
    if objective == "mean_ic":
        return float(daily["spearman_ic"].mean())
    if objective == "top_bottom_equity":
        returns = daily["top_bottom_return"].dropna().to_numpy(dtype=float)
        return float(np.prod(1.0 + returns)) if len(returns) else float("nan")
    if objective == "top_bottom_mean_return":
        return float(daily["top_bottom_return"].mean())
    raise ValueError(f"Unsupported rank objective: {objective}")


def choose_global_weight(base: pd.DataFrame, weights: list[float], objective: str) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for weight in weights:
        frame = frame_with_prediction(base, "candidate", weighted_prediction(base, weight))
        score = rank_objective_score(frame, objective, split="train")
        rows.append(
            {
                "scope": "global",
                "group": "ALL",
                "objective": objective,
                "weight_challenger": weight,
                "train_score": score,
                "train_days": int(frame.loc[frame["split"] == "train", "actual_date"].nunique()),
                "selected": False,
            }
        )
    search = pd.DataFrame(rows)
    best = search[np.isfinite(search["train_score"])].sort_values(
        ["train_score", "weight_challenger"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    selected = float(best["weight_challenger"])
    search.loc[search["weight_challenger"] == selected, "selected"] = True
    return selected, search


def choose_regime_weights(
    base: pd.DataFrame,
    weights: list[float],
    objective: str,
    min_group_days: int,
    fallback_weight: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    train = base[base["split"] == "train"].copy()
    for regime, group in train.groupby("regime", sort=True):
        regime_key = str(regime)
        train_days = int(group["actual_date"].nunique())
        if train_days < min_group_days:
            rows.append(
                {
                    "scope": "regime",
                    "group": regime_key,
                    "objective": objective,
                    "weight_challenger": fallback_weight,
                    "train_score": float("nan"),
                    "train_days": train_days,
                    "selected": False,
                }
            )
            continue
        for weight in weights:
            group_frame = frame_with_prediction(group, "candidate", weighted_prediction(group, weight))
            score = rank_objective_score(group_frame, objective, split="train")
            rows.append(
                {
                    "scope": "regime",
                    "group": regime_key,
                    "objective": objective,
                    "weight_challenger": weight,
                    "train_score": score,
                    "train_days": train_days,
                    "selected": False,
                }
            )
    search = pd.DataFrame(rows)
    selected: dict[str, float] = {}
    for group_key, group_search in search[np.isfinite(search["train_score"])].groupby("group", sort=True):
        best = group_search.sort_values(["train_score", "weight_challenger"], ascending=[False, True], kind="stable").iloc[0]
        selected[str(group_key)] = float(best["weight_challenger"])
        search.loc[
            (search["group"] == group_key)
            & (search["objective"] == objective)
            & (search["weight_challenger"] == selected[str(group_key)]),
            "selected",
        ] = True
    return selected, search


def apply_regime_weights(base: pd.DataFrame, weights_by_regime: dict[str, float], fallback_weight: float) -> np.ndarray:
    weights = base["regime"].astype(str).map(weights_by_regime).fillna(fallback_weight).to_numpy(dtype=float)
    anchor = base["prediction__anchor"].to_numpy(dtype=float)
    challenger = base["prediction__challenger"].to_numpy(dtype=float)
    return (1.0 - weights) * anchor + weights * challenger


def prediction_summary(candidate_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame in candidate_frames:
        candidate = str(frame["run_name"].iloc[0])
        for split, group in frame.groupby("split", sort=True):
            rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "n_obs": int(len(group)),
                    "rel_score": rel_score(group["error"], group["actual"]),
                    "directional_accuracy": float((np.sign(group["prediction"]) == np.sign(group["actual"])).mean()),
                    "error_q2": float(group["error"].quantile(0.2)),
                    "error_q8": float(group["error"].quantile(0.8)),
                }
            )
    return pd.DataFrame(rows)


def rank_summary(candidate_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame in candidate_frames:
        candidate = str(frame["run_name"].iloc[0])
        for split in sorted(frame["split"].unique()):
            daily = daily_rank_metrics(frame, split)
            returns = daily["top_bottom_return"].dropna().to_numpy(dtype=float)
            ic = daily["spearman_ic"].dropna().to_numpy(dtype=float)
            equity = np.cumprod(1.0 + returns) if len(returns) else np.asarray([], dtype=float)
            rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "days": int(daily["actual_date"].nunique()),
                    "mean_ic": float(ic.mean()) if len(ic) else float("nan"),
                    "ic_t_stat": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 1 and ic.std(ddof=1) > 0 else float("nan"),
                    "positive_ic_days": float((ic > 0.0).mean()) if len(ic) else float("nan"),
                    "top_bottom_equity": float(equity[-1]) if len(equity) else float("nan"),
                    "top_bottom_hit_rate": float((returns > 0.0).mean()) if len(returns) else float("nan"),
                    "top_bottom_max_drawdown": max_drawdown(equity),
                }
            )
    return pd.DataFrame(rows)


def summarize_year_stability(daily_quartile: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    val_years = windows[windows["scope"] == "val_year"].copy()
    val_daily = daily_quartile[daily_quartile["split"] == "val"].copy()
    for candidate, candidate_df in val_daily.groupby("run_name", sort=True):
        equities: list[float] = []
        for _, window in val_years.iterrows():
            segment = candidate_df[
                (candidate_df["actual_date"] >= pd.Timestamp(window["start_date"]))
                & (candidate_df["actual_date"] <= pd.Timestamp(window["end_date"]))
            ].copy()
            returns = segment["long_short_return"].to_numpy(dtype=float)
            equity = np.cumprod(1.0 + returns) if len(returns) else np.asarray([], dtype=float)
            equities.append(float(equity[-1]) if len(equity) else float("nan"))
        clean = np.asarray([value for value in equities if np.isfinite(value)], dtype=float)
        rows.append(
            {
                "candidate": candidate,
                "years": int(len(clean)),
                "avg_year_equity": float(clean.mean()) if len(clean) else float("nan"),
                "worst_year_equity": float(clean.min()) if len(clean) else float("nan"),
                "profitable_years": int(np.sum(clean > 1.0)) if len(clean) else 0,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    output_dir: Path,
    pred_summary: pd.DataFrame,
    rank_stats: pd.DataFrame,
    filter_summary: pd.DataFrame,
    stability: pd.DataFrame,
    selection: dict[str, object],
) -> None:
    val_pred = pred_summary[pred_summary["split"] == "val"].copy()
    val_rank = rank_stats[rank_stats["split"] == "val"].copy()
    val_trade = filter_summary[(filter_summary["split"] == "val") & (filter_summary["filter_name"] == "all_regimes")].copy()
    merged = (
        val_pred.merge(val_rank, on=["candidate", "split"], how="left")
        .merge(val_trade[["run_name", "final_equity", "hit_rate"]], left_on="candidate", right_on="run_name", how="left")
        .merge(stability, on="candidate", how="left")
    )

    lines = [
        "# Train-Selected Rank Router",
        "",
        "Scope: weights are selected on train rank objectives, then evaluated on validation. No test/out-sample data is used.",
        "",
        "## Selected Train Rules",
        "",
        f"- Global mean-IC weight: `{selection['global_mean_ic_weight']}`",
        f"- Global top-bottom-equity weight: `{selection['global_top_bottom_equity_weight']}`",
        f"- Regime mean-IC weights: `{selection['regime_mean_ic_weights']}`",
        f"- Regime top-bottom-equity weights: `{selection['regime_top_bottom_equity_weights']}`",
        "",
        "## Validation Rank/Trade",
        "",
        "| Candidate | rel_score | IC | t-stat | Top-bottom equity | Quartile equity | Worst year equity | Hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in merged.sort_values(["mean_ic", "top_bottom_equity"], ascending=[False, False], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {float(row['rel_score']):+.4f} | {float(row['mean_ic']):+.4f} | "
            f"{float(row['ic_t_stat']):+.2f} | {float(row['top_bottom_equity']):.3f} | "
            f"{float(row['final_equity']):.3f} | {float(row['worst_year_equity']):.3f} | "
            f"{float(row['hit_rate']):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Validation Prediction",
            "",
            "| Candidate | rel_score | Direction | Error q2/q8 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for _, row in merged.sort_values("rel_score", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {float(row['rel_score']):+.4f} | {float(row['directional_accuracy']):.1%} | "
            f"{float(row['error_q2']):+.4f} / {float(row['error_q8']):+.4f} |"
        )

    best_rank = merged.sort_values(["mean_ic", "top_bottom_equity"], ascending=[False, False], kind="stable").iloc[0]
    lines.extend(
        [
            "",
            "## Read",
            "",
            f"- Best validation rank candidate: `{best_rank['candidate']}` with IC `{float(best_rank['mean_ic']):+.4f}` and top-bottom equity `{float(best_rank['top_bottom_equity']):.3f}`.",
            "- A train-selected candidate must beat `sector19_down_up_anchor_else` to justify replacing the current simple router.",
        ]
    )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Path:
    if args.step <= 0.0 or args.step > 1.0:
        raise ValueError("--step must be in (0, 1].")
    weights = [round(value, 4) for value in np.arange(0.0, 1.0 + args.step / 2.0, args.step)]
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base(args.router_report)
    global_ic_weight, global_ic_search = choose_global_weight(base, weights, "mean_ic")
    global_tb_weight, global_tb_search = choose_global_weight(base, weights, "top_bottom_equity")
    regime_ic_weights, regime_ic_search = choose_regime_weights(
        base,
        weights,
        "mean_ic",
        args.min_group_days,
        global_ic_weight,
    )
    regime_tb_weights, regime_tb_search = choose_regime_weights(
        base,
        weights,
        "top_bottom_equity",
        args.min_group_days,
        global_tb_weight,
    )

    candidate_frames = [
        candidate_frame(base, "anchor"),
        candidate_frame(base, "challenger"),
        candidate_frame(base, "sector19_down_up_anchor_else"),
        candidate_frame(base, "avg_70_challenger"),
        frame_with_prediction(base, "train_rank_global_ic_weight", weighted_prediction(base, global_ic_weight)),
        frame_with_prediction(base, "train_rank_global_topbottom_weight", weighted_prediction(base, global_tb_weight)),
        frame_with_prediction(base, "train_rank_regime_ic_weight", apply_regime_weights(base, regime_ic_weights, global_ic_weight)),
        frame_with_prediction(base, "train_rank_regime_topbottom_weight", apply_regime_weights(base, regime_tb_weights, global_tb_weight)),
    ]
    pred_summary = prediction_summary(candidate_frames)
    rank_stats = rank_summary(candidate_frames)
    long_df = pd.concat(candidate_frames, ignore_index=True)
    daily_quartile = build_daily_quartile_returns(long_df)
    filter_summary = build_regime_filter_summary(daily_quartile)
    stability = summarize_year_stability(daily_quartile, build_windows(base))

    selection = {
        "global_mean_ic_weight": global_ic_weight,
        "global_top_bottom_equity_weight": global_tb_weight,
        "regime_mean_ic_weights": regime_ic_weights,
        "regime_top_bottom_equity_weights": regime_tb_weights,
    }

    pd.concat([global_ic_search, global_tb_search, regime_ic_search, regime_tb_search], ignore_index=True).to_csv(
        output_dir / "selection_search.csv",
        index=False,
    )
    pred_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    rank_stats.to_csv(output_dir / "rank_summary.csv", index=False)
    filter_summary.to_csv(output_dir / "regime_filter_summary.csv", index=False)
    stability.to_csv(output_dir / "year_stability.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "router_report": str(args.router_report),
                "step": args.step,
                "min_group_days": args.min_group_days,
                "selection": selection,
                "prediction_summary": pred_summary.to_dict(orient="records"),
                "rank_summary": rank_stats.to_dict(orient="records"),
                "year_stability": stability.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, pred_summary, rank_stats, filter_summary, stability, selection)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    output_dir = run_analysis(parse_args(argv))
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
