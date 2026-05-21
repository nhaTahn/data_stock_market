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

from experiments.analysis.analyze_rank_objective_offline import compute_daily_rank_metrics  # noqa: E402
from experiments.analysis.analyze_regime_performance import (  # noqa: E402
    build_daily_quartile_returns,
    build_regime_filter_summary,
    rel_score,
)
from experiments.analysis.analyze_router_rolling_validation import build_windows, max_drawdown  # noqa: E402
from experiments.packaging.build_current_vn_all_stock_hist_report import (  # noqa: E402
    DEFAULT_RANK_ROUTER_REPORT,
    DEFAULT_ROUTER_REPORT,
    add_train_selected_rank_router_candidates,
    load_rank_router_selection,
    load_router_predictions,
)

RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "stock_bias_calibration"
DEFAULT_CANDIDATES = (
    "anchor",
    "sector19_down_up_anchor_else",
    "train_rank_regime_ic_weight",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only stock-bias calibration for current VN router candidates."
    )
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--rank-router-report", type=Path, default=DEFAULT_RANK_ROUTER_REPORT)
    parser.add_argument("--stamp", default="20260427_r01")
    parser.add_argument("--output-name", default="anchor_sector19_stock_bias_calibration")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--shrink-grid", default="0,0.25,0.5,0.75,1.0")
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_base(router_report: Path, rank_router_report: Path) -> pd.DataFrame:
    frame = load_router_predictions(router_report)
    selection = load_rank_router_selection(rank_router_report)
    return add_train_selected_rank_router_candidates(frame, selection)


def compute_stock_bias(base: pd.DataFrame, candidate: str) -> pd.Series:
    column = f"candidate__{candidate}"
    train = base[base["split"] == "train"].copy()
    error = train[column].to_numpy(dtype=float) - train["actual"].to_numpy(dtype=float)
    return pd.Series(error, index=train.index).groupby(train["code"].astype(str)).mean()


def calibrated_prediction(base: pd.DataFrame, candidate: str, bias_by_code: pd.Series, shrink: float) -> np.ndarray:
    column = f"candidate__{candidate}"
    bias = base["code"].astype(str).map(bias_by_code).fillna(0.0).to_numpy(dtype=float)
    return base[column].to_numpy(dtype=float) - shrink * bias


def frame_with_prediction(base: pd.DataFrame, candidate_rule: str, prediction: np.ndarray) -> pd.DataFrame:
    out = base[["code", "split", "signal_date", "actual_date", "actual", "regime"]].copy()
    out["prediction"] = prediction
    out["model"] = candidate_rule
    out["run_name"] = candidate_rule
    out["error"] = out["actual"] - out["prediction"]
    return out


def prediction_score(frame: pd.DataFrame, split: str = "train") -> float:
    group = frame[frame["split"] == split].copy()
    return rel_score(group["error"], group["actual"])


def rank_score(frame: pd.DataFrame, split: str = "train") -> float:
    selected = frame[frame["split"] == split].copy()
    rows: list[dict[str, object]] = []
    for _, group in selected.groupby("actual_date", sort=True):
        rows.append(compute_daily_rank_metrics(group))
    daily = pd.DataFrame(rows)
    return float(daily["spearman_ic"].mean()) if not daily.empty else float("nan")


def choose_shrink(
    base: pd.DataFrame,
    candidate: str,
    bias_by_code: pd.Series,
    shrink_grid: list[float],
    objective: str,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for shrink in shrink_grid:
        frame = frame_with_prediction(
            base,
            "candidate",
            calibrated_prediction(base, candidate, bias_by_code, shrink),
        )
        if objective == "rel_score":
            score = prediction_score(frame, "train")
        elif objective == "mean_ic":
            score = rank_score(frame, "train")
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        rows.append({"candidate": candidate, "objective": objective, "shrink": shrink, "train_score": score})
    search = pd.DataFrame(rows)
    best = search.sort_values(["train_score", "shrink"], ascending=[False, True], kind="stable").iloc[0]
    return float(best["shrink"]), search


def prediction_summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame in frames:
        candidate = str(frame["run_name"].iloc[0])
        for split, group in frame.groupby("split", sort=True):
            rows.append(
                {
                    "candidate_rule": candidate,
                    "split": split,
                    "n_obs": int(len(group)),
                    "n_codes": int(group["code"].astype(str).nunique()),
                    "rel_score": rel_score(group["error"], group["actual"]),
                    "directional_accuracy": float((np.sign(group["prediction"]) == np.sign(group["actual"])).mean()),
                    "error_q2": float((group["prediction"] - group["actual"]).quantile(0.2)),
                    "error_q8": float((group["prediction"] - group["actual"]).quantile(0.8)),
                    "bias_mean": float((group["prediction"] - group["actual"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def rank_summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame in frames:
        candidate = str(frame["run_name"].iloc[0])
        for split in sorted(frame["split"].unique()):
            selected = frame[frame["split"] == split].copy()
            daily_rows: list[dict[str, object]] = []
            for _, group in selected.groupby("actual_date", sort=True):
                daily_rows.append({"actual_date": group["actual_date"].iloc[0], **compute_daily_rank_metrics(group)})
            daily = pd.DataFrame(daily_rows)
            returns = daily["top_bottom_return"].dropna().to_numpy(dtype=float)
            ic = daily["spearman_ic"].dropna().to_numpy(dtype=float)
            equity = np.cumprod(1.0 + returns) if len(returns) else np.asarray([], dtype=float)
            rows.append(
                {
                    "candidate_rule": candidate,
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
                "candidate_rule": candidate,
                "years": int(len(clean)),
                "avg_year_equity": float(clean.mean()) if len(clean) else float("nan"),
                "worst_year_equity": float(clean.min()) if len(clean) else float("nan"),
                "profitable_years": int(np.sum(clean > 1.0)) if len(clean) else 0,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    output_dir: Path,
    selection: pd.DataFrame,
    pred_summary: pd.DataFrame,
    rank_stats: pd.DataFrame,
    filter_summary: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    val_pred = pred_summary[pred_summary["split"] == "val"].copy()
    val_rank = rank_stats[rank_stats["split"] == "val"].copy()
    val_trade = filter_summary[(filter_summary["split"] == "val") & (filter_summary["filter_name"] == "all_regimes")].copy()
    merged = (
        val_pred.merge(val_rank, on=["candidate_rule", "split"], how="left")
        .merge(val_trade[["run_name", "final_equity", "hit_rate"]], left_on="candidate_rule", right_on="run_name", how="left")
        .merge(stability, on="candidate_rule", how="left")
    )

    lines = [
        "# Stock Bias Calibration",
        "",
        "Scope: per-stock bias is estimated on train only, shrink is selected on train only, then evaluated on validation. No test/out-sample data is used.",
        "",
        "## Selected Shrink",
        "",
        "| Candidate | Objective | Shrink | Train score |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, row in selection.iterrows():
        lines.append(
            f"| `{row['candidate']}` | `{row['objective']}` | {float(row['shrink']):.2f} | {float(row['train_score']):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Validation Ranking",
            "",
            "| Candidate rule | rel_score | Bias mean | IC | Top-bottom equity | Quartile equity | Worst year equity | Hit rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in merged.sort_values(["rel_score", "worst_year_equity"], ascending=[False, False], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['candidate_rule']}` | {float(row['rel_score']):+.4f} | {float(row['bias_mean']):+.5f} | "
            f"{float(row['mean_ic']):+.4f} | {float(row['top_bottom_equity']):.3f} | "
            f"{float(row['final_equity']):.3f} | {float(row['worst_year_equity']):.3f} | "
            f"{float(row['hit_rate']):.1%} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Path:
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base(args.router_report, args.rank_router_report)
    candidates = split_csv(args.candidates)
    shrink_grid = parse_float_grid(args.shrink_grid)

    frames: list[pd.DataFrame] = []
    search_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    bias_rows: list[dict[str, object]] = []
    for candidate in candidates:
        bias = compute_stock_bias(base, candidate)
        for code, value in bias.items():
            bias_rows.append({"candidate": candidate, "code": code, "train_bias": float(value)})
        frames.append(frame_with_prediction(base, candidate, base[f"candidate__{candidate}"].to_numpy(dtype=float)))
        for objective in ["rel_score", "mean_ic"]:
            shrink, search = choose_shrink(base, candidate, bias, shrink_grid, objective)
            search_frames.append(search)
            best_score = float(search.loc[search["shrink"].eq(shrink), "train_score"].iloc[0])
            candidate_rule = f"{candidate}__bias_{objective}_s{str(shrink).replace('.', 'p')}"
            frames.append(frame_with_prediction(base, candidate_rule, calibrated_prediction(base, candidate, bias, shrink)))
            selection_rows.append(
                {
                    "candidate": candidate,
                    "objective": objective,
                    "shrink": shrink,
                    "train_score": best_score,
                    "candidate_rule": candidate_rule,
                }
            )

    selection = pd.DataFrame(selection_rows)
    search_df = pd.concat(search_frames, ignore_index=True)
    bias_df = pd.DataFrame(bias_rows)
    pred_summary = prediction_summary(frames)
    rank_stats = rank_summary(frames)
    long_df = pd.concat(frames, ignore_index=True)
    daily_quartile = build_daily_quartile_returns(long_df)
    filter_summary = build_regime_filter_summary(daily_quartile)
    stability = summarize_year_stability(daily_quartile, build_windows(base))

    selection.to_csv(output_dir / "selection.csv", index=False)
    search_df.to_csv(output_dir / "shrink_search.csv", index=False)
    bias_df.to_csv(output_dir / "stock_train_bias.csv", index=False)
    pred_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    rank_stats.to_csv(output_dir / "rank_summary.csv", index=False)
    filter_summary.to_csv(output_dir / "regime_filter_summary.csv", index=False)
    stability.to_csv(output_dir / "year_stability.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "router_report": str(args.router_report),
                "rank_router_report": str(args.rank_router_report),
                "shrink_grid": shrink_grid,
                "selection": selection.to_dict(orient="records"),
                "prediction_summary": pred_summary.to_dict(orient="records"),
                "rank_summary": rank_stats.to_dict(orient="records"),
                "year_stability": stability.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, selection, pred_summary, rank_stats, filter_summary, stability)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    output_dir = run_analysis(parse_args(argv))
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
