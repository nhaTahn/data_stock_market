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
from experiments.analysis.analyze_router_rolling_validation import (  # noqa: E402
    build_windows,
    max_drawdown,
)
from experiments.packaging.build_current_vn_all_stock_hist_report import (  # noqa: E402
    DEFAULT_RANK_ROUTER_REPORT,
    DEFAULT_ROUTER_REPORT,
    add_train_selected_rank_router_candidates,
    load_rank_router_selection,
    load_router_predictions,
)

RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "stock_reliability_filter"
DEFAULT_CANDIDATES = (
    "anchor",
    "sector19_down_up_anchor_else",
    "train_rank_regime_ic_weight",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select reliable stocks on train only, then validate prediction/rank/trade stability."
    )
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--rank-router-report", type=Path, default=DEFAULT_RANK_ROUTER_REPORT)
    parser.add_argument("--stamp", default="20260427_r01")
    parser.add_argument("--output-name", default="anchor_sector19_stock_reliability")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_base(router_report: Path, rank_router_report: Path) -> pd.DataFrame:
    frame = load_router_predictions(router_report)
    selection = load_rank_router_selection(rank_router_report)
    return add_train_selected_rank_router_candidates(frame, selection)


def frame_with_candidate(base: pd.DataFrame, candidate: str, candidate_label: str, allowed_codes: set[str]) -> pd.DataFrame:
    column = f"candidate__{candidate}"
    if column not in base.columns:
        raise ValueError(f"Missing candidate column: {column}")
    selected = base[base["code"].astype(str).isin(allowed_codes)].copy()
    out = selected[["code", "split", "signal_date", "actual_date", "actual", "regime"]].copy()
    out["prediction"] = selected[column].to_numpy(dtype=float)
    out["model"] = candidate_label
    out["run_name"] = candidate_label
    out["error"] = out["actual"] - out["prediction"]
    return out


def candidate_code_metrics(base: pd.DataFrame, candidate: str, split: str) -> pd.DataFrame:
    column = f"candidate__{candidate}"
    rows: list[dict[str, object]] = []
    selected = base[base["split"] == split].copy()
    for code, group in selected.groupby("code", sort=True):
        prediction = group[column].to_numpy(dtype=float)
        actual = group["actual"].to_numpy(dtype=float)
        error = actual - prediction
        rows.append(
            {
                "candidate": candidate,
                "split": split,
                "code": str(code),
                "row_count": int(len(group)),
                "rel_score": rel_score(pd.Series(error), pd.Series(actual)),
                "directional_accuracy": float((np.sign(prediction) == np.sign(actual)).mean()),
                "error_q2": float(np.quantile(prediction - actual, 0.2)),
                "error_q8": float(np.quantile(prediction - actual, 0.8)),
                "error_band_width": float(np.quantile(prediction - actual, 0.8) - np.quantile(prediction - actual, 0.2)),
                "bias_mean": float(np.mean(prediction - actual)),
            }
        )
    return pd.DataFrame(rows)


def select_codes(train_metrics: pd.DataFrame, rule: str) -> set[str]:
    if rule == "all_stocks":
        return set(train_metrics["code"].astype(str))
    if rule == "rel_positive":
        selected = train_metrics[train_metrics["rel_score"] > 0.0]
    elif rule == "drop_bottom25_rel":
        threshold = float(train_metrics["rel_score"].quantile(0.25))
        selected = train_metrics[train_metrics["rel_score"] > threshold]
    elif rule == "drop_bottom50_rel":
        threshold = float(train_metrics["rel_score"].quantile(0.50))
        selected = train_metrics[train_metrics["rel_score"] > threshold]
    elif rule == "rel_positive_band75":
        band_threshold = float(train_metrics["error_band_width"].quantile(0.75))
        selected = train_metrics[(train_metrics["rel_score"] > 0.0) & (train_metrics["error_band_width"] <= band_threshold)]
    else:
        raise ValueError(f"Unsupported reliability rule: {rule}")
    return set(selected["code"].astype(str))


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
            for actual_date, group in selected.groupby("actual_date", sort=True):
                daily_rows.append({"actual_date": actual_date, **compute_daily_rank_metrics(group)})
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
        "# Stock Reliability Filter",
        "",
        "Scope: stock lists are selected on train only, then evaluated on validation. No test/out-sample data is used.",
        "",
        "Rules:",
        "",
        "- `all_stocks`: no reliability filter",
        "- `rel_positive`: keep stocks with train per-stock rel_score > 0",
        "- `drop_bottom25_rel`: drop the bottom 25% stocks by train per-stock rel_score",
        "- `drop_bottom50_rel`: keep only the top 50% stocks by train per-stock rel_score",
        "- `rel_positive_band75`: keep rel-positive stocks whose train error-band width is not in the worst quartile",
        "",
        "## Validation Ranking",
        "",
        "| Candidate rule | Codes | rel_score | IC | Top-bottom equity | Quartile equity | Worst year equity | Hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in merged.sort_values(["worst_year_equity", "final_equity"], ascending=[False, False], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['candidate_rule']}` | {int(row['n_codes'])} | {float(row['rel_score']):+.4f} | "
            f"{float(row['mean_ic']):+.4f} | {float(row['top_bottom_equity']):.3f} | "
            f"{float(row['final_equity']):.3f} | {float(row['worst_year_equity']):.3f} | "
            f"{float(row['hit_rate']):.1%} |"
        )

    lines.extend(["", "## Selected Codes", ""])
    for _, row in selection.iterrows():
        lines.append(
            f"- `{row['candidate_rule']}`: {int(row['n_codes'])} codes: `{row['codes']}`"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Path:
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base(args.router_report, args.rank_router_report)
    candidates = split_csv(args.candidates)
    rules = ["all_stocks", "rel_positive", "drop_bottom25_rel", "drop_bottom50_rel", "rel_positive_band75"]

    code_metric_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    for candidate in candidates:
        train_metrics = candidate_code_metrics(base, candidate, "train")
        val_metrics = candidate_code_metrics(base, candidate, "val")
        code_metric_frames.extend([train_metrics, val_metrics])
        for rule in rules:
            allowed_codes = select_codes(train_metrics, rule)
            if len(allowed_codes) < 8:
                continue
            candidate_rule = f"{candidate}__{rule}"
            candidate_frames.append(frame_with_candidate(base, candidate, candidate_rule, allowed_codes))
            selection_rows.append(
                {
                    "candidate": candidate,
                    "rule": rule,
                    "candidate_rule": candidate_rule,
                    "n_codes": int(len(allowed_codes)),
                    "codes": ",".join(sorted(allowed_codes)),
                }
            )

    selection = pd.DataFrame(selection_rows)
    code_metrics = pd.concat(code_metric_frames, ignore_index=True)
    pred_summary = prediction_summary(candidate_frames)
    rank_stats = rank_summary(candidate_frames)
    long_df = pd.concat(candidate_frames, ignore_index=True)
    daily_quartile = build_daily_quartile_returns(long_df)
    filter_summary = build_regime_filter_summary(daily_quartile)
    stability = summarize_year_stability(daily_quartile, build_windows(base))

    selection.to_csv(output_dir / "selection.csv", index=False)
    code_metrics.to_csv(output_dir / "code_metrics.csv", index=False)
    pred_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    rank_stats.to_csv(output_dir / "rank_summary.csv", index=False)
    filter_summary.to_csv(output_dir / "regime_filter_summary.csv", index=False)
    stability.to_csv(output_dir / "year_stability.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "router_report": str(args.router_report),
                "rank_router_report": str(args.rank_router_report),
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
