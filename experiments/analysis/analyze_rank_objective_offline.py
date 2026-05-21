from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_prediction_router import add_candidate_predictions  # noqa: E402
from experiments.analysis.analyze_router_rolling_validation import DEFAULT_ROUTER_REPORT  # noqa: E402

RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "rank_objective_offline"
DEFAULT_CANDIDATES = (
    "anchor",
    "challenger",
    "avg_10_challenger",
    "avg_70_challenger",
    "sector19_down_anchor_else",
    "sector19_up_anchor_else",
    "sector19_down_up_anchor_else",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate offline cross-sectional rank objectives on existing train/validation predictions."
    )
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--stamp", default="20260427_r01")
    parser.add_argument("--output-name", default="anchor_sector19_rank_objective")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--splits", default="train,val")
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_candidate_frame(router_report: Path) -> pd.DataFrame:
    df = pd.read_csv(router_report / "candidate_predictions.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["actual_date"] = pd.to_datetime(df["actual_date"])
    if not any(column.startswith("candidate__") for column in df.columns):
        df = add_candidate_predictions(df)
    return df


def build_long_candidates(df: pd.DataFrame, candidate_names: list[str], splits: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    selected = df[df["split"].isin(splits)].copy()
    for candidate in candidate_names:
        column = f"candidate__{candidate}"
        if column not in selected.columns:
            continue
        part = selected[["code", "split", "signal_date", "actual_date", "actual", "regime", column]].copy()
        part = part.rename(columns={column: "prediction"})
        part["candidate"] = candidate
        frames.append(part)
    if not frames:
        raise RuntimeError("No candidate prediction columns matched the requested candidates.")
    return pd.concat(frames, ignore_index=True)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = values.std()
    if not np.isfinite(scale) or scale < 1e-8:
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / scale


def compute_daily_rank_metrics(group: pd.DataFrame) -> dict[str, float]:
    actual = group["actual"].to_numpy(dtype=float)
    prediction = group["prediction"].to_numpy(dtype=float)
    if len(actual) < 5 or np.unique(actual).size < 2 or np.unique(prediction).size < 2:
        return {
            "spearman_ic": float("nan"),
            "pairwise_rank_loss": float("nan"),
            "market_neutral_return": float("nan"),
            "top_bottom_return": float("nan"),
        }

    spearman_ic = float(spearmanr(prediction, actual).correlation)

    actual_diff = actual[:, None] - actual[None, :]
    score_diff = zscore(prediction)[:, None] - zscore(prediction)[None, :]
    mask = actual_diff > 0.0
    pairwise_rank_loss = float("nan")
    if np.any(mask):
        pairwise_rank_loss = float(-np.log(np.clip(expit(score_diff[mask]), 1e-8, 1.0)).mean())

    centered_score = zscore(prediction)
    denominator = np.abs(centered_score).sum()
    market_neutral_return = float(np.sum((centered_score / denominator) * actual)) if denominator > 1e-8 else 0.0

    ranks = pd.Series(prediction).rank(method="first", pct=True).to_numpy()
    top = actual[ranks >= 0.75]
    bottom = actual[ranks <= 0.25]
    top_bottom_return = float(top.mean() - bottom.mean()) if len(top) and len(bottom) else float("nan")

    return {
        "spearman_ic": spearman_ic,
        "pairwise_rank_loss": pairwise_rank_loss,
        "market_neutral_return": market_neutral_return,
        "top_bottom_return": top_bottom_return,
    }


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return float(drawdown.min())


def summarize_daily_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, split), group in daily.groupby(["candidate", "split"], sort=True):
        returns = group["top_bottom_return"].dropna().to_numpy(dtype=float)
        mn_returns = group["market_neutral_return"].dropna().to_numpy(dtype=float)
        ic = group["spearman_ic"].dropna().to_numpy(dtype=float)
        pairwise = group["pairwise_rank_loss"].dropna().to_numpy(dtype=float)
        equity = np.cumprod(1.0 + returns) if len(returns) else np.asarray([], dtype=float)
        rows.append(
            {
                "candidate": candidate,
                "split": split,
                "days": int(group["actual_date"].nunique()),
                "mean_spearman_ic": float(ic.mean()) if len(ic) else float("nan"),
                "ic_t_stat": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 1 and ic.std(ddof=1) > 0 else float("nan"),
                "positive_ic_days": float((ic > 0.0).mean()) if len(ic) else float("nan"),
                "mean_pairwise_rank_loss": float(pairwise.mean()) if len(pairwise) else float("nan"),
                "mean_market_neutral_return": float(mn_returns.mean()) if len(mn_returns) else float("nan"),
                "mean_top_bottom_return": float(returns.mean()) if len(returns) else float("nan"),
                "top_bottom_equity": float(equity[-1]) if len(equity) else float("nan"),
                "top_bottom_hit_rate": float((returns > 0.0).mean()) if len(returns) else float("nan"),
                "top_bottom_max_drawdown": max_drawdown(equity),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "mean_spearman_ic"], ascending=[True, False], kind="stable")


def write_markdown(output_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Offline Rank Objective Check",
        "",
        "Scope: existing train/validation predictions only. No test/out-sample data is used.",
        "",
        "This report checks whether the current router candidates have a rank edge that would justify a future rank/portfolio objective.",
        "",
    ]
    for split in ["train", "val"]:
        split_df = summary[summary["split"] == split].copy()
        if split_df.empty:
            continue
        lines.extend(
            [
                f"## {split.capitalize()} Summary",
                "",
                "| Candidate | Days | Spearman IC | t-stat | Positive IC days | Pairwise loss | Top-bottom equity | Hit rate | Max DD |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in split_df.sort_values("mean_spearman_ic", ascending=False, kind="stable").iterrows():
            lines.append(
                "| "
                f"`{row['candidate']}` | {int(row['days'])} | "
                f"{float(row['mean_spearman_ic']):+.4f} | {float(row['ic_t_stat']):+.2f} | "
                f"{float(row['positive_ic_days']):.1%} | {float(row['mean_pairwise_rank_loss']):.4f} | "
                f"{float(row['top_bottom_equity']):.3f} | {float(row['top_bottom_hit_rate']):.1%} | "
                f"{float(row['top_bottom_max_drawdown']):.1%} |"
            )
        lines.append("")

    val = summary[summary["split"] == "val"].copy()
    if not val.empty:
        best_ic = val.sort_values("mean_spearman_ic", ascending=False, kind="stable").iloc[0]
        best_equity = val.sort_values("top_bottom_equity", ascending=False, kind="stable").iloc[0]
        lines.extend(
            [
                "## Read",
                "",
                f"- Best validation IC: `{best_ic['candidate']}` with mean Spearman IC `{float(best_ic['mean_spearman_ic']):+.4f}`.",
                f"- Best validation top-bottom equity: `{best_equity['candidate']}` with equity `{float(best_equity['top_bottom_equity']):.3f}`.",
                "- If the best IC/equity candidate differs from the best rel_score candidate, that supports developing a rank/portfolio objective as a sidecar rather than replacing the prediction anchor.",
            ]
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Path:
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_names = split_csv(args.candidates)
    splits = set(split_csv(args.splits))
    candidates = load_candidate_frame(args.router_report)
    long_df = build_long_candidates(candidates, candidate_names, splits)

    daily_rows: list[dict[str, object]] = []
    for (candidate, split, actual_date), group in long_df.groupby(["candidate", "split", "actual_date"], sort=True):
        metrics = compute_daily_rank_metrics(group)
        daily_rows.append({"candidate": candidate, "split": split, "actual_date": actual_date, **metrics})
    daily = pd.DataFrame(daily_rows)
    summary = summarize_daily_metrics(daily)

    daily.to_csv(output_dir / "daily_rank_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "router_report": str(args.router_report),
                "candidates": candidate_names,
                "splits": sorted(splits),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, summary)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    output_dir = run_analysis(parse_args(argv))
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
