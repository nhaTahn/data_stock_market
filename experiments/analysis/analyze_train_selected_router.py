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
REPORT_ROOT = RUN_ROOT / "reports" / "router_train_selected"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-selected anchor/challenger routers and validate them.")
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--stamp", default="20260426_r01")
    parser.add_argument("--output-name", default="anchor_sector19_train_selected")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-group-obs", type=int, default=250)
    return parser.parse_args(argv)


def load_base(router_report: Path) -> pd.DataFrame:
    df = pd.read_csv(router_report / "candidate_predictions.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["actual_date"] = pd.to_datetime(df["actual_date"])
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


def evaluate_trade_equity(frame: pd.DataFrame, split: str = "train", regime: str | None = None) -> float:
    selected = frame[frame["split"] == split].copy()
    if regime is not None:
        selected = selected[selected["regime"] == regime].copy()
    if selected.empty:
        return float("nan")
    daily = build_daily_quartile_returns(selected)
    if daily.empty:
        return float("nan")
    returns = daily["long_short_return"].to_numpy(dtype=float)
    return float(np.prod(1.0 + returns)) if len(returns) else float("nan")


def choose_global_weight(base: pd.DataFrame, weights: list[float], objective: str) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for weight in weights:
        frame = frame_with_prediction(base, "candidate", weighted_prediction(base, weight))
        train = frame[frame["split"] == "train"]
        if objective == "rel_score":
            score = rel_score(train["error"], train["actual"])
        elif objective == "trade_equity":
            score = evaluate_trade_equity(frame, split="train")
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        rows.append({"scope": "global", "group": "ALL", "objective": objective, "weight_challenger": weight, "train_score": score})
    search = pd.DataFrame(rows)
    best = search.sort_values(["train_score", "weight_challenger"], ascending=[False, True], kind="stable").iloc[0]
    return float(best["weight_challenger"]), search


def choose_group_weights(base: pd.DataFrame, weights: list[float], group_col: str, objective: str, min_obs: int) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    train_base = base[base["split"] == "train"]
    for group_value, group in train_base.groupby(group_col, sort=True):
        group_key = str(group_value)
        if len(group) < min_obs:
            rows.append(
                {
                    "scope": group_col,
                    "group": group_key,
                    "objective": objective,
                    "weight_challenger": 0.0,
                    "train_score": float("nan"),
                    "n_obs": int(len(group)),
                    "selected": False,
                }
            )
            continue
        for weight in weights:
            group_frame = frame_with_prediction(group, "candidate", weighted_prediction(group, weight))
            if objective == "rel_score":
                score = rel_score(group_frame["error"], group_frame["actual"])
            elif objective == "trade_equity":
                score = evaluate_trade_equity(group_frame, split="train")
            else:
                raise ValueError(f"Unsupported objective: {objective}")
            rows.append(
                {
                    "scope": group_col,
                    "group": group_key,
                    "objective": objective,
                    "weight_challenger": weight,
                    "train_score": score,
                    "n_obs": int(len(group)),
                    "selected": False,
                }
            )
    search = pd.DataFrame(rows)
    selected: dict[str, float] = {}
    if search.empty:
        return selected, search
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


def apply_group_weights(base: pd.DataFrame, group_col: str, weights_by_group: dict[str, float], fallback_weight: float) -> np.ndarray:
    weights = base[group_col].astype(str).map(weights_by_group).fillna(fallback_weight).to_numpy(dtype=float)
    anchor = base["prediction__anchor"].to_numpy(dtype=float)
    challenger = base["prediction__challenger"].to_numpy(dtype=float)
    return (1.0 - weights) * anchor + weights * challenger


def fit_ridge_stack(base: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    train = base[base["split"] == "train"].copy()
    x_train = np.column_stack(
        [
            np.ones(len(train)),
            train["prediction__anchor"].to_numpy(dtype=float),
            train["prediction__challenger"].to_numpy(dtype=float),
        ]
    )
    y_train = train["actual"].to_numpy(dtype=float)
    penalty = np.eye(x_train.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + alpha * penalty, x_train.T @ y_train)

    x_all = np.column_stack(
        [
            np.ones(len(base)),
            base["prediction__anchor"].to_numpy(dtype=float),
            base["prediction__challenger"].to_numpy(dtype=float),
        ]
    )
    return x_all @ beta, beta


def summarize_year_stability(daily_quartile: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
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
                "years": int(len(clean_equities)),
                "avg_year_equity": float(np.mean(clean_equities)) if len(clean_equities) else float("nan"),
                "worst_year_equity": float(np.min(clean_equities)) if len(clean_equities) else float("nan"),
                "profitable_years": int(np.sum(clean_equities > 1.0)) if len(clean_equities) else 0,
                "avg_year_hit_rate": float(np.nanmean(hit_rates)),
                "worst_year_max_drawdown": float(np.nanmin(drawdowns)),
            }
        )
    return pd.DataFrame(rows)


def write_plot(output_dir: Path, summary: pd.DataFrame, filter_summary: pd.DataFrame) -> None:
    val_pred = summary[summary["split"] == "val"].copy()
    val_trade = filter_summary[(filter_summary["split"] == "val") & (filter_summary["filter_name"] == "all_regimes")].copy()
    merged = val_pred.merge(val_trade[["run_name", "final_equity"]], left_on="candidate", right_on="run_name", how="left")
    merged = merged.sort_values("final_equity", ascending=True, kind="stable")

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.barh(merged["candidate"], merged["final_equity"], color="#315c72")
    axis.set_xlabel("Validation quartile equity")
    axis.set_title("Train-Selected Router Candidates")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_trade_equity.png", dpi=160)
    plt.close(fig)


def write_markdown(
    output_dir: Path,
    summary: pd.DataFrame,
    filter_summary: pd.DataFrame,
    stability: pd.DataFrame,
    selection_summary: dict[str, object],
) -> None:
    val_pred = summary[summary["split"] == "val"].copy()
    val_trade = filter_summary[(filter_summary["split"] == "val") & (filter_summary["filter_name"] == "all_regimes")].copy()
    merged = (
        val_pred.merge(val_trade[["run_name", "final_equity", "hit_rate", "mean_return"]], left_on="candidate", right_on="run_name", how="left")
        .merge(stability, on="candidate", how="left")
        .sort_values("final_equity", ascending=False, kind="stable")
    )

    lines = [
        "# Train-Selected Router",
        "",
        "Scope: parameters are selected on train only, then evaluated on validation. No test/out-sample data is used.",
        "",
        "![Validation trade equity](validation_trade_equity.png)",
        "",
        "## Selected Rules",
        "",
        f"- Global rel_score weight: `{selection_summary['global_rel_weight']}`",
        f"- Global trade weight: `{selection_summary['global_trade_weight']}`",
        f"- Ridge beta `[intercept, anchor, challenger]`: `{selection_summary['ridge_beta']}`",
        f"- Regime rel_score weights: `{selection_summary['regime_rel_weights']}`",
        f"- Regime trade weights: `{selection_summary['regime_trade_weights']}`",
        "",
        "## Validation Ranking By Trade Equity",
        "",
        "| Candidate | rel_score | Equity | Hit rate | Worst year equity | Profitable years |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in merged.iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {float(row['rel_score']):+.4f} | {float(row['final_equity']):.3f} | "
            f"{float(row['hit_rate']):.1%} | {float(row['worst_year_equity']):.3f} | {int(row['profitable_years'])} |"
        )

    lines.extend(
        [
            "",
            "## Validation Ranking By rel_score",
            "",
            "| Candidate | rel_score | Equity | Direction | Error q2/q8 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in merged.sort_values("rel_score", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {float(row['rel_score']):+.4f} | {float(row['final_equity']):.3f} | "
            f"{float(row['directional_accuracy']):.1%} | {float(row['error_q2']):+.4f} / {float(row['error_q8']):+.4f} |"
        )

    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.step <= 0.0 or args.step > 1.0:
        raise ValueError("--step must be in (0, 1].")
    weights = [round(value, 4) for value in np.arange(0.0, 1.0 + args.step / 2.0, args.step)]
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base(args.router_report)
    global_rel_weight, global_rel_search = choose_global_weight(base, weights, "rel_score")
    global_trade_weight, global_trade_search = choose_global_weight(base, weights, "trade_equity")
    regime_rel_weights, regime_rel_search = choose_group_weights(base, weights, "regime", "rel_score", args.min_group_obs)
    regime_trade_weights, regime_trade_search = choose_group_weights(base, weights, "regime", "trade_equity", args.min_group_obs)
    stock_rel_weights, stock_rel_search = choose_group_weights(base, weights, "code", "rel_score", args.min_group_obs)
    ridge_prediction, ridge_beta = fit_ridge_stack(base, args.ridge_alpha)

    candidate_frames = [
        frame_with_prediction(base, "anchor", weighted_prediction(base, 0.0)),
        frame_with_prediction(base, "challenger", weighted_prediction(base, 1.0)),
        frame_with_prediction(base, "train_global_rel_weight", weighted_prediction(base, global_rel_weight)),
        frame_with_prediction(base, "train_global_trade_weight", weighted_prediction(base, global_trade_weight)),
        frame_with_prediction(base, "train_regime_rel_weight", apply_group_weights(base, "regime", regime_rel_weights, global_rel_weight)),
        frame_with_prediction(base, "train_regime_trade_weight", apply_group_weights(base, "regime", regime_trade_weights, global_trade_weight)),
        frame_with_prediction(base, "train_stock_rel_weight", apply_group_weights(base, "code", stock_rel_weights, global_rel_weight)),
        frame_with_prediction(base, "ridge_stack", ridge_prediction),
    ]
    summary = prediction_summary(candidate_frames)
    long_df = pd.concat(candidate_frames, ignore_index=True)
    daily_quartile = build_daily_quartile_returns(long_df)
    filter_summary = build_regime_filter_summary(daily_quartile)
    stability = summarize_year_stability(daily_quartile, build_windows(base))

    selection_summary = {
        "global_rel_weight": global_rel_weight,
        "global_trade_weight": global_trade_weight,
        "regime_rel_weights": regime_rel_weights,
        "regime_trade_weights": regime_trade_weights,
        "ridge_beta": [round(float(value), 8) for value in ridge_beta],
    }

    pd.concat(
        [global_rel_search, global_trade_search, regime_rel_search, regime_trade_search, stock_rel_search],
        ignore_index=True,
    ).to_csv(output_dir / "selection_search.csv", index=False)
    summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    filter_summary.to_csv(output_dir / "regime_filter_summary.csv", index=False)
    stability.to_csv(output_dir / "year_stability.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "router_report": str(args.router_report),
                "selection_summary": selection_summary,
                "prediction_summary": summary.to_dict(orient="records"),
                "year_stability": stability.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_plot(output_dir, summary, filter_summary)
    write_markdown(output_dir, summary, filter_summary, stability, selection_summary)
    print(json.dumps({"output_dir": str(output_dir), "selection_summary": selection_summary}, indent=2))


if __name__ == "__main__":
    main()
