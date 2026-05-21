from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_regime_performance import (
    align_predictions,
    build_daily_quartile_returns,
    load_predictions,
    rel_score,
)


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "cycle_phase"


DEFAULT_RUNS = (
    "broad_signmag_prune_general_sector_full_20260424_r04",
    "broad_signmag_prune_no_fast_overlap_20260424_r01",
    "broad_signmag_prune_general_sector_breadth_20260424_r04",
    "broad_signmag_prune_compact_core12_20260424_r01",
)
DEFAULT_MODELS = ("lstm_signmag_best_by_val",)
DEFAULT_SPLITS = ("train", "val")
PHASE_ORDER = ("recovery", "uptrend", "distribution", "downtrend")


@dataclass(frozen=True)
class CyclePhaseConfig:
    trend_fast_window: int = 20
    trend_slow_window: int = 60
    breadth_up: float = 0.53
    breadth_down: float = 0.47
    min_episode_days: int = 10
    min_phase_days_for_cycle: int = 15


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a point-in-time cycle/phase report from existing VN model predictions."
    )
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="Comma-separated training run names.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model names.")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated splits.")
    parser.add_argument("--stamp", default="20260425_r01", help="Report stamp.")
    parser.add_argument("--output-name", default="current_best_cycle_phase", help="Report folder prefix.")
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_market_proxy(prediction_df: pd.DataFrame) -> pd.DataFrame:
    base_model = str(prediction_df["model"].iloc[0])
    base_df = (
        prediction_df[prediction_df["model"] == base_model]
        .drop_duplicates(["code", "Date"])
        .copy()
    )
    daily = (
        base_df.groupby("Date", as_index=False)
        .agg(
            market_return=("actual", "mean"),
            breadth=("actual", lambda values: float(np.mean(np.asarray(values, dtype=float) > 0.0))),
            stock_count=("code", "nunique"),
        )
        .sort_values("Date", kind="stable")
        .reset_index(drop=True)
    )
    daily["market_index"] = (1.0 + daily["market_return"].fillna(0.0)).cumprod()
    return daily


def _smooth_short_episodes(phases: pd.Series, min_days: int) -> pd.Series:
    smoothed = phases.copy()
    episode_id = (smoothed != smoothed.shift()).cumsum()
    episodes = (
        pd.DataFrame({"phase": smoothed, "episode_id": episode_id})
        .groupby("episode_id", sort=True)
        .agg(phase=("phase", "first"), days=("phase", "size"))
        .reset_index()
    )
    for idx, row in episodes.iterrows():
        if int(row["days"]) >= min_days or idx == 0:
            continue
        previous_phase = str(episodes.loc[idx - 1, "phase"])
        smoothed.loc[episode_id == row["episode_id"]] = previous_phase
    return smoothed


def assign_cycle_phases(daily: pd.DataFrame, config: CyclePhaseConfig) -> pd.DataFrame:
    df = daily.copy()
    df["market_return_20"] = df["market_return"].rolling(config.trend_fast_window, min_periods=10).mean()
    df["market_return_60"] = df["market_return"].rolling(config.trend_slow_window, min_periods=30).mean()
    df["breadth_20"] = df["breadth"].rolling(config.trend_fast_window, min_periods=10).mean()
    df["breadth_60"] = df["breadth"].rolling(config.trend_slow_window, min_periods=30).mean()
    df["breadth_20_delta"] = df["breadth_20"] - df["breadth_20"].shift(config.trend_fast_window)
    df["market_volatility_20"] = df["market_return"].rolling(config.trend_fast_window, min_periods=10).std()
    df["volatility_expanding_median"] = df["market_volatility_20"].expanding(min_periods=60).median()
    df["rolling_peak_60"] = df["market_index"].rolling(config.trend_slow_window, min_periods=30).max()
    df["rolling_trough_60"] = df["market_index"].rolling(config.trend_slow_window, min_periods=30).min()
    df["drawdown_60"] = df["market_index"] / df["rolling_peak_60"] - 1.0
    df["recovery_from_trough_60"] = df["market_index"] / df["rolling_trough_60"] - 1.0

    uptrend = (
        (df["market_return_20"] > 0.0)
        & (df["market_return_60"] > 0.0)
        & (df["breadth_20"] >= config.breadth_up)
    )
    downtrend = (
        (df["market_return_20"] < 0.0)
        & (df["market_return_60"] < 0.0)
        & (df["breadth_20"] <= config.breadth_down)
    )
    distribution = (
        (df["market_return_60"] >= -0.0005)
        & (df["breadth_20"] <= config.breadth_down)
        & (
            (df["market_volatility_20"] >= df["volatility_expanding_median"])
            | (df["market_return_20"] < 0.0)
        )
    )
    recovery = (
        (df["market_return_20"] > 0.0)
        & (df["breadth_20_delta"] > 0.0)
        & (df["drawdown_60"] < -0.03)
        & ((df["market_return_60"] <= 0.0) | (df["recovery_from_trough_60"] > 0.05))
    )

    phase = np.full(len(df), "transition", dtype=object)
    phase[uptrend] = "uptrend"
    phase[distribution] = "distribution"
    phase[recovery] = "recovery"
    phase[downtrend] = "downtrend"
    df["phase_raw"] = phase
    df["phase"] = _smooth_short_episodes(pd.Series(phase, index=df.index), config.min_episode_days)
    df["episode_id"] = (df["phase"] != df["phase"].shift()).cumsum()
    df["phase_age"] = df.groupby("episode_id").cumcount() + 1
    return df


def build_phase_episodes(phase_df: pd.DataFrame) -> pd.DataFrame:
    episodes = (
        phase_df.groupby("episode_id", as_index=False)
        .agg(
            phase=("phase", "first"),
            start_date=("Date", "min"),
            end_date=("Date", "max"),
            days=("Date", "size"),
            market_return=("market_return", lambda values: float(np.prod(1.0 + np.asarray(values, dtype=float)) - 1.0)),
            avg_breadth=("breadth", "mean"),
            avg_volatility_20=("market_volatility_20", "mean"),
            start_index=("market_index", "first"),
            end_index=("market_index", "last"),
        )
        .sort_values("start_date", kind="stable")
        .reset_index(drop=True)
    )
    episodes["phase_return"] = episodes["end_index"] / episodes["start_index"] - 1.0
    return episodes


def build_cycle_summary(episodes: pd.DataFrame, config: CyclePhaseConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_rows: list[dict[str, object]] = []
    episode_rows = episodes.copy()
    episode_rows["cycle_id"] = np.nan
    cycle_id = 0
    order_index = 0
    active_rows: list[tuple[int, pd.Series]] = []
    active_phase_days = {phase: 0 for phase in PHASE_ORDER}

    for idx, row in episodes.iterrows():
        phase = str(row["phase"])
        if phase not in PHASE_ORDER:
            if active_rows:
                active_rows.append((idx, row))
            continue

        expected_phase = PHASE_ORDER[order_index]
        if phase == "recovery" and order_index != 0:
            active_rows = []
            active_phase_days = {phase_name: 0 for phase_name in PHASE_ORDER}
            order_index = 0
            expected_phase = PHASE_ORDER[order_index]

        if phase == expected_phase:
            active_rows.append((idx, row))
            active_phase_days[phase] += int(row["days"])
            if order_index < len(PHASE_ORDER) - 1:
                order_index += 1
        elif active_rows and phase in PHASE_ORDER[: order_index + 1]:
            active_rows.append((idx, row))
            active_phase_days[phase] += int(row["days"])
        elif phase == "recovery":
            active_rows = [(idx, row)]
            active_phase_days = {phase_name: 0 for phase_name in PHASE_ORDER}
            active_phase_days["recovery"] = int(row["days"])
            order_index = 1

        if active_rows and all(active_phase_days[phase_name] > 0 for phase_name in PHASE_ORDER):
            enough_days = all(active_phase_days[phase_name] >= config.min_phase_days_for_cycle for phase_name in PHASE_ORDER)
            cycle_id += 1
            active_indices = [item[0] for item in active_rows]
            active_episode_df = pd.DataFrame([item[1].to_dict() for item in active_rows])
            start_date = pd.Timestamp(active_episode_df["start_date"].min())
            end_date = pd.Timestamp(active_episode_df["end_date"].max())
            cycle_rows.append(
                {
                    "cycle_id": cycle_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "days": int((end_date - start_date).days) + 1,
                    "complete": bool(enough_days),
                    "phases_present": ",".join(PHASE_ORDER),
                    "recovery_days": int(active_phase_days["recovery"]),
                    "uptrend_days": int(active_phase_days["uptrend"]),
                    "distribution_days": int(active_phase_days["distribution"]),
                    "downtrend_days": int(active_phase_days["downtrend"]),
                }
            )
            episode_rows.loc[active_indices, "cycle_id"] = cycle_id
            active_rows = []
            active_phase_days = {phase_name: 0 for phase_name in PHASE_ORDER}
            order_index = 0

    cycle_df = pd.DataFrame(cycle_rows)
    return episode_rows, cycle_df


def attach_phase(aligned: pd.DataFrame, phase_df: pd.DataFrame) -> pd.DataFrame:
    lookup = phase_df[["Date", "phase", "phase_raw", "episode_id", "phase_age"]].rename(columns={"Date": "signal_date"})
    out = aligned.merge(lookup, on="signal_date", how="left")
    out["phase"] = out["phase"].fillna("unknown")
    out["phase_raw"] = out["phase_raw"].fillna("unknown")
    return out


def summarize_prediction_by_phase(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in aligned.groupby(["run_name", "model", "split", "phase"], sort=True):
        row = dict(zip(["run_name", "model", "split", "phase"], keys, strict=True))
        row.update(
            {
                "n_obs": int(len(group)),
                "n_days": int(group["actual_date"].nunique()),
                "n_stocks": int(group["code"].nunique()),
                "rel_score": rel_score(group["actual"] - group["prediction"], group["actual"]),
                "directional_accuracy": float((np.sign(group["prediction"]) == np.sign(group["actual"])).mean()),
                "error_q2": float((group["actual"] - group["prediction"]).quantile(0.2)),
                "error_q8": float((group["actual"] - group["prediction"]).quantile(0.8)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "run_name", "phase"], kind="stable")


def build_phase_filter_summary(daily_quartile: pd.DataFrame) -> pd.DataFrame:
    filters: dict[str, set[str] | None] = {
        "all_phases": None,
        "skip_downtrend": {"distribution", "recovery", "transition", "uptrend"},
        "trade_uptrend_distribution": {"distribution", "uptrend"},
        "trade_uptrend_recovery_distribution": {"distribution", "recovery", "uptrend"},
        "trade_distribution_only": {"distribution"},
        "trade_uptrend_only": {"uptrend"},
        "trade_non_transition": {"distribution", "downtrend", "recovery", "uptrend"},
    }
    rows: list[dict[str, object]] = []
    for keys, group in daily_quartile.groupby(["run_name", "model", "split"], sort=True):
        for filter_name, phases in filters.items():
            selected = group if phases is None else group[group["regime"].isin(phases)]
            returns = selected["long_short_return"].to_numpy(dtype=float)
            row = dict(zip(["run_name", "model", "split"], keys, strict=True))
            row.update(
                {
                    "filter_name": filter_name,
                    "phases": "ALL" if phases is None else ",".join(sorted(phases)),
                    "trade_days": int(len(selected)),
                    "final_equity": float(np.prod(1.0 + returns)) if len(returns) else float("nan"),
                    "mean_return": float(np.mean(returns)) if len(returns) else float("nan"),
                    "hit_rate": float(np.mean(returns > 0.0)) if len(returns) else float("nan"),
                    "avg_trade_count": float((selected["long_count"] + selected["short_count"]).mean()) if len(selected) else float("nan"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "run_name", "final_equity"], ascending=[True, True, False], kind="stable")


def save_phase_timeline_plot(phase_df: pd.DataFrame, episodes: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "recovery": "#2ca02c",
        "uptrend": "#1f77b4",
        "distribution": "#ff7f0e",
        "downtrend": "#d62728",
        "transition": "#9e9e9e",
        "unknown": "#f0f0f0",
    }
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, height_ratios=[2.0, 1.2, 1.2])
    axes[0].plot(phase_df["Date"], phase_df["market_index"], color="#111111", linewidth=1.4)
    axes[0].set_title("Equal-weight VN market proxy with cycle phases")
    axes[0].set_ylabel("Market proxy")
    axes[0].grid(True, alpha=0.2)
    for _, episode in episodes.iterrows():
        axes[0].axvspan(
            pd.Timestamp(episode["start_date"]),
            pd.Timestamp(episode["end_date"]),
            color=colors.get(str(episode["phase"]), "#cccccc"),
            alpha=0.16,
            linewidth=0,
        )

    axes[1].plot(phase_df["Date"], phase_df["breadth_20"], color="#1f77b4", linewidth=1.2, label="breadth_20")
    axes[1].axhline(0.53, color="#2ca02c", linestyle="--", linewidth=0.9, alpha=0.8)
    axes[1].axhline(0.47, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.8)
    axes[1].set_ylabel("Breadth")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(phase_df["Date"], phase_df["market_return_20"], color="#2ca02c", linewidth=1.1, label="ret_20")
    axes[2].plot(phase_df["Date"], phase_df["market_return_60"], color="#9467bd", linewidth=1.1, label="ret_60")
    axes[2].axhline(0.0, color="#111111", linewidth=0.8, alpha=0.6)
    axes[2].set_ylabel("Rolling return")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.2)
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "cycle_phase_timeline.png", dpi=180)
    plt.close(fig)


def write_markdown(
    output_dir: Path,
    episodes: pd.DataFrame,
    cycles: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    filter_summary: pd.DataFrame,
) -> None:
    val_prediction = prediction_summary[prediction_summary["split"] == "val"].copy()
    val_filters = filter_summary[filter_summary["split"] == "val"].copy()
    phase_counts = (
        episodes.groupby("phase", as_index=False)
        .agg(episodes=("episode_id", "count"), days=("days", "sum"), avg_days=("days", "mean"))
        .sort_values("days", ascending=False, kind="stable")
    )

    lines = [
        "# Cycle Phase Report",
        "",
        "Scope: existing train/validation predictions only. No test/out-sample data is used.",
        "",
        "The market proxy is equal-weight daily return from the run prediction universe; phase labels are point-in-time and use only rolling values available at the signal date.",
        "",
        "A complete cycle requires this ordered sequence with enough duration: `recovery -> uptrend -> distribution -> downtrend`.",
        "",
        "## Phase Coverage",
        "",
        "| Phase | Episodes | Days | Avg days |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in phase_counts.iterrows():
        lines.append(f"| `{row['phase']}` | {int(row['episodes'])} | {int(row['days'])} | {float(row['avg_days']):.1f} |")

    lines.extend(["", "## Complete Cycles", ""])
    if cycles.empty:
        lines.append("No complete ordered cycle was detected under the current thresholds.")
    else:
        lines.extend(
            [
                "| Cycle | Start | End | Days | Complete | Recovery | Uptrend | Distribution | Downtrend |",
                "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in cycles.iterrows():
            lines.append(
                "| "
                f"{int(row['cycle_id'])} | {pd.Timestamp(row['start_date']).date()} | {pd.Timestamp(row['end_date']).date()} | "
                f"{int(row['days'])} | {bool(row['complete'])} | {int(row['recovery_days'])} | "
                f"{int(row['uptrend_days'])} | {int(row['distribution_days'])} | {int(row['downtrend_days'])} |"
            )

    lines.extend(
        [
            "",
            "## Validation Prediction By Phase",
            "",
            "| Run | Phase | Obs | Days | rel_score | Direction | Error q2/q8 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in val_prediction.sort_values(["run_name", "rel_score"], ascending=[True, False], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['run_name']}` | `{row['phase']}` | {int(row['n_obs'])} | {int(row['n_days'])} | "
            f"{float(row['rel_score']):+.4f} | {float(row['directional_accuracy']):.1%} | "
            f"{float(row['error_q2']):+.4f} / {float(row['error_q8']):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Validation Trade Filters",
            "",
            "| Run | Filter | Trade days | Equity | Hit rate | Mean return |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in val_filters.sort_values(["run_name", "final_equity"], ascending=[True, False], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['run_name']}` | `{row['filter_name']}` | {int(row['trade_days'])} | "
            f"{float(row['final_equity']):.3f} | {float(row['hit_rate']):.1%} | {float(row['mean_return']):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use the phase report to decide when a full cycle has been observed before trusting phase-specific feature selection.",
            "- Downtrend can show positive prediction rel_score while still producing weak long-short trade equity, so phase-specific trade filters must be judged by both prediction and trade metrics.",
            "- Treat the thresholds as research defaults; changing them counts as a new hypothesis test.",
        ]
    )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_names = split_csv(args.runs)
    model_names = split_csv(args.models)
    splits = set(split_csv(args.splits))
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CyclePhaseConfig()

    anchor_predictions = load_predictions(run_names[0])
    phase_df = assign_cycle_phases(build_market_proxy(anchor_predictions), config)
    episodes, cycles = build_cycle_summary(build_phase_episodes(phase_df), config)

    all_aligned: list[pd.DataFrame] = []
    phase_lookup = phase_df[["Date", "phase"]].rename(columns={"Date": "signal_date", "phase": "regime"})
    for run_name in run_names:
        prediction_df = load_predictions(run_name)
        for model_name in model_names:
            aligned = align_predictions(prediction_df, run_name, model_name, splits)
            if aligned.empty:
                continue
            all_aligned.append(attach_phase(aligned, phase_df))

    if not all_aligned:
        raise RuntimeError("No aligned predictions were available for the requested runs/models/splits.")

    aligned_df = pd.concat(all_aligned, ignore_index=True)
    prediction_summary = summarize_prediction_by_phase(aligned_df)
    daily_quartile = build_daily_quartile_returns(
        aligned_df.merge(phase_lookup, on="signal_date", how="left").drop(columns=["regime_x"], errors="ignore").rename(columns={"regime_y": "regime"})
    )
    filter_summary = build_phase_filter_summary(daily_quartile)

    phase_df.to_csv(output_dir / "daily_cycle_phases.csv", index=False)
    episodes.to_csv(output_dir / "phase_episodes.csv", index=False)
    cycles.to_csv(output_dir / "cycle_summary.csv", index=False)
    aligned_df.to_csv(output_dir / "aligned_predictions_with_phase.csv", index=False)
    prediction_summary.to_csv(output_dir / "prediction_by_phase.csv", index=False)
    daily_quartile.to_csv(output_dir / "daily_quartile_returns_by_phase.csv", index=False)
    filter_summary.to_csv(output_dir / "phase_filter_summary.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "runs": run_names,
                "models": model_names,
                "splits": sorted(splits),
                "config": config.__dict__,
                "cycles": cycles.to_dict(orient="records"),
                "prediction_by_phase": prediction_summary.to_dict(orient="records"),
                "phase_filter_summary": filter_summary.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    save_phase_timeline_plot(phase_df, episodes, output_dir)
    write_markdown(output_dir, episodes, cycles, prediction_summary, filter_summary)
    print(json.dumps({"output_dir": str(output_dir), "cycles": len(cycles)}, indent=2))


if __name__ == "__main__":
    main()
