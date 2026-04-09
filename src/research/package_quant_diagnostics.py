from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(ROOT))

from src.evaluation.metric import evaluate


DEFAULT_FRONTIER_SHORTLIST = (
    ROOT
    / "data/processed/assets/data_info_vn/history/training_runs/reports/research_restarts"
    / "fnb_phase5_frontier_selection/pair_frontier_shortlist.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a quant diagnostics pack for a standalone + committee candidate.")
    parser.add_argument("--standalone-run", type=Path, required=True)
    parser.add_argument("--standalone-model", default="lstm_best_by_val")
    parser.add_argument("--committee-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="Quant diagnostics")
    parser.add_argument("--frontier-shortlist", type=Path, default=DEFAULT_FRONTIER_SHORTLIST)
    parser.add_argument("--rolling-window", type=int, default=20)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_standalone_predictions(run_dir: Path, model_name: str) -> pd.DataFrame:
    path = run_dir / "reports" / "core" / "predictions.csv"
    df = pd.read_csv(path)
    df = df[(df["model"] == model_name) & (df["split"] == "test")].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "code", "actual", "prediction"]].sort_values(["Date", "code"], kind="stable")


def load_committee_predictions(committee_dir: Path) -> tuple[pd.DataFrame, dict]:
    summary = load_json(committee_dir / "best_committee_summary.json")
    df = pd.read_csv(committee_dir / "best_committee_predictions.csv")
    df = df[df["split"] == "test"].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    out = df.rename(columns={"prediction_committee": "prediction"})[["Date", "code", "actual", "prediction"]]
    return out.sort_values(["Date", "code"], kind="stable"), summary


def evaluate_frame(df: pd.DataFrame) -> dict[str, float]:
    ranked = df.sort_values(["code", "Date"], kind="stable")
    result = evaluate(
        ranked["prediction"].to_numpy(dtype=np.float32),
        ranked["actual"].to_numpy(dtype=np.float32),
        group_ids=ranked["code"].to_numpy(),
    )
    keys = ["base_loss", "abs_loss", "rel_score", "directional_accuracy"]
    return {k: float(result[k]) for k in keys}


def build_daily_ic(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, date_df in df.groupby("Date", sort=True):
        if date_df["code"].nunique() < 3:
            ic = np.nan
            ic_pearson = np.nan
        else:
            ic = date_df["prediction"].corr(date_df["actual"], method="spearman")
            ic_pearson = date_df["prediction"].corr(date_df["actual"], method="pearson")
        rows.append(
            {
                "Date": date,
                "model_label": model_label,
                "ic_spearman": ic,
                "ic_pearson": ic_pearson,
                "cross_sectional_actual_mean": float(date_df["actual"].mean()),
                "cross_sectional_prediction_mean": float(date_df["prediction"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("Date", kind="stable")
    return out


def add_rolling_stats(ic_df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = ic_df.copy()
    out["rolling_ic_spearman"] = out["ic_spearman"].rolling(window, min_periods=max(5, window // 4)).mean()
    out["rolling_ic_pearson"] = out["ic_pearson"].rolling(window, min_periods=max(5, window // 4)).mean()
    return out


def build_bucket_summary(df: pd.DataFrame, model_label: str, bucket_count: int = 10) -> pd.DataFrame:
    ranked = df.copy()
    ranked["bucket"] = pd.qcut(
        ranked["prediction"].rank(method="first"),
        q=bucket_count,
        labels=False,
        duplicates="drop",
    )
    summary = (
        ranked.groupby("bucket", as_index=False)
        .agg(
            prediction_mean=("prediction", "mean"),
            actual_mean=("actual", "mean"),
            actual_hit_rate=("actual", lambda x: float((x >= 0.0).mean())),
            count=("actual", "size"),
        )
        .sort_values("bucket")
    )
    summary["model_label"] = model_label
    return summary


def build_market_regimes(reference_df: pd.DataFrame, window: int) -> pd.DataFrame:
    daily = (
        reference_df.groupby("Date", as_index=False)["actual"]
        .mean()
        .rename(columns={"actual": "market_actual_mean"})
        .sort_values("Date", kind="stable")
    )
    daily["trend_5"] = daily["market_actual_mean"].rolling(5, min_periods=5).mean()
    daily["vol_20"] = daily["market_actual_mean"].rolling(window, min_periods=max(5, window // 2)).std()
    vol_threshold = float(daily["vol_20"].median(skipna=True))
    daily["trend_regime"] = np.where(daily["trend_5"] >= 0.0, "up", "down")
    daily["vol_regime"] = np.where(daily["vol_20"] >= vol_threshold, "high_vol", "low_vol")
    daily["regime"] = daily["trend_regime"] + "__" + daily["vol_regime"]
    return daily


def build_regime_summary(df: pd.DataFrame, regimes: pd.DataFrame, model_label: str) -> pd.DataFrame:
    merged = df.merge(regimes[["Date", "regime", "trend_regime", "vol_regime"]], on="Date", how="left")
    rows: list[dict[str, object]] = []
    for regime, regime_df in merged.groupby("regime", sort=True):
        if regime_df.empty:
            continue
        metrics = evaluate_frame(regime_df)
        rows.append(
            {
                "model_label": model_label,
                "regime": regime,
                "trend_regime": regime_df["trend_regime"].iloc[0],
                "vol_regime": regime_df["vol_regime"].iloc[0],
                "rows": int(len(regime_df)),
                "code_count": int(regime_df["code"].nunique()),
                "rel_score": metrics["rel_score"],
                "abs_loss": metrics["abs_loss"],
                "base_loss": metrics["base_loss"],
                "pred_pos_rate": float((regime_df["prediction"] > 0.0).mean()),
                "actual_pos_rate": float((regime_df["actual"] > 0.0).mean()),
                "avg_actual": float(regime_df["actual"].mean()),
                "avg_prediction": float(regime_df["prediction"].mean()),
                "avg_abs_prediction": float(regime_df["prediction"].abs().mean()),
                "avg_abs_actual": float(regime_df["actual"].abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def build_frontier_table(frontier_path: Path, committee_summary: dict, standalone_eval: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frontier_path.exists():
        frontier_df = pd.read_csv(frontier_path)
        for _, row in frontier_df.iterrows():
            rows.append(
                {
                    "candidate_type": row["selection_type"],
                    "candidate_name": row["pair_name"],
                    "val_rel_score": float(row["val_rel_score"]),
                    "test_rel_score": float(row["test_rel_score"]),
                    "test_amp_ratio": float(row.get("test_pred_abs_over_actual_abs", np.nan)),
                }
            )
    best_committee = committee_summary["best_committee"]
    rows.append(
        {
            "candidate_type": "current_standalone_best",
            "candidate_name": committee_summary["expert_run"] + ":" + best_committee["expert_model"],
            "val_rel_score": float(best_committee["expert_val_rel_score_overlap"]),
            "test_rel_score": float(standalone_eval["rel_score"]),
            "test_amp_ratio": np.nan,
        }
    )
    rows.append(
        {
            "candidate_type": "current_committee_best",
            "candidate_name": committee_summary["market_run"] + ":" + best_committee["market_model"],
            "val_rel_score": float(best_committee["committee_val_rel_score"]),
            "test_rel_score": float(best_committee["committee_test_rel_score"]),
            "test_amp_ratio": np.nan,
        }
    )
    return pd.DataFrame(rows)


def save_daily_ic_plot(ic_df: pd.DataFrame, plots_dir: Path, window: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, group in ic_df.groupby("model_label", sort=False):
        ax.plot(group["Date"], group["rolling_ic_spearman"], linewidth=1.5, label=f"{label} rolling-{window} IC")
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_title("Rolling Cross-Sectional IC (Spearman)")
    ax.set_ylabel("IC")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "test_rolling_ic_spearman.png", dpi=160)
    plt.close(fig)


def save_bucket_plot(bucket_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, group in bucket_df.groupby("model_label", sort=False):
        ax.plot(group["bucket"], group["actual_mean"], marker="o", linewidth=1.4, label=label)
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_title("Signal Bucket vs Actual Mean Return")
    ax.set_xlabel("Prediction bucket")
    ax.set_ylabel("Actual mean return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "test_signal_bucket_spread.png", dpi=160)
    plt.close(fig)


def save_regime_plot(regime_df: pd.DataFrame, plots_dir: Path) -> None:
    pivot = regime_df.pivot(index="regime", columns="model_label", values="rel_score").sort_index()
    ax = pivot.plot(kind="bar", figsize=(12, 5))
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_title("Regime Split rel_score")
    ax.set_ylabel("rel_score")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "test_regime_relscore.png", dpi=160)
    plt.close()


def save_frontier_plot(frontier_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(frontier_df))
    ax.bar(x, frontier_df["test_rel_score"], color=["#4e79a7", "#59a14f", "#f28e2b", "#9c755f", "#e15759"][: len(frontier_df)])
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(frontier_df["candidate_type"], rotation=20, ha="right")
    ax.set_title("Candidate Frontier by Test rel_score")
    ax.set_ylabel("test rel_score")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "candidate_frontier_relscore.png", dpi=160)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    *,
    label: str,
    standalone_eval: dict[str, float],
    committee_summary: dict,
    daily_ic: pd.DataFrame,
    regime_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
) -> None:
    best_committee = committee_summary["best_committee"]
    ic_snapshot = (
        daily_ic.groupby("model_label", as_index=False)["ic_spearman"]
        .mean()
        .rename(columns={"ic_spearman": "mean_daily_ic_spearman"})
    )
    regime_pivot = regime_df.pivot(index="regime", columns="model_label", values="rel_score").sort_index()
    lines = [
        f"# {label}",
        "",
        "## Snapshot",
        f"- Standalone test rel_score: `{standalone_eval['rel_score']:.6f}`",
        f"- Committee test rel_score: `{best_committee['committee_test_rel_score']:.6f}`",
        f"- Committee method: `{best_committee['method']}`",
        f"- Committee weight_expert: `{best_committee['weight_expert']}`",
        f"- Overlap codes: `{best_committee['overlap_codes']}`",
        "",
        "## Daily IC mean",
    ]
    for _, row in ic_snapshot.iterrows():
        lines.append(f"- {row['model_label']}: `{row['mean_daily_ic_spearman']:.6f}`")
    lines.extend(["", "## Regime rel_score"])
    for regime, row in regime_pivot.iterrows():
        parts = [f"{col}={val:.6f}" for col, val in row.dropna().items()]
        lines.append(f"- `{regime}`: " + ", ".join(parts))
    lines.extend(["", "## Candidate frontier"])
    for _, row in frontier_df.iterrows():
        lines.append(
            f"- `{row['candidate_type']}` / `{row['candidate_name']}`: "
            f"val `{row['val_rel_score']:.6f}`, test `{row['test_rel_score']:.6f}`"
        )
    (output_dir / "quant_diagnostics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    standalone_df = load_standalone_predictions(args.standalone_run, args.standalone_model)
    committee_df, committee_summary = load_committee_predictions(args.committee_dir)

    standalone_eval = evaluate_frame(standalone_df)
    committee_eval = evaluate_frame(committee_df)

    standalone_ic = add_rolling_stats(build_daily_ic(standalone_df, "standalone"), args.rolling_window)
    committee_ic = add_rolling_stats(build_daily_ic(committee_df, "committee"), args.rolling_window)
    daily_ic = pd.concat([standalone_ic, committee_ic], ignore_index=True)
    daily_ic.to_csv(output_dir / "daily_ic.csv", index=False)

    bucket_df = pd.concat(
        [
            build_bucket_summary(standalone_df, "standalone"),
            build_bucket_summary(committee_df, "committee"),
        ],
        ignore_index=True,
    )
    bucket_df.to_csv(output_dir / "bucket_summary.csv", index=False)

    regimes = build_market_regimes(committee_df, args.rolling_window)
    regimes.to_csv(output_dir / "regime_calendar.csv", index=False)
    regime_df = pd.concat(
        [
            build_regime_summary(standalone_df, regimes, "standalone"),
            build_regime_summary(committee_df, regimes, "committee"),
        ],
        ignore_index=True,
    )
    regime_df.to_csv(output_dir / "regime_summary.csv", index=False)

    frontier_df = build_frontier_table(args.frontier_shortlist, committee_summary, standalone_eval)
    frontier_df.to_csv(output_dir / "candidate_frontier.csv", index=False)

    meta = {
        "standalone_run": str(args.standalone_run),
        "standalone_model": args.standalone_model,
        "committee_dir": str(args.committee_dir),
        "standalone_eval": standalone_eval,
        "committee_eval": committee_eval,
        "best_committee": committee_summary["best_committee"],
    }
    (output_dir / "quant_diagnostics_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    save_daily_ic_plot(daily_ic, plots_dir, args.rolling_window)
    save_bucket_plot(bucket_df, plots_dir)
    save_regime_plot(regime_df, plots_dir)
    save_frontier_plot(frontier_df, plots_dir)

    write_summary(
        output_dir,
        label=args.label,
        standalone_eval=standalone_eval,
        committee_summary=committee_summary,
        daily_ic=daily_ic,
        regime_df=regime_df,
        frontier_df=frontier_df,
    )

    print(output_dir)


if __name__ == "__main__":
    main()
