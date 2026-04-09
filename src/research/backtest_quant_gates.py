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

from src.evaluation.metric import loss_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research backtests for regime and bucket gates on the best committee.")
    parser.add_argument("--committee-dir", type=Path, required=True)
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket-min", type=int, default=8, help="Inclusive min decile bucket for top-bucket gate.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_subset_rel_score(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    base_loss = loss_fn(df["actual"].to_numpy(dtype=float))
    if base_loss == 0:
        return np.nan
    abs_loss = loss_fn((df["actual"] - df["prediction"]).to_numpy(dtype=float))
    return float(1.0 - abs_loss / base_loss)


def build_equity_frame(selected: pd.DataFrame, all_dates: pd.Index) -> pd.DataFrame:
    daily = selected.groupby("Date", as_index=False)["actual"].mean().rename(columns={"actual": "strategy_return"})
    calendar = pd.DataFrame({"Date": all_dates}).merge(daily, on="Date", how="left")
    calendar["strategy_return"] = calendar["strategy_return"].fillna(0.0)
    calendar["equity"] = (1.0 + calendar["strategy_return"]).cumprod()
    return calendar


def summarize_gate(name: str, selected: pd.DataFrame, total_rows: int, all_dates: pd.Index) -> tuple[dict[str, object], pd.DataFrame]:
    if selected.empty:
        calendar = pd.DataFrame({"Date": all_dates, "strategy_return": 0.0})
        calendar["equity"] = 1.0
        return (
            {
                "gate_name": name,
                "trade_count": 0,
                "coverage": 0.0,
                "directional_accuracy": np.nan,
                "avg_actual_return": np.nan,
                "subset_rel_score": np.nan,
                "calendar_final_equity": 1.0,
                "active_only_final_equity": 1.0,
            },
            calendar,
        )

    calendar = build_equity_frame(selected, all_dates)
    active_only_final_equity = float((1.0 + selected["actual"]).cumprod().iloc[-1])
    summary = {
        "gate_name": name,
        "trade_count": int(len(selected)),
        "coverage": float(len(selected) / total_rows),
        "directional_accuracy": float((selected["actual"] >= 0.0).mean()),
        "avg_actual_return": float(selected["actual"].mean()),
        "subset_rel_score": compute_subset_rel_score(selected),
        "calendar_final_equity": float(calendar["equity"].iloc[-1]),
        "active_only_final_equity": active_only_final_equity,
    }
    return summary, calendar


def save_equity_plot(curves: dict[str, pd.DataFrame], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, curve in curves.items():
        ax.plot(curve["Date"], curve["equity"], linewidth=1.5, label=name)
    ax.set_title("Phase 6 Gates: Calendar Equity Comparison")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "phase6_gate_equity_curve.png", dpi=160)
    plt.close(fig)


def save_metric_bar(summary_df: pd.DataFrame, column: str, title: str, filename: str, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(summary_df["gate_name"], summary_df[column], color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759"])
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(column)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    predictions = pd.read_csv(args.committee_dir / "best_committee_predictions.csv")
    predictions = predictions[predictions["split"] == "test"].copy()
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    predictions["prediction"] = predictions["prediction_committee"]
    predictions["bucket"] = pd.qcut(
        predictions["prediction"].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    )

    regime_summary = pd.read_csv(args.quant_dir / "regime_summary.csv")
    regime_calendar = pd.read_csv(args.quant_dir / "regime_calendar.csv")
    regime_calendar["Date"] = pd.to_datetime(regime_calendar["Date"])
    positive_regimes = regime_summary[
        (regime_summary["model_label"] == "committee") & (regime_summary["rel_score"] > 0.0)
    ]["regime"].tolist()

    enriched = predictions.merge(regime_calendar[["Date", "regime"]], on="Date", how="left")
    enriched["baseline_active"] = enriched["prediction"] >= 0.0
    enriched["regime_active"] = enriched["baseline_active"] & enriched["regime"].isin(positive_regimes)
    enriched["bucket_active"] = enriched["baseline_active"] & (enriched["bucket"] >= args.bucket_min)
    enriched["combined_active"] = enriched["regime_active"] & (enriched["bucket"] >= args.bucket_min)

    all_dates = pd.Index(sorted(enriched["Date"].drop_duplicates()))
    total_rows = len(enriched)

    gates = {
        "baseline_threshold": enriched[enriched["baseline_active"]].copy(),
        "regime_gate": enriched[enriched["regime_active"]].copy(),
        "bucket_gate_top20": enriched[enriched["bucket_active"]].copy(),
        "regime_plus_bucket": enriched[enriched["combined_active"]].copy(),
    }

    summaries: list[dict[str, object]] = []
    curves: dict[str, pd.DataFrame] = {}
    for name, selected in gates.items():
        summary, curve = summarize_gate(name, selected, total_rows, all_dates)
        summaries.append(summary)
        curves[name] = curve
        curve.to_csv(output_dir / f"{name}_equity_curve.csv", index=False)
        selected.to_csv(output_dir / f"{name}_selected_rows.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "phase6_gate_summary.csv", index=False)

    meta = {
        "committee_dir": str(args.committee_dir),
        "quant_dir": str(args.quant_dir),
        "positive_regimes_from_committee_test": positive_regimes,
        "bucket_min": args.bucket_min,
    }
    (output_dir / "phase6_gate_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    save_equity_plot(curves, plots_dir)
    save_metric_bar(summary_df, "subset_rel_score", "Phase 6 Gates: Subset rel_score", "phase6_gate_relscore.png", plots_dir)
    save_metric_bar(summary_df, "calendar_final_equity", "Phase 6 Gates: Calendar Equity", "phase6_gate_calendar_equity.png", plots_dir)

    lines = [
        "# Phase 6 Gate Backtests",
        "",
        f"- Positive regimes used: `{','.join(positive_regimes)}`",
        f"- Bucket gate: `bucket >= {args.bucket_min}`",
        "",
        "## Summary",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['gate_name']}`: subset_rel_score `{row['subset_rel_score']:.6f}`, "
            f"calendar_equity `{row['calendar_final_equity']:.6f}`, "
            f"active_only_equity `{row['active_only_final_equity']:.6f}`, "
            f"coverage `{row['coverage']:.3f}`"
        )
    (output_dir / "phase6_gate_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(output_dir)


if __name__ == "__main__":
    main()
