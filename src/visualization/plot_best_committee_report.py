from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RAW_PRICE_DATASET = Path("data/processed/assets/data_info_vn/history/vn_gold_recommended.csv")


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def robust_loss(series: pd.Series) -> float:
    abs_series = series.abs()
    return float(abs_series.quantile(0.5) + 0.5 * abs_series.quantile(0.9))


def round_values(values: list[float], ndigits: int = 3) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def stat_arr_plot(values: pd.Series, title: str, output_path: Path) -> None:
    x = values.dropna().astype(float)
    if x.empty:
        return

    stats = round_values(
        [
            x.min(),
            x.quantile(0.25),
            x.mean(),
            x.quantile(0.75),
            x.max(),
        ]
    )
    stat_line = "|" + "".join(
        [
            f"  {label} = {value}  |"
            for label, value in zip(["min", "q25", "mean", "q75", "max"], stats)
        ]
    )
    print(stat_line)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.hist(x, bins=10)
    ax.set_title(title)
    ax.set_xlabel("error")
    ax.set_ylabel("count")
    stats_text = "\n".join(
        [
            f"min = {stats[0]}",
            f"q25 = {stats[1]}",
            f"mean = {stats[2]}",
            f"q75 = {stats[3]}",
            f"max = {stats[4]}",
        ]
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#666666"},
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_per_code_price_overlays(summary_path: Path, output_dir: Path) -> list[Path]:
    summary = load_summary(summary_path)
    report_dir = summary_path.parent
    predictions = pd.read_csv(report_dir / "best_committee_predictions.csv")
    prices = pd.read_csv(RAW_PRICE_DATASET, usecols=["Date", "code", "adjust"])

    prices["Date"] = pd.to_datetime(prices["Date"])
    test_df = predictions[predictions["split"] == "test"].copy()
    test_df["Date"] = pd.to_datetime(test_df["Date"])

    global_rel = float(summary["best_committee"]["committee_test_rel_score"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for code in sorted(test_df["code"].unique()):
        code_pred = test_df[test_df["code"] == code].sort_values("Date").copy()
        code_prices = prices[prices["code"] == code].sort_values("Date").copy()
        merged = code_pred.merge(code_prices, on=["Date", "code"], how="left")
        merged = merged.dropna(subset=["adjust"]).reset_index(drop=True)
        if merged.empty:
            continue

        merged["actual_next_price"] = merged["adjust"].shift(-1)
        merged["predicted_next_price"] = merged["adjust"] * (1.0 + merged["prediction_committee"])
        merged = merged.dropna(subset=["actual_next_price", "predicted_next_price"]).reset_index(drop=True)
        if merged.empty:
            continue

        actual_ret_pct = merged["actual"] * 100.0
        pred_ret_pct = merged["prediction_committee"] * 100.0
        base = robust_loss(actual_ret_pct)
        abs_loss = robust_loss(actual_ret_pct - pred_ret_pct)
        rel = 1.0 - abs_loss / base if base != 0 else 0.0

        fig = plt.figure(figsize=(11.5, 8.2), facecolor="#111111")
        fig.text(
            0.12,
            0.94,
            f"{code}  |  base = {base:.3f}  |  abs = {abs_loss:.3f}  |  rel = {rel:.3f}",
            color="white",
            fontsize=16,
            family="monospace",
            fontweight="bold",
        )
        fig.text(
            0.12,
            0.90,
            f"Global committee test rel_score = {global_rel:.4f}",
            color="#d0d0d0",
            fontsize=11,
        )

        ax = fig.add_axes([0.07, 0.12, 0.88, 0.74], facecolor="white")
        ax.plot(merged["Date"], merged["predicted_next_price"], color="black", linewidth=2.4, label="prediction")
        ax.plot(merged["Date"], merged["actual_next_price"], color="#00A33A", linewidth=2.0, label="target")
        ax.grid(True, color="#9a9a9a", alpha=0.65, linewidth=1)
        ax.legend(loc="upper left", frameon=True, fontsize=11)
        ax.tick_params(labelsize=11)
        ax.set_ylabel("next adjusted price")

        out = output_dir / f"{code}_teacher_price_overlay.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(out)

    return outputs


def build_per_code_return_error_hists(summary_path: Path, output_dir: Path) -> list[Path]:
    report_dir = summary_path.parent
    predictions = pd.read_csv(report_dir / "best_committee_predictions.csv")

    test_df = predictions[predictions["split"] == "test"].copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for code in sorted(test_df["code"].unique()):
        code_df = test_df[test_df["code"] == code].sort_values("Date").copy()
        if code_df.empty:
            continue

        errors = (code_df["actual"] - code_df["prediction_committee"]) * 100.0
        out = output_dir / f"{code}_return_error_hist.png"
        stat_arr_plot(errors, f"{code} error = actual_returns - prediction_returns (%)", out)
        outputs.append(out)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot report artifacts for the best committee run.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to best_committee_summary.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save per-code price overlays. Defaults to per_code_price_overlays next to summary.",
    )
    parser.add_argument(
        "--return-hist-dir",
        type=Path,
        default=None,
        help="Directory to save per-code return-error histograms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.summary.with_name("per_code_price_overlays")
    return_hist_dir = args.return_hist_dir or args.summary.with_name("per_code_return_error_hists")
    outputs = build_per_code_price_overlays(args.summary, output_dir)
    return_hist_outputs = build_per_code_return_error_hists(args.summary, return_hist_dir)
    for out in outputs:
        print(out)
    for out in return_hist_outputs:
        print(out)


if __name__ == "__main__":
    main()
