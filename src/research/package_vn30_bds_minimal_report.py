from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
DATA_PATH = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"

from src.evaluation.metric import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal 2-plot VN30 report from a training run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="lstm_best_by_val")
    parser.add_argument("--title", default="VN30 Minimal Report")
    parser.add_argument("--backtest-suffix", default="advisor_clean")
    parser.add_argument("--code-list-path", type=Path, default=None)
    parser.add_argument("--exclude-codes", default="")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_code_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8")
    return [token.strip().upper() for token in re.split(r"[\s,]+", text) if token.strip()]


def parse_code_list_arg(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_test_frame(run_dir: Path, model_name: str) -> pd.DataFrame:
    preds = pd.read_csv(run_dir / "reports" / "core" / "predictions.csv")
    preds = preds[(preds["split"] == "test") & (preds["model"] == model_name)].copy()
    preds["Date"] = pd.to_datetime(preds["Date"])

    prices = pd.read_csv(DATA_PATH, usecols=["Date", "code", "adjust", "close"])
    prices["Date"] = pd.to_datetime(prices["Date"])
    price_col = "adjust" if "adjust" in prices.columns else "close"
    merged = preds.merge(prices[["Date", "code", price_col]], on=["Date", "code"], how="left")
    merged = merged.rename(columns={price_col: "base_price"})
    merged["actual_next_price"] = merged["base_price"] * (1.0 + merged["actual"])
    merged["predicted_next_price"] = merged["base_price"] * (1.0 + merged["prediction"])
    merged["return_error"] = merged["prediction"] - merged["actual"]
    return merged.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)


def build_code_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for code, part in df.groupby("code", sort=True):
        part = part.sort_values("Date", kind="stable").reset_index(drop=True)
        metric = evaluate(part["prediction"].to_numpy(), part["actual"].to_numpy())
        error = part["return_error"].to_numpy(dtype=float)
        rows.append(
            {
                "code": str(code),
                "actual_mean": float(part["actual"].mean()),
                "prediction_mean": float(part["prediction"].mean()),
                "error_mean": float(error.mean()),
                "abs_error_mean": float(np.abs(error).mean()),
                "rel_score": float(metric["rel_score"]),
                "error_min": float(np.min(error)),
                "error_q25": float(np.quantile(error, 0.25)),
                "error_mean_hist": float(np.mean(error)),
                "error_q75": float(np.quantile(error, 0.75)),
                "error_max": float(np.max(error)),
            }
        )
    return pd.DataFrame(rows).sort_values("code", kind="stable").reset_index(drop=True)


def compute_filtered_summary_metrics(df: pd.DataFrame) -> dict[str, float]:
    metric = evaluate(df["prediction"].to_numpy(), df["actual"].to_numpy(), group_ids=df["code"].to_numpy())
    active = df[df["prediction"] >= 0.0].copy()
    if active.empty:
        return {
            "test_rel_score": float(metric["rel_score"]),
            "trade_count": 0,
            "coverage": 0.0,
            "final_equity": 1.0,
            "directional_accuracy": float("nan"),
        }
    return {
        "test_rel_score": float(metric["rel_score"]),
        "trade_count": int(len(active)),
        "coverage": float(len(active) / len(df)),
        "final_equity": float((1.0 + active["actual"]).cumprod().iloc[-1]),
        "directional_accuracy": float(np.mean(active["actual"] >= 0.0)),
    }


def render_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for value in row.tolist():
            if isinstance(value, str):
                cells.append(value)
            else:
                cells.append(f"{float(value):.6f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def resolve_subplot_layout(code_count: int) -> tuple[int, int]:
    if code_count > 20:
        ncols = 5
    elif code_count > 12:
        ncols = 4
    elif code_count > 6:
        ncols = 3
    else:
        ncols = 2
    nrows = int(np.ceil(code_count / ncols))
    return nrows, ncols


def save_price_plot(df: pd.DataFrame, code_summary: pd.DataFrame, plots_dir: Path, title: str, codes: list[str]) -> str:
    nrows, ncols = resolve_subplot_layout(len(codes))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.3 * nrows), sharex=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, code in zip(axes.flatten(), codes):
        part = df[df["code"] == code].copy()
        summary_row = code_summary[code_summary["code"] == code].iloc[0]
        ax.plot(part["Date"], part["actual_next_price"], label="Actual next price", linewidth=1.5, color="#222222")
        ax.plot(part["Date"], part["predicted_next_price"], label="Predicted next price", linewidth=1.2, color="#1f77b4")
        ax.set_title(code)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        stats_text = (
            f"actual_mean={summary_row['actual_mean']:.4f}\n"
            f"pred_mean={summary_row['prediction_mean']:.4f}\n"
            f"error_mean={summary_row['error_mean']:.4f}\n"
            f"abs_error={summary_row['abs_error_mean']:.4f}\n"
            f"rel_score={summary_row['rel_score']:.4f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
        )
    for ax in axes.flatten()[len(codes):]:
        ax.axis("off")
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"{title}: Actual vs Predicted Price from Return", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    target = plots_dir / "test_actual_vs_predicted_price_from_return.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return str(target)


def save_error_hist(df: pd.DataFrame, code_summary: pd.DataFrame, plots_dir: Path, title: str, codes: list[str]) -> str:
    nrows, ncols = resolve_subplot_layout(len(codes))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.3 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, code in zip(axes.flatten(), codes):
        part = df[df["code"] == code].copy()
        summary_row = code_summary[code_summary["code"] == code].iloc[0]
        ax.hist(part["return_error"], bins=30, color="#d62728", alpha=0.75)
        ax.axvline(0.0, linestyle="--", color="#555555", linewidth=1.0)
        ax.axvline(summary_row["error_q25"], linestyle=":", color="#1f77b4", linewidth=1.0)
        ax.axvline(summary_row["error_mean_hist"], linestyle="-", color="#2ca02c", linewidth=1.0)
        ax.axvline(summary_row["error_q75"], linestyle=":", color="#ff7f0e", linewidth=1.0)
        ax.set_title(code)
        ax.set_xlabel("prediction - actual")
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)
        stats_text = (
            f"min={summary_row['error_min']:.4f}\n"
            f"q25={summary_row['error_q25']:.4f}\n"
            f"mean={summary_row['error_mean_hist']:.4f}\n"
            f"q75={summary_row['error_q75']:.4f}\n"
            f"max={summary_row['error_max']:.4f}"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
        )
    for ax in axes.flatten()[len(codes):]:
        ax.axis("off")
    fig.suptitle(f"{title}: Return Error Histogram", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    target = plots_dir / "test_return_error_hist.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return str(target)


def main() -> None:
    args = parse_args()
    ensure_clean_dir(args.out_dir)
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_df = load_test_frame(args.run_dir, args.model)
    requested_codes = load_code_list(args.code_list_path)
    explicit_excludes = parse_code_list_arg(args.exclude_codes)
    available_codes = sorted(test_df["code"].dropna().astype(str).unique().tolist())
    codes = [code for code in requested_codes if code in available_codes] if requested_codes else available_codes
    if explicit_excludes:
        codes = [code for code in codes if code not in set(explicit_excludes)]
    missing_codes = [code for code in requested_codes if code not in available_codes]
    test_df = test_df[test_df["code"].isin(codes)].copy()
    code_summary = build_code_summary(test_df)
    metrics = load_json(args.run_dir / "reports" / "core" / "metrics.json")
    filtered_metrics = compute_filtered_summary_metrics(test_df)
    price_plot = save_price_plot(test_df, code_summary, plots_dir, args.title, codes)
    error_plot = save_error_hist(test_df, code_summary, plots_dir, args.title, codes)

    summary_table = render_markdown_table(code_summary)
    missing_block = ""
    if requested_codes:
        missing_block = (
            f"- requested codes from list: `{','.join(requested_codes)}`\n"
            f"- available codes in run: `{','.join(codes)}`\n"
            f"- missing codes in run: `{','.join(missing_codes) if missing_codes else 'none'}`\n"
            f"- explicitly excluded from report: `{','.join(explicit_excludes) if explicit_excludes else 'none'}`\n"
        )
    summary = f"""# {args.title}

- run: `{args.run_dir.name}`
- codes: `{",".join(codes)}`
- model: `{args.model}`
- val rel_score: `{float(metrics[args.model]['val']['rel_score']):.6f}`
- filtered test rel_score: `{filtered_metrics['test_rel_score']:.6f}`
- filtered backtest final equity: `{filtered_metrics['final_equity']:.6f}`
- filtered backtest directional accuracy: `{filtered_metrics['directional_accuracy']:.6f}`
- filtered trade count: `{filtered_metrics['trade_count']}`
- filtered coverage: `{filtered_metrics['coverage']:.6f}`
{missing_block}

## Plots

- `test_actual_vs_predicted_price_from_return.png`
- `test_return_error_hist.png`

## Per-Code Summary

{summary_table}
"""
    (args.out_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    manifest = {
        "summary_md": str(args.out_dir / "summary.md"),
        "price_plot": price_plot,
        "error_hist_plot": error_plot,
        "run_dir": str(args.run_dir),
        "model": args.model,
        "codes": codes,
        "requested_codes": requested_codes,
        "missing_codes": missing_codes,
        "excluded_codes": explicit_excludes,
        "filtered_metrics": filtered_metrics,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
