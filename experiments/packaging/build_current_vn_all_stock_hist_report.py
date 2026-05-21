from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
GOLD_REPORT_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "gold" / "reports"
DEFAULT_ROUTER_REPORT = RUN_ROOT / "reports" / "router_analysis" / "anchor_sector19_router_20260425_r01"
DEFAULT_RANK_ROUTER_REPORT = RUN_ROOT / "reports" / "rank_router_train_selected" / "anchor_sector19_rank_router_20260427_r01"


@dataclass(frozen=True)
class SplitWindow:
    split: str
    label: str
    start: str | None
    end: str | None


DEFAULT_SPLIT_WINDOWS = (
    SplitWindow("train", "Train", None, "2020-03-31"),
    SplitWindow("val", "In-sample", "2020-04-01", "2022-11-15"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all-stock VN histogram report for current anchor/router candidates."
    )
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--rank-router-report", type=Path, default=DEFAULT_RANK_ROUTER_REPORT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GOLD_REPORT_ROOT / "current_vn_all_stock_hist_20260427_r01",
    )
    parser.add_argument(
        "--candidates",
        default="anchor,sector19_down_up_anchor_else,train_rank_regime_ic_weight",
    )
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "candidate"


def loss_fn(values: np.ndarray) -> float:
    values = np.abs(np.asarray(values, dtype=float))
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, 0.5) + 0.5 * np.quantile(values, 0.9))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base_loss = loss_fn(actual)
    error_loss = loss_fn(actual - prediction)
    if not np.isfinite(base_loss) or base_loss <= 0.0:
        return float("nan")
    return float(1.0 - error_loss / base_loss)


def stabilized_proxy(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    error_abs = np.abs(actual - np.asarray(prediction, dtype=float))
    base_floor = max(loss_fn(actual), 1e-4)
    denom = np.maximum(np.abs(actual), base_floor)
    return np.clip(1.0 - error_abs / denom, -1.5, 1.0)


def load_router_predictions(router_report: Path) -> pd.DataFrame:
    path = router_report / "candidate_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing router candidate predictions: {path}")
    df = pd.read_csv(path)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["actual_date"] = pd.to_datetime(df["actual_date"])
    return df[df["split"].isin({"train", "val"})].copy()


def load_rank_router_selection(rank_router_report: Path) -> dict[str, object]:
    path = rank_router_report / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing rank-router summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"Rank-router summary has no selection object: {path}")
    return selection


def add_train_selected_rank_router_candidates(df: pd.DataFrame, selection: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    anchor = out["prediction__anchor"].to_numpy(dtype=float)
    challenger = out["prediction__challenger"].to_numpy(dtype=float)

    global_ic_weight = float(selection["global_mean_ic_weight"])
    regime_ic_weights = {
        str(key): float(value)
        for key, value in dict(selection.get("regime_mean_ic_weights", {})).items()
    }
    regime_weight = out["regime"].astype(str).map(regime_ic_weights).fillna(global_ic_weight).to_numpy(dtype=float)
    out["candidate__train_rank_regime_ic_weight"] = (1.0 - regime_weight) * anchor + regime_weight * challenger

    global_tb_weight = float(selection["global_top_bottom_equity_weight"])
    regime_tb_weights = {
        str(key): float(value)
        for key, value in dict(selection.get("regime_top_bottom_equity_weights", {})).items()
    }
    regime_tb_weight = out["regime"].astype(str).map(regime_tb_weights).fillna(global_tb_weight).to_numpy(dtype=float)
    out["candidate__train_rank_regime_topbottom_weight"] = (1.0 - regime_tb_weight) * anchor + regime_tb_weight * challenger
    return out


def summarize_split(frame: pd.DataFrame, candidate: str, window: SplitWindow) -> dict[str, object]:
    column = f"candidate__{candidate}"
    selected = frame[frame["split"] == window.split].copy()
    if column not in selected.columns:
        raise ValueError(f"Missing candidate column: {column}")
    prediction = selected[column].to_numpy(dtype=float)
    actual = selected["actual"].to_numpy(dtype=float)
    error = prediction - actual
    proxy = stabilized_proxy(actual, prediction)
    return {
        "candidate": candidate,
        "split": window.split,
        "label": window.label,
        "date_start": window.start or selected["actual_date"].min().strftime("%Y-%m-%d"),
        "date_end": window.end or selected["actual_date"].max().strftime("%Y-%m-%d"),
        "row_count": int(len(selected)),
        "code_count": int(selected["code"].astype(str).nunique()),
        "rel_score": rel_score(actual, prediction),
        "directional_accuracy": float((np.sign(prediction) == np.sign(actual)).mean()),
        "error_q2": float(np.quantile(error, 0.2)),
        "error_q8": float(np.quantile(error, 0.8)),
        "error_mean": float(np.mean(error)),
        "error_std": float(np.std(error)),
        "proxy_mean": float(np.mean(proxy)),
        "proxy_median": float(np.median(proxy)),
        "proxy_positive_share": float(np.mean(proxy > 0.0)),
        "proxy_hard_left_share": float(np.mean(proxy < -0.5)),
    }


def summarize_by_code(frame: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        column = f"candidate__{candidate}"
        for (split, code), group in frame.groupby(["split", "code"], sort=True):
            prediction = group[column].to_numpy(dtype=float)
            actual = group["actual"].to_numpy(dtype=float)
            error = prediction - actual
            rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "code": code,
                    "row_count": int(len(group)),
                    "rel_score": rel_score(actual, prediction),
                    "directional_accuracy": float((np.sign(prediction) == np.sign(actual)).mean()),
                    "error_q2": float(np.quantile(error, 0.2)),
                    "error_q8": float(np.quantile(error, 0.8)),
                    "error_band_width": float(np.quantile(error, 0.8) - np.quantile(error, 0.2)),
                    "error_mean": float(np.mean(error)),
                    "error_std": float(np.std(error)),
                }
            )
    return pd.DataFrame(rows)


def arrays_for_split(frame: pd.DataFrame, candidate: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    column = f"candidate__{candidate}"
    selected = frame[frame["split"] == split].copy()
    prediction = selected[column].to_numpy(dtype=float)
    actual = selected["actual"].to_numpy(dtype=float)
    return prediction - actual, stabilized_proxy(actual, prediction)


def build_candidate_plot(
    frame: pd.DataFrame,
    candidate: str,
    summaries: list[dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, len(DEFAULT_SPLIT_WINDOWS), figsize=(14.2, 8.2), squeeze=False)
    summary_by_split = {str(item["split"]): item for item in summaries}
    colors = {
        "error": "#345995",
        "proxy": "#4f8f68",
        "q2": "#f28e2b",
        "q8": "#d1495b",
        "zero": "#222222",
        "mean": "#d1495b",
        "agg": "#345995",
    }
    for idx, window in enumerate(DEFAULT_SPLIT_WINDOWS):
        summary = summary_by_split[window.split]
        error_values, proxy_values = arrays_for_split(frame, candidate, window.split)
        error_ax = axes[0, idx]
        proxy_ax = axes[1, idx]

        error_ax.hist(error_values, bins=60, color=colors["error"], alpha=0.82)
        error_ax.axvline(0.0, color=colors["zero"], linewidth=1.0, alpha=0.4)
        error_ax.axvline(summary["error_q2"], color=colors["q2"], linestyle="--", linewidth=1.4, label="q2")
        error_ax.axvline(summary["error_q8"], color=colors["q8"], linestyle="--", linewidth=1.4, label="q8")
        error_ax.set_xlim(np.quantile(error_values, 0.005), np.quantile(error_values, 0.995))
        error_ax.set_title(f"{window.label} | E = prediction - actual")
        error_ax.set_xlabel("E")
        error_ax.set_ylabel("count")
        error_ax.grid(True, alpha=0.2)
        error_ax.legend(loc="upper left", frameon=True, fontsize=8)
        error_ax.text(
            1.02,
            0.96,
            "\n".join(
                [
                    f"{summary['date_start']}..{summary['date_end']}",
                    f"stocks={summary['code_count']}",
                    f"rows={summary['row_count']}",
                    f"q2={summary['error_q2']:+.4f}",
                    f"q8={summary['error_q8']:+.4f}",
                    f"mean={summary['error_mean']:+.4f}",
                    f"std={summary['error_std']:.4f}",
                ]
            ),
            transform=error_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.1,
            clip_on=False,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
        )

        proxy_ax.hist(proxy_values, bins=np.linspace(-1.5, 1.0, 41), color=colors["proxy"], alpha=0.82)
        proxy_ax.axvline(0.0, color=colors["zero"], linewidth=1.0, alpha=0.4)
        proxy_ax.axvline(summary["proxy_mean"], color=colors["mean"], linewidth=1.4, label="proxy mean")
        proxy_ax.axvline(summary["rel_score"], color=colors["agg"], linestyle="--", linewidth=1.4, label="aggregate rel_score")
        proxy_ax.set_title(f"{window.label} | Stabilized relative_score proxy")
        proxy_ax.set_xlabel("proxy relative_score")
        proxy_ax.set_ylabel("count")
        proxy_ax.grid(True, alpha=0.2)
        proxy_ax.legend(loc="upper left", frameon=True, fontsize=8)
        proxy_ax.text(
            1.02,
            0.96,
            "\n".join(
                [
                    f"rel_score={summary['rel_score']:+.4f}",
                    f"dir_acc={summary['directional_accuracy']:.1%}",
                    f"proxy_mean={summary['proxy_mean']:+.4f}",
                    f"proxy_median={summary['proxy_median']:+.4f}",
                    f"proxy>0={summary['proxy_positive_share']:.1%}",
                    f"proxy<-0.5={summary['proxy_hard_left_share']:.1%}",
                ]
            ),
            transform=proxy_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.1,
            clip_on=False,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
        )

    fig.suptitle(f"VN all-stock histogram | {candidate}", fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.82, top=0.88, bottom=0.09, hspace=0.34, wspace=0.42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_overlay_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    val = summary_df[summary_df["split"] == "val"].copy()
    val = val.sort_values("rel_score", ascending=True, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].barh(val["candidate"], val["rel_score"], color="#345995", alpha=0.86)
    axes[0].axvline(0.0, color="#222222", linewidth=1.0, alpha=0.45)
    axes[0].set_title("In-sample aggregate rel_score")
    axes[0].set_xlabel("rel_score")
    axes[0].grid(axis="x", alpha=0.22)

    width = np.maximum(val["error_q8"] - val["error_q2"], 0.0)
    axes[1].barh(val["candidate"], width, color="#d1495b", alpha=0.8)
    axes[1].set_title("In-sample error band width q8 - q2")
    axes[1].set_xlabel("width")
    axes[1].grid(axis="x", alpha=0.22)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_readme(output_dir: Path, summary_df: pd.DataFrame, plot_paths: dict[str, Path], overlay_path: Path) -> None:
    val = summary_df[summary_df["split"] == "val"].sort_values("rel_score", ascending=False, kind="stable")
    code_summary_path = output_dir / "core" / "all_stock_hist_by_code.csv"
    by_code = pd.read_csv(code_summary_path) if code_summary_path.exists() else pd.DataFrame()
    val_by_code = by_code[by_code["split"].eq("val")].copy() if not by_code.empty else pd.DataFrame()
    weak_lines: list[str] = []
    if not val_by_code.empty:
        anchor_weak = (
            val_by_code[val_by_code["candidate"].eq("anchor")]
            .sort_values(["rel_score", "error_band_width"], ascending=[True, False], kind="stable")
            .head(8)
        )
        weak_lines = [
            "",
            "## Weakest In-Sample Stocks For Anchor",
            "",
            "| Code | rel_score | Direction | Error q2/q8 | Error band | Bias mean | Rows |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for _, row in anchor_weak.iterrows():
            weak_lines.append(
                "| "
                f"`{row['code']}` | {float(row['rel_score']):+.4f} | {float(row['directional_accuracy']):.1%} | "
                f"{float(row['error_q2']):+.4f} / {float(row['error_q8']):+.4f} | "
                f"{float(row['error_band_width']):.4f} | {float(row['error_mean']):+.4f} | {int(row['row_count'])} |"
            )

    lines = [
        "# Current VN All-Stock Histogram Report",
        "",
        "Scope: train and in-sample validation only. No out-sample/test data is used.",
        "",
        "Universe: all stock rows available in the current VN anchor/router prediction frame.",
        "",
        "Conventions:",
        "",
        "- `E = prediction - actual`",
        "- `q2` and `q8` are the 20% and 80% quantiles of `E` over all stocks and days in the split",
        "- relative_score histogram uses a stabilized local proxy; aggregate `rel_score` uses the repo q50/q90 loss formula",
        "",
        "## Quick Read",
        "",
        "| Candidate | In-sample rel_score | Direction | Error q2/q8 | Proxy > 0 | Rows | Stocks |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for _, row in val.iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {float(row['rel_score']):+.4f} | {float(row['directional_accuracy']):.1%} | "
            f"{float(row['error_q2']):+.4f} / {float(row['error_q8']):+.4f} | "
            f"{float(row['proxy_positive_share']):.1%} | {int(row['row_count'])} | {int(row['code_count'])} |"
        )

    lines.extend(weak_lines)
    lines.extend(
        [
            "",
            "## Overlay",
            "",
            f"![overlay]({overlay_path.relative_to(output_dir).as_posix()})",
            "",
            "## Candidate Histograms",
            "",
        ]
    )
    for candidate, plot_path in plot_paths.items():
        rel = plot_path.relative_to(output_dir).as_posix()
        lines.extend([f"### `{candidate}`", "", f"![{candidate}]({rel})", ""])

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    core_dir = output_dir / "core"
    plots_dir = output_dir / "plots"
    core_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    frame = load_router_predictions(args.router_report)
    selection = load_rank_router_selection(args.rank_router_report)
    frame = add_train_selected_rank_router_candidates(frame, selection)
    candidates = split_csv(args.candidates)

    summary_rows: list[dict[str, object]] = []
    plot_paths: dict[str, Path] = {}
    for candidate in candidates:
        candidate_summaries = [summarize_split(frame, candidate, window) for window in DEFAULT_SPLIT_WINDOWS]
        summary_rows.extend(candidate_summaries)
        plot_path = plots_dir / f"{slugify(candidate)}__all_stock_hist.png"
        build_candidate_plot(frame, candidate, candidate_summaries, plot_path)
        plot_paths[candidate] = plot_path

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = core_dir / "all_stock_hist_summary.csv"
    summary_json = core_dir / "all_stock_hist_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    by_code_df = summarize_by_code(frame, candidates)
    by_code_df.to_csv(core_dir / "all_stock_hist_by_code.csv", index=False)

    overlay_path = plots_dir / "candidate_overlay_val_relscore_error_band.png"
    build_overlay_plot(summary_df, overlay_path)
    write_readme(output_dir, summary_df, plot_paths, overlay_path)

    manifest = {
        "router_report": str(args.router_report),
        "rank_router_report": str(args.rank_router_report),
        "output_dir": str(output_dir),
        "candidates": candidates,
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "by_code_csv": str(core_dir / "all_stock_hist_by_code.csv"),
        "plots": {candidate: str(path) for candidate, path in plot_paths.items()},
        "overlay_plot": str(overlay_path),
        "uses_out_sample": False,
    }
    (core_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
