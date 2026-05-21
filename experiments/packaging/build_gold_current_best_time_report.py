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
GOLD_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "gold"

from src.evaluation.metric import evaluate


@dataclass(frozen=True)
class TimeWindow:
    key: str
    label: str
    start: str | None
    end: str | None


@dataclass(frozen=True)
class PackageSpec:
    package_name: str
    package_dir: Path
    roles: tuple[str, ...]


DEFAULT_TIME_WINDOWS: tuple[TimeWindow, ...] = (
    TimeWindow(key="train", label="Train", start=None, end="2020-03-31"),
    TimeWindow(key="in_sample", label="In-sample", start="2020-04-01", end="2022-11-15"),
)

OUT_SAMPLE_WINDOW = TimeWindow(key="out_sample", label="Out-sample", start="2022-11-16", end=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build time-window histogram report for the current best gold packages."
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=GOLD_ROOT,
        help="Gold package root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GOLD_ROOT / "reports" / "current_best_time_windows_20260424",
        help="Output report directory.",
    )
    parser.add_argument(
        "--include-out-sample",
        action="store_true",
        help="Include out-sample only when explicitly finalizing a model choice.",
    )
    return parser.parse_args()


def resolve_time_windows(include_out_sample: bool) -> tuple[TimeWindow, ...]:
    if include_out_sample:
        return (*DEFAULT_TIME_WINDOWS, OUT_SAMPLE_WINDOW)
    return DEFAULT_TIME_WINDOWS


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_package_specs(gold_root: Path) -> list[PackageSpec]:
    gold_index = load_json(gold_root / "gold_index.json")
    package_roles: dict[str, list[str]] = {}
    for key, value in gold_index.items():
        if not key.startswith("current_best_"):
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        package_roles.setdefault(value, []).append(key)

    package_specs: list[PackageSpec] = []
    for package_name, roles in sorted(package_roles.items()):
        package_dir = gold_root / package_name
        if package_dir.exists():
            package_specs.append(
                PackageSpec(
                    package_name=package_name,
                    package_dir=package_dir,
                    roles=tuple(sorted(roles)),
                )
            )
    return package_specs


def read_package_readme(package_dir: Path) -> str:
    readme_path = package_dir / "README.md"
    if not readme_path.exists():
        return ""
    return readme_path.read_text(encoding="utf-8")


def resolve_model_name(package_dir: Path) -> str:
    core_dir = package_dir / "core"
    prediction_files = sorted(core_dir.glob("predictions*.csv"))
    for path in prediction_files:
        df = pd.read_csv(path, usecols=["model"])
        model_names = sorted(df["model"].dropna().astype(str).unique().tolist())
        if len(model_names) == 1:
            return model_names[0]

    readme_text = read_package_readme(package_dir)
    match = re.search(r"^Model:\s*`([^`]+)`", readme_text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()

    raise FileNotFoundError(f"Unable to resolve model name for package: {package_dir}")


def resolve_source_run_dir(package_dir: Path) -> Path | None:
    readme_text = read_package_readme(package_dir)
    source_match = re.search(r"^Source run:\s*`([^`]+)`", readme_text, flags=re.MULTILINE)
    if source_match:
        source_path = Path(source_match.group(1).strip())
        return source_path if source_path.is_absolute() else ROOT / source_path
    return None


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_full_prediction_frame(package_dir: Path, model_name: str) -> pd.DataFrame:
    package_prediction_path = package_dir / "core" / "predictions.csv"
    if package_prediction_path.exists():
        package_df = read_predictions(package_prediction_path)
        model_df = package_df[package_df["model"] == model_name].copy()
        if not model_df.empty and model_df["Date"].min() <= pd.Timestamp("2022-11-15"):
            return model_df

    source_run_dir = resolve_source_run_dir(package_dir)
    if source_run_dir is None:
        if package_prediction_path.exists():
            package_df = read_predictions(package_prediction_path)
            model_df = package_df[package_df["model"] == model_name].copy()
            if not model_df.empty:
                return model_df
        raise FileNotFoundError(f"Unable to resolve full predictions for package: {package_dir}")

    private_path = source_run_dir / "holdout_private" / "predictions_full.csv"
    core_path = source_run_dir / "reports" / "core" / "predictions.csv"
    source_path = private_path if private_path.exists() else core_path
    source_df = read_predictions(source_path)
    model_df = source_df[source_df["model"] == model_name].copy()
    if model_df.empty:
        raise ValueError(f"Model {model_name} not found in {source_path}")
    return model_df


def filter_time_window(df: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if window.start is not None:
        mask &= df["Date"] >= pd.Timestamp(window.start)
    if window.end is not None:
        mask &= df["Date"] <= pd.Timestamp(window.end)
    return df.loc[mask].copy()


def _prepare_rel_score_histogram_stats(base: np.ndarray, error_eval: np.ndarray) -> dict[str, float | np.ndarray]:
    base_abs = np.abs(np.asarray(base, dtype=float))
    error_abs = np.abs(np.asarray(error_eval, dtype=float))

    q50_base, q90_base = np.quantile(base_abs, [0.5, 0.9])
    q50_error, q90_error = np.quantile(error_abs, [0.5, 0.9])

    base_loss = float(q50_base + 0.5 * q90_base)
    abs_loss = float(q50_error + 0.5 * q90_error)
    rel_score = 1.0 - (abs_loss / base_loss) if base_loss > 0 else float("nan")

    proxy_floor = max(base_loss, 1e-4)
    proxy_denom = np.maximum(base_abs, proxy_floor)
    stabilized_proxy_rel_score = 1.0 - (error_abs / proxy_denom)
    stabilized_proxy_rel_score = np.clip(stabilized_proxy_rel_score, -1.5, 1.0)

    return {
        "stabilized_proxy_rel_score": stabilized_proxy_rel_score,
        "base_loss": base_loss,
        "abs_loss": abs_loss,
        "rel_score": float(rel_score),
        "proxy_floor": float(proxy_floor),
        "positive_share": float(np.mean(stabilized_proxy_rel_score > 0.0)),
        "hard_left_share": float(np.mean(stabilized_proxy_rel_score < -0.5)),
    }


def summarize_time_window(period_df: pd.DataFrame, window: TimeWindow) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    work = period_df.sort_values(["code", "Date"], kind="stable").copy()
    raw_row_count = int(len(work))
    empty_summary = {
        "window": window.key,
        "label": window.label,
        "date_start": window.start or "-inf",
        "date_end": window.end or "+inf",
        "raw_row_count": raw_row_count,
        "aligned_row_count": 0,
        "code_count": int(work["code"].astype(str).nunique()) if raw_row_count else 0,
        "rel_score": float("nan"),
        "directional_accuracy": float("nan"),
        "error_q2": float("nan"),
        "error_q8": float("nan"),
        "error_mean": float("nan"),
        "error_std": float("nan"),
        "proxy_mean": float("nan"),
        "proxy_median": float("nan"),
        "proxy_positive_share": float("nan"),
        "proxy_hard_left_share": float("nan"),
    }
    if raw_row_count == 0:
        return empty_summary, None

    try:
        eval_result = evaluate(
            work["prediction"].to_numpy(dtype=float),
            work["actual"].to_numpy(dtype=float),
            group_ids=work["code"].astype(str).to_numpy(),
        )
    except (ValueError, ZeroDivisionError):
        return {
            **empty_summary,
            "date_start": window.start or work["Date"].min().strftime("%Y-%m-%d"),
            "date_end": window.end or work["Date"].max().strftime("%Y-%m-%d"),
        }, None

    error_eval = np.asarray(eval_result["error"], dtype=float)
    error_gold = -error_eval
    proxy_stats = _prepare_rel_score_histogram_stats(eval_result["base"], error_eval)
    arrays = {
        "error_gold": error_gold,
        "proxy": np.asarray(proxy_stats["stabilized_proxy_rel_score"], dtype=float),
    }

    summary = {
        "window": window.key,
        "label": window.label,
        "date_start": window.start or work["Date"].min().strftime("%Y-%m-%d"),
        "date_end": window.end or work["Date"].max().strftime("%Y-%m-%d"),
        "raw_row_count": raw_row_count,
        "aligned_row_count": int(len(error_gold)),
        "code_count": int(work["code"].astype(str).nunique()),
        "rel_score": float(eval_result["rel_score"]),
        "directional_accuracy": float(eval_result["directional_accuracy"]),
        "error_q2": float(np.quantile(error_gold, 0.2)),
        "error_q8": float(np.quantile(error_gold, 0.8)),
        "error_mean": float(np.mean(error_gold)),
        "error_std": float(np.std(error_gold)),
        "proxy_mean": float(np.mean(arrays["proxy"])),
        "proxy_median": float(np.median(arrays["proxy"])),
        "proxy_positive_share": float(proxy_stats["positive_share"]),
        "proxy_hard_left_share": float(proxy_stats["hard_left_share"]),
    }
    return summary, arrays


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "model"


def build_package_plot(
    package_name: str,
    model_name: str,
    roles: tuple[str, ...],
    period_summaries: list[dict[str, object]],
    period_arrays: dict[str, dict[str, np.ndarray]],
    time_windows: tuple[TimeWindow, ...],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, len(time_windows), figsize=(6.0 * len(time_windows), 8), squeeze=False)
    colors = {
        "error": "#4e79a7",
        "proxy": "#59a14f",
        "q2": "#f28e2b",
        "q8": "#e15759",
        "zero": "#222222",
        "mean": "#d62728",
        "agg": "#1f77b4",
    }

    summary_by_key = {str(item["window"]): item for item in period_summaries}

    for idx, window in enumerate(time_windows):
        error_ax = axes[0, idx]
        proxy_ax = axes[1, idx]
        summary = summary_by_key[window.key]
        arrays = period_arrays.get(window.key)

        if arrays is None:
            error_ax.set_visible(False)
            proxy_ax.set_visible(False)
            continue

        error_values = arrays["error_gold"]
        proxy_values = arrays["proxy"]

        error_ax.hist(error_values, bins=50, color=colors["error"], alpha=0.8)
        error_ax.axvline(0.0, color=colors["zero"], linewidth=1.0, alpha=0.35)
        error_ax.axvline(float(summary["error_q2"]), color=colors["q2"], linestyle="--", linewidth=1.4, label="q2")
        error_ax.axvline(float(summary["error_q8"]), color=colors["q8"], linestyle="--", linewidth=1.4, label="q8")
        error_ax.set_title(f"{window.label} | E = prediction - actual")
        error_ax.set_xlabel("E")
        error_ax.set_ylabel("count")
        error_ax.grid(True, alpha=0.2)
        error_ax.legend(loc="upper right")
        error_ax.text(
            0.98,
            0.98,
            "\n".join(
                [
                    f"dates={summary['date_start']}..{summary['date_end']}",
                    f"codes={summary['code_count']}",
                    f"raw_rows={summary['raw_row_count']}",
                    f"aligned_rows={summary['aligned_row_count']}",
                    f"q2={summary['error_q2']:+.4f}",
                    f"q8={summary['error_q8']:+.4f}",
                    f"mean={summary['error_mean']:+.4f}",
                    f"std={summary['error_std']:.4f}",
                ]
            ),
            transform=error_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

        proxy_ax.hist(proxy_values, bins=np.linspace(-1.5, 1.0, 41), color=colors["proxy"], alpha=0.8)
        proxy_ax.axvline(0.0, color=colors["zero"], linewidth=1.0, alpha=0.35)
        proxy_ax.axvline(float(summary["proxy_mean"]), color=colors["mean"], linewidth=1.4, label="mean proxy")
        proxy_ax.axvline(float(summary["rel_score"]), color=colors["agg"], linestyle="--", linewidth=1.4, label="aggregate rel_score")
        proxy_ax.set_title(f"{window.label} | Stabilized rel_score proxy")
        proxy_ax.set_xlabel("proxy rel_score")
        proxy_ax.set_ylabel("count")
        proxy_ax.grid(True, alpha=0.2)
        proxy_ax.legend(loc="upper left")
        proxy_ax.text(
            0.98,
            0.98,
            "\n".join(
                [
                    f"rel_score={summary['rel_score']:+.4f}",
                    f"dir_acc={summary['directional_accuracy']:.4f}",
                    f"mean={summary['proxy_mean']:+.4f}",
                    f"median={summary['proxy_median']:+.4f}",
                    f"share(proxy>0)={summary['proxy_positive_share']:.1%}",
                    f"share(proxy<-0.5)={summary['proxy_hard_left_share']:.1%}",
                ]
            ),
            transform=proxy_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    role_text = ", ".join(roles)
    fig.suptitle(f"{package_name} | {model_name} | roles: {role_text}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_readme(
    output_dir: Path,
    package_results: list[dict[str, object]],
    summary_csv_path: Path,
    time_windows: tuple[TimeWindow, ...],
    include_out_sample: bool,
) -> None:
    time_window_lines = []
    for window in time_windows:
        start_text = window.start or "..."
        end_text = window.end or "..."
        time_window_lines.append(f"- `{window.key}`: `[{start_text}, {end_text}]`")

    lines = [
        "# Current Best Time-Window Histogram Report",
        "",
        "Report này gom các package đang được `gold_index.json` xem là `current_best_*`.",
        "",
        "Các cửa sổ thời gian:",
        "",
        *time_window_lines,
        "",
        "Quy ước:",
        "",
        "- `E = prediction - actual`",
        "- `rel_score` dùng cùng công thức hiện tại của repo, tính lại riêng trong từng cửa sổ",
        "- histogram `rel_score` bên dưới dùng stabilized local proxy để tránh méo mạnh khi `|actual|` quá nhỏ",
        f"- `out_sample` {'đã được bật' if include_out_sample else 'không được đưa vào report mặc định'}",
        "",
        f"Summary CSV: `{summary_csv_path.relative_to(output_dir)}`",
        "",
        "## Packages",
        "",
    ]

    for item in package_results:
        plot_rel = Path(item["plot_path"]).relative_to(output_dir)
        package_rel = Path(item["package_dir"]).relative_to(GOLD_ROOT)
        roles_text = ", ".join(item["roles"])
        lines.extend(
            [
                f"### `{item['package_name']}`",
                "",
                f"- roles: `{roles_text}`",
                f"- model: `{item['model_name']}`",
                f"- package_dir: `{package_rel}`",
                f"- plot: `{plot_rel}`",
                "",
                f"![{item['package_name']}]({plot_rel.as_posix()})",
                "",
            ]
        )

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    gold_root = args.gold_root.resolve()
    output_dir = args.output_dir.resolve()
    time_windows = resolve_time_windows(bool(args.include_out_sample))
    core_dir = output_dir / "core"
    plots_dir = output_dir / "plots"
    core_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    package_specs = load_package_specs(gold_root)
    summary_rows: list[dict[str, object]] = []
    package_results: list[dict[str, object]] = []

    for spec in package_specs:
        model_name = resolve_model_name(spec.package_dir)
        full_df = load_full_prediction_frame(spec.package_dir, model_name)
        period_summaries: list[dict[str, object]] = []
        period_arrays: dict[str, dict[str, np.ndarray]] = {}

        for window in time_windows:
            period_df = filter_time_window(full_df, window)
            summary, arrays = summarize_time_window(period_df, window)
            summary["package_name"] = spec.package_name
            summary["model_name"] = model_name
            summary["roles"] = ",".join(spec.roles)
            period_summaries.append(summary)
            if arrays is not None:
                period_arrays[window.key] = arrays
            summary_rows.append(summary)

        plot_path = plots_dir / f"{slugify(spec.package_name)}__{slugify(model_name)}__time_hist.png"
        build_package_plot(
            spec.package_name,
            model_name,
            spec.roles,
            period_summaries,
            period_arrays,
            time_windows,
            plot_path,
        )

        pd.DataFrame(period_summaries).to_csv(
            core_dir / f"{slugify(spec.package_name)}__time_summary.csv",
            index=False,
        )
        package_results.append(
            {
                "package_name": spec.package_name,
                "package_dir": str(spec.package_dir),
                "model_name": model_name,
                "roles": list(spec.roles),
                "plot_path": str(plot_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = core_dir / "current_best_time_window_summary.csv"
    summary_json_path = core_dir / "current_best_time_window_summary.json"
    summary_df.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    write_readme(
        output_dir,
        package_results,
        summary_csv_path,
        time_windows,
        bool(args.include_out_sample),
    )

    manifest = {
        "gold_root": str(gold_root),
        "output_dir": str(output_dir),
        "include_out_sample": bool(args.include_out_sample),
        "time_windows": [
            {"key": window.key, "label": window.label, "start": window.start, "end": window.end}
            for window in time_windows
        ],
        "packages": package_results,
        "summary_csv": str(summary_csv_path),
        "summary_json": str(summary_json_path),
    }
    (core_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
