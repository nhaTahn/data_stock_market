from __future__ import annotations

import shutil
from pathlib import Path


REPORT_SUBDIRS = {
    "core": "core",
    "plots": "plots",
    "metric_series": "metric_series",
    "benchmark": "benchmark",
    "backtests": "backtests",
}


def ensure_report_dirs(run_dir: Path) -> dict[str, Path]:
    root = run_dir / "reports"
    root.mkdir(parents=True, exist_ok=True)
    paths = {"root": root}
    for key, name in REPORT_SUBDIRS.items():
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = path
    return paths


def clear_report_files(run_dir: Path, buckets: tuple[str, ...]) -> None:
    report_dirs = ensure_report_dirs(run_dir)
    for bucket in buckets:
        bucket_dir = report_dirs[bucket]
        for path in bucket_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)


def cleanup_report_noise(run_dir: Path) -> None:
    report_dirs = ensure_report_dirs(run_dir)
    core_dir = report_dirs["core"]
    for path in core_dir.iterdir():
        if not path.is_file():
            continue
        if (
            path.name.startswith("history_seed_")
            or path.name.startswith("history_quantile_seed_")
            or path.name.startswith("history_attention_seed_")
            or path.name.startswith("history_event_seed_")
            or path.name.startswith("history_signmag_seed_")
        ):
            path.unlink(missing_ok=True)


def report_plot_path(run_dir: Path, filename: str) -> Path:
    return ensure_report_dirs(run_dir)["plots"] / filename


def report_core_path(run_dir: Path, filename: str) -> Path:
    return ensure_report_dirs(run_dir)["core"] / filename


def report_metric_series_path(run_dir: Path, filename: str) -> Path:
    return ensure_report_dirs(run_dir)["metric_series"] / filename


def report_benchmark_path(run_dir: Path, filename: str) -> Path:
    return ensure_report_dirs(run_dir)["benchmark"] / filename


def report_backtest_path(run_dir: Path, filename: str) -> Path:
    return ensure_report_dirs(run_dir)["backtests"] / filename


def resolve_run_artifact(run_dir: Path, filename: str, bucket: str = "core") -> Path:
    bucket_map = {
        "core": report_core_path,
        "plots": report_plot_path,
        "metric_series": report_metric_series_path,
        "benchmark": report_benchmark_path,
        "backtests": report_backtest_path,
    }
    candidate = bucket_map[bucket](run_dir, filename)
    if candidate.exists():
        return candidate
    legacy = run_dir / filename
    return legacy


def cleanup_legacy_report_artifacts(run_dir: Path) -> None:
    keep_suffixes = {".keras", ".joblib"}
    keep_names = {
        "feature_scaler.npz",
        "target_scaler.npz",
        "linear_regression.joblib",
    }
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix in keep_suffixes or path.name in keep_names:
            continue
        if (
            path.name.startswith("model")
            or path.name.startswith("history_seed_")
            or path.name.startswith("history_quantile_seed_")
            or path.name.startswith("history_attention_seed_")
            or path.name.startswith("history_event_seed_")
            or path.name.startswith("history_signmag_seed_")
        ):
            continue
        path.unlink(missing_ok=True)


def mirror_run_artifacts(run_dir: Path) -> None:
    report_dirs = ensure_report_dirs(run_dir)
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        if (
            path.name.startswith("history_seed_")
            or path.name.startswith("history_quantile_seed_")
            or path.name.startswith("history_attention_seed_")
            or path.name.startswith("history_event_seed_")
            or path.name.startswith("history_signmag_seed_")
        ):
            continue
        target_dir = None
        if path.name.startswith("benchmark_fischer_krauss"):
            target_dir = report_dirs["benchmark"]
        elif path.name.startswith("metric_series_"):
            target_dir = report_dirs["metric_series"]
        elif path.suffix == ".png":
            target_dir = report_dirs["plots"]
        elif path.name.startswith("threshold_backtest") or path.name.startswith("strategy_backtest"):
            target_dir = report_dirs["backtests"]
        elif path.suffix in {".json", ".csv"}:
            target_dir = report_dirs["core"]
        if target_dir is None:
            continue
        target = target_dir / path.name
        try:
            shutil.copy2(path, target)
        except shutil.SameFileError:
            continue
