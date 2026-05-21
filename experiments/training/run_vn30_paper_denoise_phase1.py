from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
LOG_BASE = RUN_BASE / "overnight_logs"
MAIN_CLI = ROOT / "main.py"
PYTHON_BIN = ROOT / "venv" / "bin" / "python"
VN30_LIST_PATH = ROOT / "market_lists" / "vn30.txt"
BASELINE_MANIFEST = RUN_BASE / "reports" / "advisor_shortlist" / "vn30_panel_allcodes_minimal_20260411" / "manifest.json"

STAMP = "20260412_paper_phase1"

BASE_FEATURES = [
    "open_level_20",
    "high_level_20",
    "low_level_20",
    "close_level_20",
    "volume_level_20",
    "open_delta_1",
    "high_delta_1",
    "low_delta_1",
    "close_delta_1",
    "volume_delta_1",
    "intraday_return",
    "gap_open",
    "close_position",
    "bb_width",
    "volume_ratio_20",
    "volatility_20",
    "momentum_5",
    "momentum_20",
    "macd_hist",
    "rsi_14",
    "vnindex_return",
    "vingroup_momentum",
    "a_d_ratio",
    "day_of_week",
]

DENOISE_FEATURES = [
    "fft_trend_gap_32",
    "wavelet_trend_gap_32",
    "savgol_trend_gap_11",
    "kalman_trend_gap",
    "fft_noise_ratio_32",
    "wavelet_noise_ratio_32",
    "savgol_noise_ratio_11",
    "kalman_noise_ratio",
    "denoise_consensus_gap",
    "denoise_method_dispersion",
]

CASES = [
    {
        "name": f"vn30_paper_repr_{STAMP}",
        "feature_phase": "paper_v1",
        "sequence_normalization": "none",
        "feature_columns": BASE_FEATURES,
    },
    {
        "name": f"vn30_paper_repr_instnorm_{STAMP}",
        "feature_phase": "paper_v1",
        "sequence_normalization": "instance_zscore",
        "feature_columns": BASE_FEATURES,
    },
    {
        "name": f"vn30_paper_denoise_{STAMP}",
        "feature_phase": "paper_denoise_v1",
        "sequence_normalization": "instance_zscore",
        "feature_columns": BASE_FEATURES + DENOISE_FEATURES,
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def load_stocks() -> str:
    stocks = [token.strip().upper() for token in VN30_LIST_PATH.read_text(encoding="utf-8").replace(",", "\n").splitlines() if token.strip()]
    stocks = [code for code in stocks if code != "VPL"]
    return ",".join(stocks)


def best_models(run_dir: Path) -> dict[str, object]:
    metrics = load_json(run_dir / "reports" / "core" / "metrics.json")
    rows: list[tuple[str, float, float]] = []
    for model_name, payload in metrics.items():
        val = payload.get("val", {}).get("rel_score")
        test = payload.get("test", {}).get("rel_score")
        if val is None or test is None:
            continue
        rows.append((model_name, float(val), float(test)))
    rows.sort(key=lambda item: (item[1], item[2]), reverse=True)
    best_by_val = rows[0] if rows else ("", float("-inf"), float("-inf"))
    best_by_test = sorted(rows, key=lambda item: item[2], reverse=True)[0] if rows else ("", float("-inf"), float("-inf"))
    return {
        "best_by_val_model": best_by_val[0],
        "best_by_val_rel_score": best_by_val[1],
        "best_by_val_test_rel_score": best_by_val[2],
        "best_by_test_model": best_by_test[0],
        "best_test_rel_score": best_by_test[2],
        "best_by_test_val_rel_score": best_by_test[1],
    }


def build_cmd(case: dict[str, object], stocks: str) -> list[str]:
    return [
        str(PYTHON_BIN),
        str(MAIN_CLI),
        "train",
        "--run-name",
        str(case["name"]),
        "--target-mode",
        "return",
        "--loss",
        "rel_score",
        "--stocks",
        stocks,
        "--feature-selection-mode",
        "sector_config",
        "--feature-columns",
        ",".join(case["feature_columns"]),
        "--feature-phase",
        str(case["feature_phase"]),
        "--sequence-normalization",
        str(case["sequence_normalization"]),
        "--window-size",
        "15",
        "--lstm-units",
        "64,32",
        "--dropout",
        "0.05",
        "--lr",
        "0.0005",
        "--batch-size",
        "64",
        "--epochs",
        "24",
        "--patience",
        "7",
        "--target-normalizer",
        "volatility_20",
        "--lstm-seeds",
        "42,52,62",
        "--sample-weight-mode",
        "magnitude",
    ]


def main() -> None:
    log_dir = LOG_BASE / f"{STAMP}_vn30_paper_phase1"
    log_dir.mkdir(parents=True, exist_ok=True)
    stocks = load_stocks()
    baseline = load_json(BASELINE_MANIFEST)

    rows: list[dict[str, object]] = [
        {
            "case_name": "baseline_existing_panel",
            "run_name": Path(baseline["run_dir"]).name,
            "sequence_normalization": "none",
            "feature_phase": "none",
            "best_by_val_model": baseline["model"],
            "best_by_val_rel_score": None,
            "best_by_val_test_rel_score": float(baseline["filtered_metrics"]["test_rel_score"]),
            "best_by_test_model": baseline["model"],
            "best_test_rel_score": float(baseline["filtered_metrics"]["test_rel_score"]),
            "best_by_test_val_rel_score": None,
            "log_path": "",
        }
    ]

    for case in CASES:
        run_name = str(case["name"])
        log_path = log_dir / f"{run_name}.log"
        run_cmd(build_cmd(case, stocks), log_path)
        best = best_models(RUN_BASE / run_name)
        rows.append(
            {
                "case_name": run_name,
                "run_name": run_name,
                "sequence_normalization": case["sequence_normalization"],
                "feature_phase": case["feature_phase"],
                **best,
                "feature_count": len(case["feature_columns"]),
                "log_path": str(log_path),
            }
        )

    summary_path = log_dir / "vn30_paper_phase1_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
