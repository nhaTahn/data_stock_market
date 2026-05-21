from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
LOG_BASE = RUN_BASE / "overnight_logs"
MAIN_CLI = ROOT / "main.py"
PYTHON_BIN = ROOT / "venv" / "bin" / "python"

FAMILY_PREFIXES = (
    "lstm",
    "lstm_attention",
    "lstm_pcie_lite",
    "lstm_signmag",
)


@dataclass(frozen=True)
class UniversePreset:
    name: str
    stocks: tuple[str, ...]
    note: str


UNIVERSE_PRESETS = {
    "bds": UniversePreset(
        name="bds",
        stocks=("KOS", "DXG", "NLG", "DIG", "TCH", "VHM"),
        note="Existing BDS edge set with known positive sector-level rel_score history.",
    ),
    "bank": UniversePreset(
        name="bank",
        stocks=("ACB", "BID", "CTG", "MBB", "TCB", "VCB"),
        note="VN30 bank core for clearer sector regime structure.",
    ),
    "vin": UniversePreset(
        name="vin",
        stocks=("VIC", "VHM", "VRE"),
        note="Vingroup cluster with strong shared path dependence.",
    ),
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run narrow PCIE-lite phase 1 experiments.")
    parser.add_argument("--stamp", default="20260417_pcie_lite_phase1")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lstm-seeds", default="42,52,62")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--sample-weight-mode", choices=["none", "magnitude"], default="magnitude")
    parser.add_argument("--pcie-lite-future-steps", default="3,5")
    parser.add_argument("--pcie-lite-patch-lengths", default="5,10")
    parser.add_argument("--pcie-lite-patch-stride", type=int, default=5)
    parser.add_argument("--pcie-lite-patch-dim", type=int, default=16)
    parser.add_argument(
        "--pcie-lite-base-columns",
        default="open,high,low,close,volume",
    )
    parser.add_argument(
        "--pcie-lite-context-columns",
        default="vnindex_return,market_leader_return,a_d_ratio,day_of_week",
    )
    parser.add_argument("--include-universe", action="append", choices=sorted(UNIVERSE_PRESETS.keys()), default=None)
    parser.add_argument("--disable-attention-family", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-case", action="append", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def model_matches_prefix(model_name: str, family_prefix: str) -> bool:
    return model_name == family_prefix or model_name.startswith(f"{family_prefix}_")


def summarize_family(metrics: dict[str, dict[str, dict[str, float]]], family_prefix: str) -> dict[str, object]:
    rows: list[tuple[str, float, float]] = []
    for model_name, payload in metrics.items():
        if not model_matches_prefix(model_name, family_prefix):
            continue
        val_rel = payload.get("val", {}).get("rel_score")
        test_rel = payload.get("test", {}).get("rel_score")
        if val_rel is None or test_rel is None:
            continue
        rows.append((model_name, float(val_rel), float(test_rel)))
    if not rows:
        return {
            "best_by_val_model": None,
            "best_by_val_rel_score": None,
            "best_by_val_test_rel_score": None,
            "best_by_test_model": None,
            "best_test_rel_score": None,
            "best_by_test_val_rel_score": None,
        }
    rows.sort(key=lambda item: (item[1], item[2]), reverse=True)
    best_by_val = rows[0]
    best_by_test = sorted(rows, key=lambda item: item[2], reverse=True)[0]
    return {
        "best_by_val_model": best_by_val[0],
        "best_by_val_rel_score": best_by_val[1],
        "best_by_val_test_rel_score": best_by_val[2],
        "best_by_test_model": best_by_test[0],
        "best_test_rel_score": best_by_test[2],
        "best_by_test_val_rel_score": best_by_test[1],
    }


def summarize_run(run_dir: Path) -> dict[str, object]:
    metrics = load_json(run_dir / "reports" / "core" / "metrics.json")
    config = load_json(run_dir / "reports" / "core" / "config.json")
    summary: dict[str, object] = {
        "run_name": run_dir.name,
        "stocks": config.get("stocks"),
        "window_size": config.get("window_size"),
        "pcie_lite_future_steps": config.get("lstm_pcie_lite_future_steps"),
        "pcie_lite_patch_length": config.get("lstm_pcie_lite_patch_length"),
        "pcie_lite_patch_stride": config.get("lstm_pcie_lite_patch_stride"),
        "pcie_lite_patch_dim": config.get("lstm_pcie_lite_patch_dim"),
    }
    for family_prefix in FAMILY_PREFIXES:
        family_summary = summarize_family(metrics, family_prefix)
        for key, value in family_summary.items():
            summary[f"{family_prefix}_{key}"] = value
    pcie_test = summary.get("lstm_pcie_lite_best_test_rel_score")
    lstm_test = summary.get("lstm_best_test_rel_score")
    if pcie_test is not None and lstm_test is not None:
        summary["pcie_gain_vs_lstm_best_test"] = float(pcie_test) - float(lstm_test)
    else:
        summary["pcie_gain_vs_lstm_best_test"] = None
    return summary


def build_cases(args: argparse.Namespace) -> list[tuple[str, UniversePreset, int, int]]:
    selected_universes = args.include_universe or ["bds", "bank", "vin"]
    future_steps = parse_csv_ints(args.pcie_lite_future_steps)
    patch_lengths = parse_csv_ints(args.pcie_lite_patch_lengths)
    cases: list[tuple[str, UniversePreset, int, int]] = []
    for universe_name in selected_universes:
        preset = UNIVERSE_PRESETS[universe_name]
        for future_step in future_steps:
            for patch_length in patch_lengths:
                case_name = f"{preset.name}_h{future_step}_p{patch_length}"
                cases.append((case_name, preset, future_step, patch_length))
    if args.only_case:
        allowed = set(parse_csv_strings(",".join(args.only_case)))
        cases = [case for case in cases if case[0] in allowed]
    if not cases:
        raise ValueError("No experiment cases selected.")
    return cases


def build_cmd(args: argparse.Namespace, run_name: str, preset: UniversePreset, future_steps: int, patch_length: int) -> list[str]:
    cmd = [
        str(PYTHON_BIN),
        str(MAIN_CLI),
        "train",
        "--run-name",
        run_name,
        "--target-mode",
        "return",
        "--loss",
        "rel_score",
        "--stocks",
        ",".join(preset.stocks),
        "--feature-selection-mode",
        "sector_config",
        "--window-size",
        str(args.window_size),
        "--lstm-units",
        args.lstm_units,
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--target-normalizer",
        args.target_normalizer,
        "--lstm-seeds",
        args.lstm_seeds,
        "--sample-weight-mode",
        args.sample_weight_mode,
        "--enable-pcie-lite-family",
        "--pcie-lite-base-columns",
        args.pcie_lite_base_columns,
        "--pcie-lite-context-columns",
        args.pcie_lite_context_columns,
        "--pcie-lite-future-steps",
        str(future_steps),
        "--pcie-lite-patch-length",
        str(patch_length),
        "--pcie-lite-patch-stride",
        str(args.pcie_lite_patch_stride),
        "--pcie-lite-patch-dim",
        str(args.pcie_lite_patch_dim),
    ]
    if not args.disable_attention_family:
        cmd.append("--enable-attention-family")
    return cmd


def main() -> None:
    args = parse_args()
    log_dir = LOG_BASE / f"{args.stamp}_pcie_lite_phase1"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for case_name, preset, future_steps, patch_length in build_cases(args):
        run_name = f"{case_name}_{args.stamp}"
        log_path = log_dir / f"{run_name}.log"
        cmd = build_cmd(args, run_name, preset, future_steps, patch_length)
        run_cmd(cmd, log_path, args.dry_run)

        row: dict[str, object] = {
            "case_name": case_name,
            "run_name": run_name,
            "universe": preset.name,
            "stocks": ",".join(preset.stocks),
            "universe_note": preset.note,
            "pcie_lite_future_steps": future_steps,
            "pcie_lite_patch_length": patch_length,
            "pcie_lite_patch_stride": args.pcie_lite_patch_stride,
            "window_size": args.window_size,
            "log_path": str(log_path),
            "dry_run": bool(args.dry_run),
        }
        if not args.dry_run:
            row.update(summarize_run(RUN_BASE / run_name))
        rows.append(row)

    summary = {
        "summary_path": str(log_dir / "pcie_lite_phase1_summary.json"),
        "cases": rows,
    }
    (log_dir / "pcie_lite_phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
