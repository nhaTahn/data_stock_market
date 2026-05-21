from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
MAIN_CLI = ROOT / "main.py"
PYTHON_BIN = ROOT / "venv" / "bin" / "python"


@dataclass(frozen=True)
class ExpertCase:
    name: str
    sector: str
    stocks: str
    window_size: int
    lstm_units: str
    lr: float
    dropout: float
    sample_weight_mode: str
    seeds: str
    epochs: int = 24
    patience: int = 7

    @property
    def current_link_name(self) -> str:
        return f"vn30gold_expert_{self.name}_current"


EXPERT_CASES: tuple[ExpertCase, ...] = (
    ExpertCase(
        name="fnb",
        sector="Thực phẩm và đồ uống",
        stocks="KDC,SAB,SBT,VNM",
        window_size=5,
        lstm_units="64,32",
        lr=0.0005,
        dropout=0.05,
        sample_weight_mode="magnitude",
        seeds="42,52,62,82",
    ),
    ExpertCase(
        name="bds",
        sector="Bất động sản",
        stocks="KOS,DXG,NLG,DIG,TCH,VHM",
        window_size=20,
        lstm_units="64,32",
        lr=0.0005,
        dropout=0.05,
        sample_weight_mode="magnitude",
        seeds="42,52,62,82",
    ),
    ExpertCase(
        name="bank",
        sector="Ngân hàng",
        stocks="VCB,TCB,CTG,BID,ACB,MBB",
        window_size=15,
        lstm_units="64,32",
        lr=0.0005,
        dropout=0.05,
        sample_weight_mode="magnitude",
        seeds="42,52,62,82",
    ),
    ExpertCase(
        name="finance",
        sector="Dịch vụ tài chính",
        stocks="BSI,CTS,EVF,FTS,HCM,SSI,VCI,VIX,VND",
        window_size=15,
        lstm_units="64,32",
        lr=0.0005,
        dropout=0.05,
        sample_weight_mode="magnitude",
        seeds="42,52,62,82",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VN30 gold expert runs and refresh symlinks.")
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_symlink(link_path: Path, target_dir: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            shutil.rmtree(link_path)
    link_path.symlink_to(target_dir.name)


def build_train_cmd(case: ExpertCase, run_name: str) -> list[str]:
    return [
        str(PYTHON_BIN),
        str(MAIN_CLI),
        "train",
        "--run-name",
        run_name,
        "--target-mode",
        "return",
        "--loss",
        "rel_score",
        "--sector",
        case.sector,
        "--stocks",
        case.stocks,
        "--feature-selection-mode",
        "sector_config",
        "--window-size",
        str(case.window_size),
        "--lstm-units",
        case.lstm_units,
        "--dropout",
        str(case.dropout),
        "--lr",
        str(case.lr),
        "--batch-size",
        "64",
        "--epochs",
        str(case.epochs),
        "--patience",
        str(case.patience),
        "--target-normalizer",
        "volatility_20",
        "--lstm-seeds",
        case.seeds,
        "--sample-weight-mode",
        case.sample_weight_mode,
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for case in EXPERT_CASES:
        run_name = f"vn30gold_expert_{case.name}_{args.stamp}"
        run_dir = RUN_BASE / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_cmd(build_train_cmd(case, run_name))

        current_link = RUN_BASE / case.current_link_name
        refresh_symlink(current_link, run_dir)

        metrics = read_json(run_dir / "reports" / "core" / "metrics.json")
        family = read_json(run_dir / "reports" / "core" / "family_selection_summary.json")
        best_plain = family.get("lstm", {}).get("best_by_val")
        best_signmag = family.get("lstm_signmag", {}).get("best_by_val")
        rows.append(
            {
                "expert": case.name,
                "sector": case.sector,
                "stocks": case.stocks,
                "run_name": run_name,
                "current_link": case.current_link_name,
                "best_plain": best_plain,
                "best_plain_test_rel_score": metrics.get(best_plain, {}).get("test", {}).get("rel_score"),
                "best_signmag": best_signmag,
                "best_signmag_test_rel_score": metrics.get(best_signmag, {}).get("test", {}).get("rel_score"),
            }
        )

    csv_path = args.output_dir / "vn30_gold_expert_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "expert",
                "sector",
                "stocks",
                "run_name",
                "current_link",
                "best_plain",
                "best_plain_test_rel_score",
                "best_signmag",
                "best_signmag_test_rel_score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = args.output_dir / "vn30_gold_expert_manifest.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_csv": str(csv_path), "manifest_json": str(json_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
