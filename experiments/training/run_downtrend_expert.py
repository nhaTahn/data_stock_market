import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Downtrend Expert.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test with 1 seed and 1 epoch.")
    parser.add_argument("--run-name", default="downtrend_expert_phase_ic_sector19", help="Run name.")
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    
    phase_ic_sector19 = (
        "open_delta_1", "low_delta_1", "volume_delta_1", "low_level_20", "close_level_20", 
        "intraday_return", "gap_open", "close_position", "bb_width", "volume_ratio_20", 
        "momentum_20", "macd_hist", "sector_positive_ratio", "sector_ad_ratio", 
        "sector_momentum_rank", "sector_momentum_20", "relative_sector_momentum_20", 
        "sector_return", "alpha_sector"
    )
    
    python_bin = ROOT / "venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path("python")
        
    cmd = [
        str(python_bin), str(ROOT / "main.py"), "train",
        "--run-name", args.run_name,
        "--feature-columns", ",".join(phase_ic_sector19),
        "--regime-filter", "downtrend",
        "--feature-phase", "paper_v1"
    ]
    
    if args.smoke_test:
        cmd.extend([
            "--lstm-seeds", "42",
            "--epochs", "2",
        ])
    else:
        cmd.extend([
            "--lstm-seeds", "42,52,62,72,82",
        ])
        
    print(f"Running Downtrend Expert with command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
