import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_pipeline.market_config import TRAIN_START_DATE, get_market_config
from src.data_pipeline.quality_dataset_core import build_market_quality_dataset


def load_code_list(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return tuple(line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quality dataset for a market.")
    parser.add_argument("--market", required=True, help="Market code (VN, US, JP, etc.)")
    parser.add_argument("--train-start-date", default=TRAIN_START_DATE)
    parser.add_argument("--force-include-codes-path", type=Path, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = get_market_config(
        args.market,
        train_start_date=args.train_start_date,
        force_include_codes=load_code_list(args.force_include_codes_path),
    )
    print(f"Building quality dataset for {args.market}...")
    build_market_quality_dataset(config)
    print("Saved:", config.output_dir / f"{config.output_prefix}_gold_recommended.csv")
    print("Saved:", config.output_dir / f"{config.output_prefix}_quality_dataset.csv")

if __name__ == "__main__":
    main()
