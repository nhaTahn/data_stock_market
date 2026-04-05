from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_pipeline.market_config import TRAIN_START_DATE, get_market_config
from src.data_pipeline.quality_dataset_core import build_market_quality_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build US quality dataset.")
    parser.add_argument("--train-start-date", default=TRAIN_START_DATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_market_config("US", train_start_date=args.train_start_date)
    build_market_quality_dataset(config)
    print("Saved:", config.output_dir / f"{config.output_prefix}_gold_recommended.csv")


if __name__ == "__main__":
    main()
