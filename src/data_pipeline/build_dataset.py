from __future__ import annotations

import argparse
from pathlib import Path

from src.data_pipeline.market_config import TRAIN_START_DATE, get_market_config
from src.data_pipeline.quality_dataset_core import build_market_quality_dataset


def load_code_list(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return tuple(line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def add_build_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--market", required=True, help="Market code (VN, US, JP, etc.)")
    parser.add_argument("--train-start-date", default=TRAIN_START_DATE)
    parser.add_argument("--force-include-codes-path", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build quality dataset for a market.")
    add_build_dataset_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_dataset_command(args: argparse.Namespace) -> None:
    config = get_market_config(
        args.market,
        train_start_date=args.train_start_date,
        force_include_codes=load_code_list(args.force_include_codes_path),
    )
    print(f"Building quality dataset for {args.market}...")
    build_market_quality_dataset(config)
    print("Saved:", config.output_dir / f"{config.output_prefix}_gold_recommended.csv")
    print("Saved:", config.output_dir / f"{config.output_prefix}_quality_dataset.csv")


def main(argv: list[str] | None = None) -> None:
    build_dataset_command(parse_args(argv))


if __name__ == "__main__":
    main()
