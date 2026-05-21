from __future__ import annotations

import argparse
from pathlib import Path

from src.models.config import ALL_FEATURE_COLUMNS
from src.reporting.feature_report import write_feature_formula_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the feature formula report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/feature_formula_report.md"),
        help="Output markdown path for the feature formula reference.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Optional comma-separated feature list. Defaults to the current configured feature universe.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    feature_columns = (
        tuple(item.strip() for item in args.features.split(",") if item.strip())
        if args.features
        else tuple(ALL_FEATURE_COLUMNS)
    )
    output_path = write_feature_formula_report(args.output, feature_columns)
    print(output_path)


if __name__ == "__main__":
    main()
