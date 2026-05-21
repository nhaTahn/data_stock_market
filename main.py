from __future__ import annotations

import argparse
import sys


def _add_data_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    data_parser = subparsers.add_parser("data", help="Data pipeline commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_subparsers.add_parser("build", help="Build a market quality dataset.")
    data_subparsers.add_parser("fetch", help="Fetch raw market history.")


def _add_train_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    subparsers.add_parser("train", help="Train forecasting models.")


def _add_report_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    report_parser = subparsers.add_parser("report", help="Reporting commands.")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)
    report_subparsers.add_parser("update-run", help="Rebuild a saved run report.")
    report_subparsers.add_parser("feature-formulas", help="Write the feature formula reference report.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for data_stock_market.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_data_commands(subparsers)
    _add_train_command(subparsers)
    _add_report_commands(subparsers)
    return parser


def _dispatch_data(command: str, argv: list[str]) -> None:
    if command == "build":
        from src.data_pipeline.build_dataset import main as build_main

        build_main(argv)
        return
    if command == "fetch":
        from src.data_pipeline.fetch_cli import main as fetch_main

        fetch_main(argv)
        return
    raise ValueError(f"Unsupported data command: {command}")


def _dispatch_report(command: str, argv: list[str]) -> None:
    if command == "update-run":
        from src.reporting.update_run_reports import main as update_run_reports_main

        update_run_reports_main(argv)
        return
    if command == "feature-formulas":
        from src.reporting.write_feature_report import main as write_feature_report_main

        write_feature_report_main(argv)
        return
    raise ValueError(f"Unsupported report command: {command}")


def main(argv: list[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] in {"-h", "--help"}:
        build_parser().parse_args(args_list)
        return

    if args_list[0] == "data" and len(args_list) >= 2:
        _dispatch_data(args_list[1], args_list[2:])
        return
    if args_list[0] == "train":
        from src.models.training.pipeline import main as train_main

        train_main(args_list[1:])
        return
    if args_list[0] == "report" and len(args_list) >= 2:
        _dispatch_report(args_list[1], args_list[2:])
        return

    build_parser().parse_args(args_list)


if __name__ == "__main__":
    main()
