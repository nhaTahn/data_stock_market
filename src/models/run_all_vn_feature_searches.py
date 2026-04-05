from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.search_feature_combinations import build_run_dir, override_config, run_feature_search
from src.models.summarize_vn_sector_features import main as summarize_sector_main
from src.models.train_lstm import load_frame
from src.utils.vn_sector import DEFAULT_UNIVERSE_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature search for all VN stocks and refresh sector summary.")
    parser.add_argument("--universe-path", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--target-mode", choices=["price", "growth", "return", "return_3d", "return_5d"], default="return")
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--val-end-date", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--min-combo-size", type=int, default=1)
    parser.add_argument("--max-combo-size", type=int, default=2)
    parser.add_argument("--min-rel-score", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_stocks(universe_path: Path) -> list[str]:
    df = pd.read_csv(universe_path, usecols=["code"]).dropna().drop_duplicates().sort_values("code")
    return df["code"].tolist()


def run_single_stock(stock: str, args: argparse.Namespace) -> dict[str, object]:
    cli_args = argparse.Namespace(
        data_path=None,
        stocks=stock,
        target_mode=args.target_mode,
        train_end_date=args.train_end_date,
        val_end_date=args.val_end_date,
        window_size=args.window_size,
        feature_columns=args.feature_columns,
        min_combo_size=args.min_combo_size,
        max_combo_size=args.max_combo_size,
        min_rel_score=args.min_rel_score,
        top_k=args.top_k,
        run_name=f"search_{stock.lower()}_expanded",
    )
    config = override_config(cli_args)
    run_dir = config.output_dir / cli_args.run_name
    if run_dir.exists() and not args.overwrite:
        with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
            return json.load(f)
    run_dir = build_run_dir(config.output_dir, cli_args.run_name, config.target_mode)
    df = load_frame(config.data_path, stock)
    _, _, summary = run_feature_search(
        df,
        stocks=stock,
        config=config,
        min_combo_size=args.min_combo_size,
        max_combo_size=args.max_combo_size,
        min_rel_score=args.min_rel_score,
        top_k=args.top_k,
        run_dir=run_dir,
    )
    return summary


def main() -> None:
    args = parse_args()
    stocks = load_stocks(args.universe_path)
    rows: list[dict[str, object]] = []
    total = len(stocks)
    for idx, stock in enumerate(stocks, start=1):
        summary = run_single_stock(stock, args)
        rows.append(summary)
        print(f"[{idx}/{total}] {stock}: best={summary['best_features']} test={summary['best_test_rel_score']:.6f}")

    out = pd.DataFrame(rows).sort_values(["best_test_rel_score", "best_val_rel_score"], ascending=[False, False])
    out_path = ROOT / "data" / "assets" / "data_info_vn" / "history" / "training_runs" / "vn_all_stock_search_summary.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    summarize_sector_main([])


if __name__ == "__main__":
    main()
