from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

from src.utils.vn_sector import DEFAULT_INDUSTRY_PATH, DEFAULT_UNIVERSE_PATH, build_vn_stock_sector_map


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize VN stock sectors and feature search results.")
    parser.add_argument("--universe-path", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--industry-path", type=Path, default=DEFAULT_INDUSTRY_PATH)
    parser.add_argument(
        "--training-runs-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs",
    )
    parser.add_argument("--top-combos-per-stock", type=int, default=10)
    parser.add_argument("--top-features", type=int, default=5)
    parser.add_argument("--top-combos", type=int, default=3)
    return parser.parse_args(argv)


def load_single_stock_runs(training_runs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(training_runs_dir.glob("search_*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        stock = summary.get("stocks")
        if not isinstance(stock, str) or "," in stock:
            continue
        results_path = summary_path.parent / "feature_search_results.csv"
        if not results_path.exists():
            continue
        results = pd.read_csv(results_path)
        if results.empty:
            continue
        best_test = results.sort_values(["test_rel_score", "val_rel_score"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "run_name": summary_path.parent.name,
                "stock": stock,
                "best_by_val": summary.get("best_features"),
                "best_val_rel_score": summary.get("best_val_rel_score"),
                "best_by_test": best_test["features"],
                "best_test_rel_score": float(best_test["test_rel_score"]),
                "best_test_val_rel_score": float(best_test["val_rel_score"]),
            }
        )
    return pd.DataFrame(rows)


def build_feature_rows(training_runs_dir: Path, stocks: list[str], top_combos_per_stock: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stock in stocks:
        run_dir = training_runs_dir / f"search_{stock.lower()}_expanded"
        results_path = run_dir / "feature_search_results.csv"
        if not results_path.exists():
            continue
        results = pd.read_csv(results_path)
        top = results.sort_values(["test_rel_score", "val_rel_score"], ascending=[False, False]).head(top_combos_per_stock)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            combo = row["features"]
            for feature in combo.split(","):
                rows.append(
                    {
                        "stock": stock,
                        "combo_rank_in_stock": rank,
                        "combo": combo,
                        "feature": feature,
                        "test_rel_score": float(row["test_rel_score"]),
                        "val_rel_score": float(row["val_rel_score"]),
                    }
                )
    return pd.DataFrame(rows)


def format_feature_list(df: pd.DataFrame, top_n: int) -> str:
    if df.empty:
        return ""
    top = df.sort_values(["stocks_hit", "appearances", "max_test_rel_score"], ascending=[False, False, False]).head(top_n)
    return " | ".join(
        f"{row.feature} ({int(row.stocks_hit)} mã, {int(row.appearances)} lần, max={row.max_test_rel_score:.3f})"
        for row in top.itertuples()
    )


def format_combo_list(df: pd.DataFrame, top_n: int) -> str:
    if df.empty:
        return ""
    top = df.sort_values(["test_rel_score", "val_rel_score"], ascending=[False, False]).head(top_n)
    return " | ".join(
        f"{row.stock}:{row.combo} (test={row.test_rel_score:.3f}, val={row.val_rel_score:.3f})"
        for row in top.itertuples()
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    training_runs_dir = args.training_runs_dir

    stock_sector_map = build_vn_stock_sector_map(args.universe_path, args.industry_path)
    stock_sector_map.to_csv(training_runs_dir.parent / "vn_stock_sector_map.csv", index=False)
    stock_sector_map_stock = stock_sector_map.rename(columns={"code": "stock"})

    stock_run_summary = load_single_stock_runs(training_runs_dir)
    if stock_run_summary.empty:
        stock_run_summary = stock_sector_map_stock.copy()
    else:
        stock_run_summary = stock_sector_map_stock.merge(stock_run_summary, on="stock", how="left")
    stock_run_summary.to_csv(training_runs_dir / "stock_sector_search_summary.csv", index=False)

    searched_stocks = stock_run_summary.loc[stock_run_summary["best_by_test"].notna(), "stock"].dropna().tolist()
    feature_rows = build_feature_rows(training_runs_dir, searched_stocks, args.top_combos_per_stock)
    if feature_rows.empty:
        raise ValueError("No single-stock search runs found to summarize.")

    feature_rows = stock_sector_map_stock.merge(feature_rows, on="stock", how="inner")
    feature_rows.to_csv(training_runs_dir / "sector_feature_long.csv", index=False)

    feature_rollup = (
        feature_rows.groupby(["sector", "feature"], as_index=False)
        .agg(
            stocks_hit=("stock", "nunique"),
            appearances=("feature", "size"),
            mean_test_rel_score=("test_rel_score", "mean"),
            max_test_rel_score=("test_rel_score", "max"),
        )
        .sort_values(["sector", "stocks_hit", "appearances", "max_test_rel_score"], ascending=[True, False, False, False])
    )
    feature_rollup.to_csv(training_runs_dir / "sector_feature_top5_long.csv", index=False)

    combo_rollup = (
        feature_rows.loc[:, ["sector", "stock", "combo", "test_rel_score", "val_rel_score"]]
        .drop_duplicates()
        .sort_values(["sector", "test_rel_score", "val_rel_score"], ascending=[True, False, False])
    )
    combo_rollup.to_csv(training_runs_dir / "sector_combo_top3_long.csv", index=False)

    sector_counts = (
        stock_sector_map.groupby("sector", as_index=False)
        .agg(
            universe_stock_count=("code", "nunique"),
            stocks=("code", lambda s: ",".join(sorted(s))),
        )
        .rename(columns={"code": "stock"})
    )

    searched_sector_counts = (
        feature_rows.groupby("sector", as_index=False)
        .agg(
            searched_stock_count=("stock", "nunique"),
            searched_stocks=("stock", lambda s: ",".join(sorted(set(s)))),
        )
    )

    sector_summary_rows: list[dict[str, object]] = []
    for sector in sector_counts["sector"]:
        sector_features = feature_rollup[feature_rollup["sector"] == sector]
        sector_combos = combo_rollup[combo_rollup["sector"] == sector]
        sector_summary_rows.append(
            {
                "sector": sector,
                "top_5_features": format_feature_list(sector_features, args.top_features),
                "top_3_combos": format_combo_list(sector_combos, args.top_combos),
            }
        )

    sector_summary = sector_counts.merge(searched_sector_counts, on="sector", how="left").merge(
        pd.DataFrame(sector_summary_rows),
        on="sector",
        how="left",
    )
    sector_summary["searched_stock_count"] = sector_summary["searched_stock_count"].fillna(0).astype(int)
    sector_summary["searched_stocks"] = sector_summary["searched_stocks"].fillna("")
    sector_summary["top_5_features"] = sector_summary["top_5_features"].fillna("")
    sector_summary["top_3_combos"] = sector_summary["top_3_combos"].fillna("")
    sector_summary = sector_summary.sort_values(["searched_stock_count", "universe_stock_count", "sector"], ascending=[False, False, True])
    sector_summary.to_csv(training_runs_dir / "sector_feature_final_summary.csv", index=False)

    print("Saved:", training_runs_dir.parent / "vn_stock_sector_map.csv")
    print("Saved:", training_runs_dir / "stock_sector_search_summary.csv")
    print("Saved:", training_runs_dir / "sector_feature_top5_long.csv")
    print("Saved:", training_runs_dir / "sector_combo_top3_long.csv")
    print("Saved:", training_runs_dir / "sector_feature_final_summary.csv")
    print(sector_summary.loc[:, ["sector", "searched_stock_count", "searched_stocks", "top_5_features", "top_3_combos"]].to_string(index=False))


if __name__ == "__main__":
    main()
