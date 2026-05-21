from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIVERSE_PATH = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_INDUSTRY_PATH = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "symbols_by_industries.csv"


def load_industry_reference(industry_path: Path = DEFAULT_INDUSTRY_PATH) -> pd.DataFrame:
    df = pd.read_csv(industry_path)
    return df.rename(
        columns={
            "symbol": "code",
            "organ_name": "company_name",
            "icb_name2": "sector",
            "icb_name3": "industry",
            "icb_name4": "sub_industry",
        }
    )


def build_vn_stock_sector_map(
    universe_path: Path = DEFAULT_UNIVERSE_PATH,
    industry_path: Path = DEFAULT_INDUSTRY_PATH,
) -> pd.DataFrame:
    universe = pd.read_csv(universe_path, usecols=["code"]).dropna().drop_duplicates().sort_values("code")
    industry = load_industry_reference(industry_path)
    cols = ["code", "company_name", "sector", "industry", "sub_industry", "com_type_code"]
    merged = universe.merge(industry.loc[:, cols], on="code", how="left")
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["industry"] = merged["industry"].fillna("Unknown")
    merged["sub_industry"] = merged["sub_industry"].fillna("Unknown")
    return merged
