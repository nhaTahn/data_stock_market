from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.training.feature_normalization import add_multimarket_feature_normalization


class FeatureNormalizationTest(unittest.TestCase):
    def test_multimarket_views_are_grouped_by_market_and_feature_type(self) -> None:
        rows: list[dict[str, object]] = []
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        for market in ("VN", "US"):
            market_offset = 100.0 if market == "US" else 0.0
            for code_index, code in enumerate(("AAA", "BBB", "CCC")):
                for day_index, date in enumerate(dates):
                    rows.append(
                        {
                            "market": market,
                            "code": f"{market}:{code}",
                            "Date": date,
                            "momentum_20": market_offset + code_index + day_index * 0.1,
                            "volatility_20": 0.01 * (code_index + 1) + day_index * 0.001,
                            "vnindex_return": 0.001 * day_index + (0.01 if market == "US" else 0.0),
                            "day_of_week": float(day_index % 5),
                            "sector_momentum_rank_pct": code_index / 2.0,
                            "is_top_2_sector": float(code_index < 2),
                        }
                    )
        frame = pd.DataFrame(rows)

        result = add_multimarket_feature_normalization(
            frame,
            (
                "momentum_20",
                "volatility_20",
                "vnindex_return",
                "day_of_week",
                "sector_momentum_rank_pct",
                "is_top_2_sector",
            ),
            rolling_window=3,
            min_periods=2,
        )

        expected_columns = {
            "momentum_20_roll_z",
            "momentum_20_cs_z",
            "momentum_20_cs_rank",
            "volatility_20_roll_z",
            "volatility_20_cs_z",
            "volatility_20_cs_rank",
            "vnindex_return_market_roll_z",
            "day_of_week_sin",
            "day_of_week_cos",
            "sector_momentum_rank_pct",
            "is_top_2_sector",
        }
        self.assertTrue(expected_columns.issubset(set(result.feature_columns)))
        self.assertNotIn("vnindex_return_cs_z", result.feature_columns)

        last_date = dates[-1]
        for market in ("VN", "US"):
            part = result.frame[(result.frame["market"] == market) & (result.frame["Date"] == last_date)]
            ranks = part.sort_values("momentum_20")["momentum_20_cs_rank"].to_numpy()
            np.testing.assert_allclose(ranks, np.array([0.0, 0.5, 1.0]))

        market_context = result.frame[
            (result.frame["market"] == "VN") & (result.frame["Date"] == last_date)
        ]["vnindex_return_market_roll_z"]
        self.assertEqual(int(market_context.nunique(dropna=False)), 1)


if __name__ == "__main__":
    unittest.main()
