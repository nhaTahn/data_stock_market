"""
analyze_feature_correlation.py
==============================
Phân tích tương quan giữa các feature đầu vào để:
  1. Phát hiện feature pairs bị redundant (|corr| > ngưỡng)
  2. Gợi ý feature set không bị multicollinear cho LSTM
  3. Xuất heatmap PNG và CSV summary

Dùng cho sector cụ thể hoặc toàn bộ data.

Cách chạy:
  # Phân tích sector Bất động sản
  python -m experiments.analysis.analyze_feature_correlation --sector "Bất động sản"

  # Phân tích tất cả sector (mỗi sector 1 output folder)
  python -m experiments.analysis.analyze_feature_correlation --all-sectors

  # Phân tích 1 nhóm mã cụ thể
  python -m experiments.analysis.analyze_feature_correlation --stocks BCM,DIG,KOS,NLG,VHM,VIC

  # Đặt ngưỡng correlation khác (mặc định 0.75)
  python -m experiments.analysis.analyze_feature_correlation --sector "Bất động sản" --threshold 0.70
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from src.models.config import ALL_FEATURE_COLUMNS, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR
from src.models.training import split_frame_by_date
from src.utils.features import ensure_columns
from src.utils.vn_sector import build_vn_stock_sector_map

# ─────────────────────────────────────────────
# Sector → danh sách mã (từ logs overnight)
# ─────────────────────────────────────────────
SECTOR_STOCKS: dict[str, list[str]] = {
    "Bất động sản": [
        "BCM", "DIG", "DXG", "HDC", "HDG", "KBC", "KDH", "KOS",
        "NLG", "PDR", "SIP", "SJS", "SZC", "TCH", "VHM", "VIC", "VPI", "VRE",
    ],
    "Ngân hàng": [
        "ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB",
        "SHB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB",
    ],
    "Thực phẩm và đồ uống": [
        "ANV", "DBC", "HAG", "KDC", "MSN", "PAN", "SAB", "SBT", "VHC", "VNM",
    ],
    "Dịch vụ tài chính": [
        "BSI", "CTS", "EVF", "FTS", "HCM", "SSI", "VCI", "VIX", "VND",
    ],
    "Xây dựng và Vật liệu": [
        "BMP", "CII", "CTD", "CTR", "HHV", "HT1", "PC1", "VCG", "VGC",
    ],
    "Điện, nước & xăng dầu khí đốt": [
        "BWE", "GAS", "NT2", "POW", "PPC", "REE",
    ],
    "Hàng & Dịch vụ Công nghiệp": [
        "GEX", "GMD", "PVT", "VSC", "VTP",
    ],
    "Hóa chất": [
        "DCM", "DGC", "DPM", "GVR", "PHR",
    ],
    "Tài nguyên Cơ bản": [
        "HPG", "HSG", "NKG", "PTB",
    ],
    "Bán lẻ": [
        "DGW", "FRT", "MWG",
    ],
    "Công nghệ Thông tin": [
        "CMG", "FPT",
    ],
    "Du lịch và Giải trí": [
        "SCS", "VJC",
    ],
    "Dầu khí": [
        "PLX", "PVD",
    ],
    "Hàng cá nhân & Gia dụng": [
        "PNJ", "TLG",
    ],
    "Bảo hiểm": ["BVH"],
    "Y tế": ["IMP"],
}

# Semantic groups để đề xuất feature nên drop
SEMANTIC_GROUPS: dict[str, list[str]] = {
    "MA_price_gap": ["ma_5_gap", "ma_10_gap", "ma_20_gap", "ma_50_gap", "ma_200_gap", "ma_20_ma_200_gap"],
    "Bollinger": ["bb_position", "bb_zscore", "bb_width", "bb_mid_20"],
    "Momentum": ["momentum_5", "momentum_20", "return_3", "return_10", "adjust_return"],
    "Volume_ratio": ["volume_ratio_5", "volume_ratio_20", "volume_zscore_20", "volume_change"],
    "Volatility": ["volatility_5", "volatility_10", "volatility_20", "volatility_ratio", "atr_gap", "true_range"],
    "Price_range": ["range_pct", "close_position", "upper_shadow", "lower_shadow", "high_close_gap", "close_low_gap", "body_pct"],
    "OBV_volume": ["obv", "obv_change", "signed_volume"],
    "VWAP": ["vwap_gap", "vwap_proxy", "vwap_gap_20", "vwap_20"],
    "MACD": ["macd", "macd_signal", "macd_hist"],
    "Wyckoff": ["effort_result_ratio", "buying_pressure", "selling_pressure", "wyckoff_phase_60d"],
}


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def load_and_filter(
    data_path: Path,
    stocks: list[str] | None,
    train_end_date: str,
    val_end_date: str,
) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    if "code" not in df.columns:
        raise ValueError("Cột 'code' không tồn tại trong data file.")
    if stocks:
        df = df[df["code"].isin(stocks)].copy()
        missing = set(stocks) - set(df["code"].unique())
        if missing:
            print(f"  ⚠️  Không tìm thấy mã: {sorted(missing)} — bỏ qua")
    if df.empty:
        raise ValueError("DataFrame rỗng sau khi filter stocks.")
    df = ensure_columns(df)
    # Dùng toàn bộ data (train+val+test) để tính correlation
    return df


def select_available_features(df: pd.DataFrame, feature_cols: tuple[str, ...]) -> list[str]:
    return [c for c in feature_cols if c in df.columns and df[c].notna().sum() > 30]


def compute_correlation_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sub = df[features].copy().replace([np.inf, -np.inf], np.nan)
    return sub.corr(method="pearson")


def find_redundant_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.75,
) -> list[dict[str, object]]:
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "pearson_corr": round(float(val), 4),
                    "abs_corr": round(abs(float(val)), 4),
                })
    return sorted(pairs, key=lambda x: x["abs_corr"], reverse=True)


def suggest_keep_features(
    features: list[str],
    redundant_pairs: list[dict[str, object]],
) -> tuple[list[str], list[str]]:
    """
    Greedy: Duyệt từng pair redundant, bỏ feature thứ 2 nếu feature thứ 1 đã được giữ.
    Ưu tiên giữ features xuất hiện nhiều hơn trong overnight search results.
    """
    # Độ ưu tiên: features nằm trong top overnight recommendations
    priority_order = [
        "vwap_gap", "bb_width", "volatility_20", "gap_open", "intraday_return",
        "close_position", "obv_change", "momentum_5", "momentum_20", "above_ma_200",
        "lower_shadow", "upper_shadow", "atr_gap", "ma_200_gap", "ma_20_gap",
        "volume_ratio_20", "volume_zscore_20", "rsi_14", "macd_hist", "macd",
        "bb_position", "rolling_max_20_gap", "wyckoff_phase_60d", "effort_result_ratio",
        "buying_pressure", "selling_pressure", "vwap_gap_20", "alpha_sector",
        "vnindex_return", "vingroup_momentum", "a_d_ratio", "day_of_week",
    ]
    # Sắp xếp features theo priority
    feature_set = set(features)
    ordered = [f for f in priority_order if f in feature_set]
    ordered += [f for f in features if f not in ordered]

    to_drop: set[str] = set()
    for pair in redundant_pairs:
        fa, fb = pair["feature_a"], pair["feature_b"]
        # Nếu cả hai chưa bị drop, drop cái có priority thấp hơn
        if fa not in to_drop and fb not in to_drop:
            rank_a = ordered.index(fa) if fa in ordered else 9999
            rank_b = ordered.index(fb) if fb in ordered else 9999
            if rank_a <= rank_b:
                to_drop.add(fb)
            else:
                to_drop.add(fa)

    keep = [f for f in ordered if f not in to_drop]
    drop = [f for f in ordered if f in to_drop]
    return keep, drop


def plot_heatmap(
    corr_matrix: pd.DataFrame,
    out_path: Path,
    title: str,
    threshold: float = 0.75,
) -> None:
    n = len(corr_matrix)
    fig_size = max(12, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    masked = corr_matrix.copy()
    im = ax.imshow(masked.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr_matrix.index, fontsize=7)

    # Highlight ô vượt ngưỡng
    for i in range(n):
        for j in range(n):
            if i != j and abs(masked.values[i, j]) >= threshold:
                val = masked.values[i, j]
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=5,
                    color="white" if abs(val) > 0.9 else "black",
                    fontweight="bold",
                )

    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Heatmap → {out_path}")


def plot_missing_rate(df: pd.DataFrame, features: list[str], out_path: Path, title: str) -> None:
    """Bar chart tỷ lệ NaN của từng feature."""
    rates = {f: df[f].isna().mean() * 100 for f in features}
    rates = dict(sorted(rates.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.35), 5))
    colors = ["#e74c3c" if v > 30 else "#f39c12" if v > 10 else "#2ecc71" for v in rates.values()]
    ax.bar(range(len(rates)), list(rates.values()), color=colors)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(list(rates.keys()), rotation=90, fontsize=7)
    ax.set_ylabel("NaN %")
    ax.set_title(title, fontsize=11)
    ax.axhline(y=30, color="red", linestyle="--", alpha=0.5, label="30% threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Missing rate → {out_path}")


def analyze_sector(
    sector_name: str,
    stocks: list[str],
    data_path: Path,
    out_dir: Path,
    train_end_date: str,
    val_end_date: str,
    threshold: float,
    feature_pool: tuple[str, ...],
) -> dict[str, object]:
    print(f"\n{'='*60}")
    print(f"🏢 Sector: {sector_name} ({len(stocks)} mã: {', '.join(stocks)})")
    print(f"{'='*60}")

    sector_dir = out_dir / sector_name.replace(" ", "_").replace("&", "and").replace(",", "")
    sector_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df = load_and_filter(data_path, stocks, train_end_date, val_end_date)
    except Exception as e:
        print(f"  ❌ Lỗi load data: {e}")
        return {"sector": sector_name, "error": str(e)}

    actual_stocks = sorted(df["code"].unique().tolist())
    print(f"  ✅ Loaded {len(df):,} rows, {len(actual_stocks)} mã: {actual_stocks}")

    # Chọn features có sẵn
    features = select_available_features(df, feature_pool)
    print(f"  📋 Features available: {len(features)}/{len(feature_pool)}")

    if len(features) < 2:
        print("  ⚠️  Quá ít features. Bỏ qua sector này.")
        return {"sector": sector_name, "error": "too_few_features"}

    # Missing rate chart
    plot_missing_rate(
        df, features,
        out_path=sector_dir / "missing_rate.png",
        title=f"Missing rate — {sector_name}",
    )

    # Drop features bị NaN > 50%
    high_missing = [f for f in features if df[f].isna().mean() > 0.5]
    if high_missing:
        print(f"  ⚠️  Drop features NaN>50%: {high_missing}")
    features = [f for f in features if f not in high_missing]

    # Tính correlation
    corr = compute_correlation_matrix(df, features)

    # Heatmap toàn bộ
    plot_heatmap(
        corr,
        out_path=sector_dir / "correlation_heatmap_full.png",
        title=f"Correlation Matrix — {sector_name} (all {len(features)} features)",
        threshold=threshold,
    )

    # Redundant pairs
    redundant = find_redundant_pairs(corr, threshold=threshold)
    print(f"  🔴 Redundant pairs (|corr| ≥ {threshold}): {len(redundant)}")
    for p in redundant[:10]:
        print(f"       {p['feature_a']:30s} ↔ {p['feature_b']:30s}  corr={p['pearson_corr']:+.3f}")
    if len(redundant) > 10:
        print(f"       ... và {len(redundant)-10} cặp khác")

    # Gợi ý keep/drop
    keep, drop = suggest_keep_features(features, redundant)
    print(f"\n  ✅ Giữ lại ({len(keep)} features):")
    for f in keep:
        print(f"       ✓ {f}")
    print(f"\n  ❌ Nên bỏ ({len(drop)} features) do redundant:")
    for f in drop:
        print(f"       ✗ {f}")

    # Heatmap sau pruning
    if len(keep) >= 2:
        corr_pruned = compute_correlation_matrix(df, keep)
        plot_heatmap(
            corr_pruned,
            out_path=sector_dir / "correlation_heatmap_pruned.png",
            title=f"Correlation Matrix (after pruning) — {sector_name} ({len(keep)} features)",
            threshold=threshold,
        )

    # Semantic group analysis
    print(f"\n  📦 Phân tích theo nhóm semantic:")
    group_analysis = {}
    for group_name, group_features in SEMANTIC_GROUPS.items():
        present = [f for f in group_features if f in features]
        if len(present) < 2:
            continue
        group_corr = corr.loc[present, present]
        max_corr = group_corr.where(
            np.triu(np.ones(group_corr.shape), k=1).astype(bool)
        ).stack().abs().max()
        group_analysis[group_name] = {
            "features_present": present,
            "max_internal_corr": round(float(max_corr) if not np.isnan(max_corr) else 0.0, 3),
        }
        is_high = max_corr >= threshold if not np.isnan(max_corr) else False
        flag = "🔴" if is_high else "🟢"
        print(f"       {flag} {group_name}: {present} | max_corr={max_corr:.3f}")

    # Lưu CSV redundant pairs
    if redundant:
        pd.DataFrame(redundant).to_csv(sector_dir / "redundant_pairs.csv", index=False)

    # Lưu feature recommendation
    recommendation = {
        "sector": sector_name,
        "stocks_found": actual_stocks,
        "threshold": threshold,
        "total_features_available": len(features),
        "features_to_keep": keep,
        "features_to_drop": drop,
        "redundant_pairs_count": len(redundant),
        "semantic_group_analysis": group_analysis,
        "suggested_lstm_config_snippet": {
            "sector_features": {sector_name: keep},
        },
    }
    rec_path = sector_dir / "feature_recommendation.json"
    with rec_path.open("w", encoding="utf-8") as f:
        json.dump(recommendation, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Recommendation → {rec_path}")

    # Print config snippet
    print(f"\n  📝 Snippet gợi ý cho lstm_config.json:")
    print(f'     "sector_features": {{')
    print(f'       "{sector_name}": {json.dumps(keep, ensure_ascii=False)}')
    print(f'     }}')

    return recommendation


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phân tích tương quan features theo sector để pruning trước khi train LSTM."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "feature_correlation")
    parser.add_argument(
        "--sector",
        default="Bất động sản",
        help="Tên sector cụ thể cần phân tích (mặc định: 'Bất động sản').",
    )
    parser.add_argument(
        "--stocks",
        default=None,
        help="Danh sách mã tùy chọn, phân cách bằng dấu phẩy (ghi đè --sector).",
    )
    parser.add_argument(
        "--all-sectors",
        action="store_true",
        help="Phân tích tất cả các sector.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Ngưỡng |correlation| để coi là redundant (mặc định: 0.75).",
    )
    parser.add_argument("--train-end-date", default="2023-12-31")
    parser.add_argument("--val-end-date", default="2024-12-31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Feature pool: dùng ALL_FEATURE_COLUMNS từ config
    feature_pool = ALL_FEATURE_COLUMNS

    print(f"🔍 Phân tích Feature Correlation")
    print(f"   Data path : {args.data_path}")
    print(f"   Output dir: {args.out_dir}")
    print(f"   Threshold : |corr| ≥ {args.threshold}")
    print(f"   Feature pool size: {len(feature_pool)}")

    if args.stocks:
        # Mode: danh sách mã tùy chọn
        stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
        analyze_sector(
            sector_name="custom_group",
            stocks=stocks,
            data_path=args.data_path,
            out_dir=args.out_dir,
            train_end_date=args.train_end_date,
            val_end_date=args.val_end_date,
            threshold=args.threshold,
            feature_pool=feature_pool,
        )
    elif args.all_sectors:
        # Mode: tất cả sector
        all_results = []
        for sector_name, stocks in SECTOR_STOCKS.items():
            result = analyze_sector(
                sector_name=sector_name,
                stocks=stocks,
                data_path=args.data_path,
                out_dir=args.out_dir,
                train_end_date=args.train_end_date,
                val_end_date=args.val_end_date,
                threshold=args.threshold,
                feature_pool=feature_pool,
            )
            all_results.append(result)

        # Tổng hợp summary
        summary_rows = []
        for r in all_results:
            if "error" in r:
                continue
            summary_rows.append({
                "sector": r["sector"],
                "features_to_keep": len(r.get("features_to_keep", [])),
                "features_to_drop": len(r.get("features_to_drop", [])),
                "redundant_pairs": r.get("redundant_pairs_count", 0),
                "keep_list": ",".join(r.get("features_to_keep", [])),
            })
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = args.out_dir / "all_sectors_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n✅ Summary tất cả sector → {summary_path}")
    else:
        # Mode: sector đơn (mặc định)
        sector_name = args.sector
        if sector_name not in SECTOR_STOCKS:
            available = list(SECTOR_STOCKS.keys())
            print(f"❌ Sector '{sector_name}' không tồn tại. Các sector available:")
            for s in available:
                print(f"   - {s}")
            sys.exit(1)
        stocks = SECTOR_STOCKS[sector_name]
        analyze_sector(
            sector_name=sector_name,
            stocks=stocks,
            data_path=args.data_path,
            out_dir=args.out_dir,
            train_end_date=args.train_end_date,
            val_end_date=args.val_end_date,
            threshold=args.threshold,
            feature_pool=feature_pool,
        )


if __name__ == "__main__":
    main()
