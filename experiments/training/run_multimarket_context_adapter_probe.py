"""Multi-market context adapter probe.

Lightweight framework step before expensive LSTM training:
- build common portable OHLCV features,
- add market-level context features available for VN/US/JP,
- evaluate simple Ridge baselines on raw and cross-sectionally demeaned alpha.

Holdout/test is not used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402

OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_context_adapter_probe_20260525"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/multimarket_context_adapter_probe_20260525"
TRAIN_END = "2020-03-31"
VAL_START = "2020-04-01"
VAL_END = "2022-11-15"
FOLD_DAYS = 21
MARKETS = {
    "VN": ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv",
    "US": ROOT / "data/processed/assets/data_info_us/history/us_gold_recommended.csv",
    "JP": ROOT / "data/processed/assets/data_info_jp/history/jp_gold_recommended.csv",
}
PORTABLE_FEATURES = (
    "close_return",
    "adjust_return",
    "range_pct",
    "body_pct",
    "volume_change",
    "momentum_5",
    "momentum_20",
    "volatility_5",
    "volatility_20",
    "ma_5_gap",
    "ma_20_gap",
)
CONTEXT_FEATURES = (
    "market_close_return",
    "market_return_5",
    "market_return_20",
    "market_volatility_20",
    "market_breadth_pos_ratio",
    "market_abs_return",
    "cs_momentum_20_rank",
    "cs_volatility_20_rank",
    "cs_ma_20_gap_rank",
    "cs_volume_change_rank",
)


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def metric(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    base = robust_loss(actual)
    abs_err = np.abs(err)
    return {
        "n": int(len(actual)),
        "rel_score": float(1.0 - robust_loss(err) / base) if base > 0 else float("nan"),
        "absE_robust": robust_loss(err),
        "base_robust": base,
        "absE_q90": float(np.quantile(abs_err, 0.9)) if len(abs_err) else float("nan"),
        "DA": float(np.mean(np.sign(actual) == np.sign(pred))) if len(actual) else float("nan"),
        "pred_actual_q90_ratio": (
            float(np.quantile(np.abs(pred), 0.9) / max(np.quantile(np.abs(actual), 0.9), 1e-8))
            if len(actual) else float("nan")
        ),
    }


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    daily = out.groupby("Date").agg(
        market_close_return=("close_return", "mean"),
        market_breadth_pos_ratio=("close_return", lambda series: float((series > 0).mean())),
    ).sort_index()
    daily["market_return_5"] = daily["market_close_return"].rolling(5, min_periods=2).mean()
    daily["market_return_20"] = daily["market_close_return"].rolling(20, min_periods=5).mean()
    daily["market_volatility_20"] = daily["market_close_return"].rolling(20, min_periods=5).std()
    daily["market_abs_return"] = daily["market_close_return"].abs()
    for col in daily.columns:
        out[col] = out["Date"].map(daily[col])
    out["cs_momentum_20_rank"] = out.groupby("Date")["momentum_20"].rank(pct=True)
    out["cs_volatility_20_rank"] = out.groupby("Date")["volatility_20"].rank(pct=True)
    out["cs_ma_20_gap_rank"] = out.groupby("Date")["ma_20_gap"].rank(pct=True)
    out["cs_volume_change_rank"] = out.groupby("Date")["volume_change"].rank(pct=True)
    return out


def fold_ids(dates: pd.Series) -> np.ndarray:
    unique_dates = pd.Series(pd.to_datetime(dates).unique()).sort_values().reset_index(drop=True)
    date_to_pos = {date: idx for idx, date in enumerate(unique_dates)}
    return pd.to_datetime(dates).map(date_to_pos).to_numpy(dtype=int) // FOLD_DAYS


def demean_by_date(df: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmp = df[["Date", "target_next_return"]].copy()
    tmp["pred"] = pred
    tmp["actual_alpha"] = tmp["target_next_return"] - tmp.groupby("Date")["target_next_return"].transform("mean")
    tmp["pred_alpha"] = tmp["pred"] - tmp.groupby("Date")["pred"].transform("mean")
    return tmp["actual_alpha"].to_numpy(dtype=np.float32), tmp["pred_alpha"].to_numpy(dtype=np.float32)


def ridge_predict(train: pd.DataFrame, val: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, fit_intercept=True, random_state=0))
    model.fit(train[list(features)].to_numpy(dtype=np.float32), train["target_next_return"].to_numpy(dtype=np.float32))
    return model.predict(val[list(features)].to_numpy(dtype=np.float32)).astype(np.float32)


def evaluate_market(market: str, path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = load_training_frame(path, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = add_context_features(frame)
    required = ["Date", "code", "target_next_return", *PORTABLE_FEATURES, *CONTEXT_FEATURES]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{market} missing {missing}")
    frame = frame[required].replace([np.inf, -np.inf], np.nan).dropna(subset=["target_next_return", *PORTABLE_FEATURES, *CONTEXT_FEATURES])
    train = frame[frame["Date"] <= TRAIN_END].copy()
    val = frame[(frame["Date"] >= VAL_START) & (frame["Date"] <= VAL_END)].copy()

    predictions = {
        "ridge_portable": ridge_predict(train, val, PORTABLE_FEATURES),
        "ridge_context_adapter": ridge_predict(train, val, (*PORTABLE_FEATURES, *CONTEXT_FEATURES)),
        "zero": np.zeros(len(val), dtype=np.float32),
    }
    actual = val["target_next_return"].to_numpy(dtype=np.float32)
    overall_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    ids = fold_ids(val["Date"])
    for model, pred in predictions.items():
        row = metric(actual, pred)
        alpha_actual, alpha_pred = demean_by_date(val, pred)
        alpha_row = metric(alpha_actual, alpha_pred)
        row.update(
            {
                "market": market,
                "model": model,
                "alpha_rel_score": alpha_row["rel_score"],
                "alpha_absE_robust": alpha_row["absE_robust"],
                "train_rows": len(train),
                "val_rows": len(val),
                "n_codes": val["code"].nunique(),
            }
        )
        overall_rows.append(row)
        for fold_id in sorted(set(ids)):
            mask = ids == fold_id
            fold = metric(actual[mask], pred[mask])
            fa, fp = demean_by_date(val.loc[mask], pred[mask])
            alpha_fold = metric(fa, fp)
            dates = val.loc[mask, "Date"]
            fold.update(
                {
                    "market": market,
                    "model": model,
                    "fold_id": int(fold_id),
                    "test_start": dates.min().date().isoformat(),
                    "test_end": dates.max().date().isoformat(),
                    "alpha_rel_score": alpha_fold["rel_score"],
                    "alpha_absE_robust": alpha_fold["absE_robust"],
                }
            )
            fold_rows.append(fold)
    return overall_rows, fold_rows


def bootstrap_context_vs_portable(folds: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for market, group in folds.groupby("market"):
        pivot = group.pivot(index="fold_id", columns="model", values=metric_col)
        joined = pivot[["ridge_context_adapter", "ridge_portable"]].dropna()
        diff = (joined["ridge_context_adapter"] - joined["ridge_portable"]).to_numpy(float)
        idx = rng.integers(0, len(diff), size=(20000, len(diff)))
        boot = diff[idx].mean(axis=1)
        rows.append(
            {
                "market": market,
                "metric": metric_col,
                "n_folds": int(len(diff)),
                "mean_delta_context_vs_portable": float(diff.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "p_boot_delta_le_0": float(np.mean(boot <= 0)),
                "positive_delta_folds": int(np.sum(diff > 0)),
            }
        )
    return pd.DataFrame(rows)


def build_report(overall: pd.DataFrame, sig: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Multi-Market Context Adapter Probe",
            "",
            "Protocol: common portable features + market context adapter, train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.",
            "",
            "## Overall Metrics",
            "",
            overall.round(6).to_markdown(index=False),
            "",
            "## Paired Bootstrap: Context Adapter vs Portable Ridge",
            "",
            sig.round(6).to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- This is a lightweight adapter probe, not the final LSTM model.",
            "- Positive regular rel_score with negative alpha_rel_score indicates market-drift capture, not stock-selection skill.",
            "- Use this to decide which context features should enter the next heteroscedastic ensemble run.",
        ]
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for market, path in MARKETS.items():
        rows, folds = evaluate_market(market, path)
        overall_rows.extend(rows)
        fold_rows.extend(folds)
    overall = pd.DataFrame(overall_rows).sort_values(["market", "rel_score"], ascending=[True, False])
    folds = pd.DataFrame(fold_rows)
    sig = pd.concat(
        [
            bootstrap_context_vs_portable(folds, "rel_score"),
            bootstrap_context_vs_portable(folds, "alpha_rel_score"),
        ],
        ignore_index=True,
    )
    for df, name in [(overall, "overall_metrics.csv"), (folds, "fold_metrics.csv"), (sig, "bootstrap_context_vs_portable.csv")]:
        df.to_csv(OUTPUT / name, index=False)
        df.to_csv(GOLD / name, index=False)
    report = build_report(overall, sig)
    (OUTPUT / "context_adapter_report.md").write_text(report, encoding="utf-8")
    (GOLD / "context_adapter_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
