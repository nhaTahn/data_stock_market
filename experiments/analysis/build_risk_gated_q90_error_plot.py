"""Build risk-gated q90(|E|) plot to inspect safer accepted market days."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training import evaluate_meta_ensemble_calibration as meta  # noqa: E402
from experiments.training import evaluate_regime_calibration as regime  # noqa: E402

DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
VN100_SYMBOLS = ROOT / "data/external/zInfo/data_info_vn/vn100_symbols.csv"
OUTPUT_DIR = ROOT / "gold/vn_transition_pressure_20260512/plots/teacher_style_abs_error_vn100_insample"
REPORT_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/teacher_style_abs_error_vn100_insample"


def read_symbols(path: Path) -> set[str]:
    df = pd.read_csv(path)
    col = "symbol" if "symbol" in df.columns else "code"
    return set(df[col].dropna().astype(str).str.upper())


def build_index_proxy() -> pd.DataFrame:
    symbols = read_symbols(VN100_SYMBOLS)
    raw = pd.read_csv(DATA, usecols=["Date", "code", "adjust"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = raw[raw["code"].isin(symbols)].sort_values(["code", "Date"], kind="stable")
    raw["ret"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    idx = raw.groupby("Date", sort=True)["ret"].mean().rename("index_proxy_return").reset_index()
    idx["index_proxy"] = (1 + idx["index_proxy_return"].fillna(0)).cumprod()
    idx["index_proxy_rebased"] = idx["index_proxy"] / idx["index_proxy"].dropna().iloc[0] * 100
    return idx


def build_daily_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    ytr, yv, ptr, pv, sigtr, sigv = meta.load_anchor_seed_predictions()
    dtr, dv = regime.load_dates()
    market = regime.build_market_features(dtr, dv, ytr, yv)
    rvtr = market["train"]["vol10"]
    rvv = market["val"]["vol10"]
    qtr = market["train"]["q905"]
    qv = market["val"]["q905"]
    btr = ptr.mean(axis=1)
    bv = pv.mean(axis=1)
    rve, qe, grid = regime.fit_2d_scales(ytr, btr, rvtr, qtr)
    regtr = regime.apply_2d_scales(btr, rvtr, qtr, rve, qe, grid)
    regv = regime.apply_2d_scales(bv, rvv, qv, rve, qe, grid)
    xtr = meta.make_meta_features(ptr, sigtr, rvtr, qtr)
    xv = meta.make_meta_features(pv, sigv, rvv, qv)
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, learning_rate=0.03, max_leaf_nodes=8, l2_regularization=0.2, random_state=43)
    model.fit(xtr, ytr)
    mt = np.asarray(model.predict(xtr), dtype=np.float32)
    mv = np.asarray(model.predict(xv), dtype=np.float32)
    _, blendtr, blendv = meta.train_selected_blend(ytr, regtr, mt, yv, regv, mv)
    train = pd.DataFrame({"Date": pd.to_datetime(dtr), "abs_error": np.abs(ytr - blendtr), "vol10": rvtr, "q905": qtr})
    val = pd.DataFrame({"Date": pd.to_datetime(dv), "abs_error": np.abs(yv - blendv), "vol10": rvv, "q905": qv})
    def daily(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("Date", sort=True).agg(
            n_stocks=("abs_error", "count"),
            q90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))),
            median_abs_error=("abs_error", "median"),
            vol10=("vol10", "mean"),
            q905=("q905", "mean"),
        ).reset_index()
    return daily(train), daily(val)


def add_train_rank_risk(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    val = val.copy()
    train["risk_score"] = 0.5 * train["vol10"].rank(pct=True) + 0.5 * train["q905"].rank(pct=True)
    train_vol = np.sort(train["vol10"].to_numpy())
    train_q = np.sort(train["q905"].to_numpy())
    val["risk_score"] = (
        0.5 * np.searchsorted(train_vol, val["vol10"].to_numpy(), side="right") / len(train_vol)
        + 0.5 * np.searchsorted(train_q, val["q905"].to_numpy(), side="right") / len(train_q)
    )
    return train, val


def plot_gated(val: pd.DataFrame, index: pd.DataFrame, threshold: float, output_path: Path) -> pd.DataFrame:
    frame = val.merge(index[["Date", "index_proxy_rebased"]], on="Date", how="left").sort_values("Date").reset_index(drop=True)
    frame["accepted"] = frame["risk_score"] <= threshold
    frame["year"] = frame["Date"].dt.year
    years = sorted(frame["year"].unique())
    fig, axes = plt.subplots(len(years), 1, figsize=(14, 3.7 * len(years)), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, year in zip(axes, years):
        part = frame[frame["year"].eq(year)].reset_index(drop=True)
        x = np.arange(len(part))
        ax2 = ax.twinx()
        ax.plot(x, part["index_proxy_rebased"], color="#1f8bb6", linewidth=1.25, label="VN100")
        rejected = ~part["accepted"]
        if rejected.any():
            ax.scatter(x[rejected], part.loc[rejected, "index_proxy_rebased"], color="#9ca3af", s=9, alpha=0.55, label="abstain days")
        ax2.plot(x, part["q90_abs_error"] * 100, color="#e63946", linestyle="--", linewidth=1.0, alpha=0.35, label="q90 full")
        acc = part["accepted"]
        ax2.scatter(x[acc], part.loc[acc, "q90_abs_error"] * 100, color="#e63946", s=14, alpha=0.9, label="q90 accepted")
        ax2.axhline(3.5, color="#991b1b", linestyle=":", linewidth=0.8)
        ax2.axhline(5.0, color="#7f1d1d", linestyle="-.", linewidth=0.8)
        accepted_part = part[acc]
        n_acc = int(acc.sum())
        med = float(accepted_part["q90_abs_error"].median() * 100) if n_acc else float("nan")
        p90 = float(accepted_part["q90_abs_error"].quantile(0.90) * 100) if n_acc else float("nan")
        gt35 = int((accepted_part["q90_abs_error"] > 0.035).sum()) if n_acc else 0
        ax.set_title(f"{year} accepted={n_acc}/{len(part)} | med q90={med:.2f}%, p90={p90:.2f}%, >3.5%={gt35}", loc="left", fontsize=9, fontweight="bold")
        ax.grid(alpha=0.20)
        ax.set_xlabel("Trading day")
        ax.tick_params(axis="y", labelcolor="#1f8bb6")
        ax2.tick_params(axis="y", labelcolor="#e63946")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[0].twinx().get_legend_handles_labels()
    fig.suptitle(f"Risk-gated VN100 vs q90(|E|) — accepted days only, threshold={threshold:.3f}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train, val = build_daily_predictions()
    train, val = add_train_rank_risk(train, val)
    threshold = float(np.quantile(train["risk_score"], 0.50))
    index = build_index_proxy()
    for output_dir in [OUTPUT_DIR, REPORT_DIR]:
        frame = plot_gated(val, index, threshold, output_dir / "risk_gated_vn100_q90_abs_error_by_year_validation.png")
        frame.to_csv(output_dir / "risk_gated_vn100_q90_abs_error_by_year_validation.csv", index=False)
        accepted = frame[frame["accepted"]]
        summary = frame.groupby("year").agg(
            n_days=("Date", "nunique"),
            accepted_days=("accepted", "sum"),
            full_median_q90=("q90_abs_error", "median"),
            full_days_gt_3p5=("q90_abs_error", lambda values: int((values > 0.035).sum())),
        ).reset_index()
        acc_summary = accepted.groupby("year").agg(
            accepted_median_q90=("q90_abs_error", "median"),
            accepted_p90_q90=("q90_abs_error", lambda values: float(np.quantile(values, 0.90))),
            accepted_days_gt_3p5=("q90_abs_error", lambda values: int((values > 0.035).sum())),
            accepted_days_gt_5=("q90_abs_error", lambda values: int((values > 0.05).sum())),
        ).reset_index()
        summary = summary.merge(acc_summary, on="year", how="left")
        summary.to_csv(output_dir / "risk_gated_vn100_q90_abs_error_by_year_validation_summary.csv", index=False)
    print(summary.round(6).to_markdown(index=False))
    print(OUTPUT_DIR / "risk_gated_vn100_q90_abs_error_by_year_validation.png")


if __name__ == "__main__":
    main()
