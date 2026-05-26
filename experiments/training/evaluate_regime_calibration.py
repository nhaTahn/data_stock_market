"""Regime-conditional calibration for VN calibrated ensemble predictions.

Key insight: daily realized volatility (lagged 5 days) has 0.76 correlation
with prediction error Q90. By learning per-volatility-decile scale multipliers
on train data we improve both rel_score and tail metrics.

Protocol: train-selected; validation readout only. Holdout/test not used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.evaluate_fixed_train_relscore_calibration import choose_train_calibration  # noqa
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset  # noqa
from src.models.training.pipeline import load_frame as load_training_frame  # noqa
from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler  # noqa

PRED_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_combined_full5_20260521"
DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/regime_calibration_20260527"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/regime_calibration_20260527"
SEEDS = (43, 52, 62, 71, 82)
N_DECILES = 10


# ── helpers ────────────────────────────────────────────────────────────────────

def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def rel_score(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[mask], pred[mask]
    base = robust_loss(actual)
    return float(1.0 - robust_loss(actual - pred) / base) if base > 0 else float("nan")


def metrics(actual: np.ndarray, pred: np.ndarray, dates: pd.Series) -> dict:
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred, dates = actual[mask], pred[mask], dates[mask]
    err = actual - pred
    daily_q90 = (
        pd.DataFrame({"Date": pd.to_datetime(dates), "abs_e": np.abs(err)})
        .groupby("Date")["abs_e"].quantile(0.9)
    )
    return {
        "rel_score": rel_score(actual, pred),
        "median_abs_e": float(np.quantile(np.abs(err), 0.5)),
        "q90_abs_e": float(np.quantile(np.abs(err), 0.9)),
        "q95_abs_e": float(np.quantile(np.abs(err), 0.95)),
        "share_abs_e_gt_035": float((np.abs(err) > 0.035).mean()),
        "share_abs_e_gt_050": float((np.abs(err) > 0.05).mean()),
        "share_abs_e_gt_080": float((np.abs(err) > 0.08).mean()),
        "daily_violation_gt_035": int((daily_q90 > 0.035).sum()),
        "daily_violation_share": float((daily_q90 > 0.035).mean()),
        "DA": float(np.mean(np.sign(actual) == np.sign(pred))),
    }


# ── data loading ────────────────────────────────────────────────────────────────

def load_predictions() -> tuple:
    mu_t_list, mu_v_list, sig_t_list, sig_v_list = [], [], [], []
    y_train_ref = y_val_ref = None
    for s in SEEDS:
        d = np.load(PRED_DIR / f"predictions_seed_{s}.npz", allow_pickle=True)
        y_t = d["y_train"].astype(np.float32)
        mu_t = d["mu_train"].astype(np.float32)
        sig_t = d["sigma_train"].astype(np.float32)
        mu_v = d["mu_val"].astype(np.float32)
        sig_v = d["sigma_val"].astype(np.float32)
        sc, cl = choose_train_calibration(y_t, mu_t, sig_t)
        mu_t = mu_t * sc; mu_v = mu_v * sc
        if cl is not None:
            mu_t = np.clip(mu_t, -cl * sig_t, cl * sig_t)
            mu_v = np.clip(mu_v, -cl * sig_v, cl * sig_v)
        mu_t_list.append(mu_t); mu_v_list.append(mu_v)
        sig_t_list.append(sig_t); sig_v_list.append(sig_v)
        y_train_ref = y_t
        y_val_ref = d["y_val"].astype(np.float32)

    mu_t = np.stack(mu_t_list, 1).mean(1)
    mu_v = np.stack(mu_v_list, 1).mean(1)
    # second global calibration
    sc2, cl2 = choose_train_calibration(y_train_ref, mu_t, np.full_like(mu_t, 0.02))
    mu_t = mu_t * sc2; mu_v = mu_v * sc2
    return (
        y_train_ref, y_val_ref,
        mu_t.astype(np.float32), mu_v.astype(np.float32),
        np.stack(sig_t_list, 1).mean(1), np.stack(sig_v_list, 1).mean(1),
        np.stack(mu_t_list, 1).std(1) * sc2,  # seed disagreement train
        np.stack(mu_v_list, 1).std(1) * sc2,  # seed disagreement val
    )


def load_dates() -> tuple[pd.Series, pd.Series]:
    frame = load_training_frame(str(DATA), stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    train_df = frame.loc[frame["Date"] <= "2020-03-31"].copy()
    scaler = fit_feature_scaler(train_df.dropna(subset=features), features)
    scaled = apply_feature_scaler(frame, scaler)
    _, _, meta_all = build_sequence_dataset(
        scaled, features, "target_next_return", 15, sequence_normalization="none"
    )
    dummy_x = np.zeros((len(meta_all), 1), dtype=np.float32)
    splits = split_sequence_dataset(
        dummy_x, meta_all["target"].to_numpy(), meta_all, "2020-03-31", "2022-11-15"
    )
    t_meta = splits["train"][2].reset_index(drop=True)
    v_meta = splits["val"][2].reset_index(drop=True)
    return pd.to_datetime(t_meta["Date"]), pd.to_datetime(v_meta["Date"])


# ── regime feature ──────────────────────────────────────────────────────────────

def build_rv_lag5(
    dates_train: pd.Series,
    dates_val: pd.Series,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lagged 5-day rolling realized volatility (std of daily returns)."""
    # daily realized vol: std of all stock returns for that day
    all_dates = pd.concat([
        pd.DataFrame({"date": dates_train, "actual": y_train}),
        pd.DataFrame({"date": dates_val,   "actual": y_val}),
    ], ignore_index=True)
    daily_rv = all_dates.groupby("date")["actual"].std().rename("rv").reset_index().sort_values("date")
    daily_rv["rv_lag5"] = daily_rv["rv"].shift(5).rolling(5, min_periods=2).mean()
    daily_rv["rv_lag5"] = daily_rv["rv_lag5"].fillna(daily_rv["rv_lag5"].median())

    # Merge back
    def merge_rv(dates: pd.Series) -> np.ndarray:
        df = pd.DataFrame({"date": dates})
        merged = df.merge(daily_rv[["date", "rv_lag5"]], on="date", how="left")
        return merged["rv_lag5"].fillna(daily_rv["rv_lag5"].median()).to_numpy(dtype=np.float32)

    return merge_rv(dates_train), merge_rv(dates_val)


# ── regime calibration ──────────────────────────────────────────────────────────

def fit_regime_scales(
    y_train: np.ndarray,
    mu_train: np.ndarray,
    rv_train: np.ndarray,
    n_deciles: int = N_DECILES,
) -> tuple[np.ndarray, list[dict]]:
    """Learn per-decile scale on train data; returns decile edges + scales."""
    edges = np.percentile(rv_train, np.linspace(0, 100, n_deciles + 1))
    scales = []
    for i in range(n_deciles):
        lo, hi = edges[i], edges[i + 1]
        mask = (rv_train >= lo) & (rv_train < hi) if i < n_deciles - 1 else (rv_train >= lo)
        y_sub, p_sub = y_train[mask], mu_train[mask]
        if mask.sum() < 50:
            scales.append({"decile": i + 1, "lo": lo, "hi": hi, "scale": 1.0,
                           "train_rs": rel_score(y_sub, p_sub), "n": int(mask.sum())})
            continue
        best_sc, best_rs = 1.0, rel_score(y_sub, p_sub)
        for s_try in np.linspace(0.3, 1.5, 49):
            r = rel_score(y_sub, p_sub * s_try)
            if r > best_rs:
                best_rs, best_sc = r, s_try
        scales.append({"decile": i + 1, "lo": float(lo), "hi": float(hi),
                       "scale": float(best_sc), "train_rs": float(best_rs), "n": int(mask.sum())})
    return edges, scales


def apply_regime_scales(
    mu: np.ndarray,
    rv: np.ndarray,
    edges: np.ndarray,
    scales: list[dict],
) -> np.ndarray:
    out = mu.copy()
    n = len(scales)
    for i, row in enumerate(scales):
        mask = (rv >= row["lo"]) & (rv < row["hi"]) if i < n - 1 else (rv >= row["lo"])
        out[mask] *= row["scale"]
    return out.astype(np.float32)


# ── plots ───────────────────────────────────────────────────────────────────────

def plot_comparison(
    y: np.ndarray,
    pred_base: np.ndarray,
    pred_regime: np.ndarray,
    dates: pd.Series,
    scales: list[dict],
    out1: Path, g1: Path,
    out2: Path, g2: Path,
    out3: Path, g3: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, title, pred in [
        (axes[0], "Anchor baseline", pred_base),
        (axes[1], "Regime-calibrated", pred_regime),
    ]:
        err = y - pred
        ax.hist(err * 100, bins=120, density=True, color="#4a90e2", alpha=0.65)
        for q, color, label in [(0.75, "#9955cc", "Q75"), (0.9, "red", "Q90"), (0.95, "darkred", "Q95")]:
            val = np.quantile(np.abs(err), q) * 100
            ax.axvline(val, color=color, linestyle=":", label=f"±{label}={val:.2f}%")
            ax.axvline(-val, color=color, linestyle=":")
        ax.axvline(3.5, color="black", linestyle="--", linewidth=1, label="±3.5%")
        ax.axvline(-3.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Error (%)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_xlim(-12, 12)
    axes[0].set_ylabel("Density")
    fig.suptitle("Regime Calibration — Error Histogram", fontweight="bold")
    fig.tight_layout()
    for p in [out1, g1]:
        fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Q90 timeseries
    def daily_q90(pred: np.ndarray) -> pd.Series:
        return pd.DataFrame({"Date": pd.to_datetime(dates), "abs_e": np.abs(y - pred)}).groupby("Date")["abs_e"].quantile(0.9)

    b = daily_q90(pred_base) * 100
    r = daily_q90(pred_regime) * 100
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(b.index, b.values, color="#888888", alpha=0.6, linewidth=0.8, label=f"Baseline Q90 viol>{(b>3.5).sum()}d")
    ax.plot(r.index, r.values, color="#1a73e8", alpha=0.85, linewidth=0.8, label=f"Regime-cal Q90 viol>{(r>3.5).sum()}d")
    ax.plot(b.index, b.rolling(21, min_periods=5).mean(), color="#666666", linewidth=2.0, alpha=0.9)
    ax.plot(r.index, r.rolling(21, min_periods=5).mean(), color="#1a73e8", linewidth=2.0)
    ax.axhline(3.0, color="green", linestyle="--", label="Target 3.0%")
    ax.axhline(3.5, color="red", linestyle="--", label="Violation 3.5%")
    ax.set_title("Daily Q90(|E|): Baseline vs Regime-Calibrated", fontweight="bold")
    ax.set_ylabel("Q90(|E|) %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for p in [out2, g2]:
        fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Scale per decile bar chart
    deciles_n = [s["decile"] for s in scales]
    scale_vals = [s["scale"] for s in scales]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#e8505b" if s < 1.0 else "#4caf50" for s in scale_vals]
    ax.bar(deciles_n, scale_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(deciles_n)
    ax.set_xticklabels([f"D{d}" for d in deciles_n], fontsize=9)
    ax.set_xlabel("Volatility decile (D1=low, D10=high)")
    ax.set_ylabel("Scale multiplier")
    ax.set_title("Regime-Calibration: Per-Volatility-Decile Scale", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(ax.patches, scale_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    for p in [out3, g3]:
        fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)

    print("Loading predictions and dates …")
    (y_train, y_val, mu_train, mu_val,
     sig_train, sig_val, std_train, std_val) = load_predictions()
    dates_train, dates_val = load_dates()

    print("Building lagged realized volatility …")
    rv_train, rv_val = build_rv_lag5(dates_train, dates_val, y_train, y_val)

    print(f"rv correlation with daily Q90 error: computing …")
    df_rv_corr = pd.DataFrame({"date": dates_val, "abs_e": np.abs(y_val - mu_val), "rv": rv_val})
    daily_rv_corr = df_rv_corr.groupby("date").agg(rv=("rv","first"), q90_e=("abs_e",lambda x: x.quantile(.9))).reset_index()
    rv_corr = float(daily_rv_corr["rv"].corr(daily_rv_corr["q90_e"]))

    print("Fitting regime scales on train …")
    edges, scales = fit_regime_scales(y_train, mu_train, rv_train)

    print("Applying regime scales to validation …")
    mu_regime = apply_regime_scales(mu_val, rv_val, edges, scales)

    baseline = metrics(y_val, mu_val, dates_val)
    baseline["variant"] = "anchor_baseline"
    regime = metrics(y_val, mu_regime, dates_val)
    regime["variant"] = "regime_calibrated"

    comparison = pd.DataFrame([baseline, regime])
    comparison.to_csv(OUTPUT / "comparison.csv", index=False)
    comparison.to_csv(GOLD / "comparison.csv", index=False)

    scales_df = pd.DataFrame(scales)
    scales_df.to_csv(OUTPUT / "decile_scales.csv", index=False)
    scales_df.to_csv(GOLD / "decile_scales.csv", index=False)

    plot_comparison(
        y_val, mu_val, mu_regime, dates_val, scales,
        OUTPUT / "regime_histogram.png", GOLD / "regime_histogram.png",
        OUTPUT / "regime_q90_timeseries.png", GOLD / "regime_q90_timeseries.png",
        OUTPUT / "regime_scales_bar.png", GOLD / "regime_scales_bar.png",
    )

    summary_text = "\n".join([
        "# Regime-Conditional Calibration",
        "",
        "Protocol: lagged-rv regime calibration learned on train; validation readout only. Holdout/test not used.",
        "",
        f"**rv-error correlation (val)**: {rv_corr:.4f}",
        "",
        "## Comparison: Baseline vs Regime-Calibrated",
        "",
        comparison[
            ["variant", "rel_score", "q90_abs_e", "q95_abs_e",
             "daily_violation_gt_035", "daily_violation_share",
             "share_abs_e_gt_050", "DA"]
        ].round(6).to_markdown(index=False),
        "",
        "## Per-Decile Scales (learned on train)",
        "",
        scales_df[["decile", "lo", "hi", "scale", "train_rs", "n"]].round(5).to_markdown(index=False),
        "",
        json.dumps({"output_dir": str(OUTPUT), "gold_dir": str(GOLD)}, indent=2),
    ])
    (OUTPUT / "summary.md").write_text(summary_text, encoding="utf-8")
    (GOLD / "summary.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
