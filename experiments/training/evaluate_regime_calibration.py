"""Regime-conditional calibration + conformal prediction intervals.

Three contributions evaluated on VN validation (holdout/test not used):

1. 1D-Regime: per-volatility-decile scale → best rel_score improvement
2. 2D-Regime: joint (vol × market-magnitude) grid → best tail-share improvement
3. Conformal Intervals: split-conformal 90% coverage intervals from train residuals

All features use lagged values (no look-ahead).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.evaluate_fixed_train_relscore_calibration import choose_train_calibration  # noqa
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset  # noqa
from src.models.training.pipeline import load_frame as load_training_frame  # noqa
from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler  # noqa

PRED_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_combined_full5_20260521"
DATA     = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
OUTPUT   = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/regime_calibration_20260527"
GOLD     = ROOT / "gold/vn_transition_pressure_20260512/plots/regime_calibration_20260527"
SEEDS    = (43, 52, 62, 71, 82)
N_DECILES_1D = 10
N_DECILES_2D = 6
CONFORMAL_ALPHA = 0.10


# ── helpers ─────────────────────────────────────────────────────────────────

def robust_loss(v: np.ndarray) -> float:
    c = v[np.isfinite(v)]
    return float(np.quantile(np.abs(c), .5) + .5 * np.quantile(np.abs(c), .9)) if len(c) else float("nan")


def rel_score(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    base = robust_loss(y[m])
    return float(1 - robust_loss((y - p)[m]) / base) if base > 0 else float("nan")


def metrics_row(y: np.ndarray, p: np.ndarray, dates: pd.Series) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    y, p, dates = y[m], p[m], dates[m]
    err = y - p
    daily_q90 = (pd.DataFrame({"Date": pd.to_datetime(dates), "ae": np.abs(err)})
                 .groupby("Date")["ae"].quantile(0.9))
    return {
        "rel_score":               rel_score(y, p),
        "q90_abs_e":               float(np.quantile(np.abs(err), .9)),
        "q95_abs_e":               float(np.quantile(np.abs(err), .95)),
        "share_abs_e_gt_035":      float((np.abs(err) > .035).mean()),
        "share_abs_e_gt_050":      float((np.abs(err) > .05).mean()),
        "daily_violation_gt_035":  int((daily_q90 > .035).sum()),
        "daily_violation_share":   float((daily_q90 > .035).mean()),
        "DA":                      float(np.mean(np.sign(y) == np.sign(p))),
    }


# ── data loading ─────────────────────────────────────────────────────────────

def load_predictions():
    mu_t, mu_v = [], []
    y_train = y_val = None
    for s in SEEDS:
        d  = np.load(PRED_DIR / f"predictions_seed_{s}.npz", allow_pickle=True)
        yt = d["y_train"].astype(np.float32)
        mt = d["mu_train"].astype(np.float32); st = d["sigma_train"].astype(np.float32)
        mv = d["mu_val"].astype(np.float32);   sv = d["sigma_val"].astype(np.float32)
        sc, cl = choose_train_calibration(yt, mt, st)
        mt, mv = mt * sc, mv * sc
        if cl is not None:
            mt = np.clip(mt, -cl * st, cl * st)
            mv = np.clip(mv, -cl * sv, cl * sv)
        mu_t.append(mt); mu_v.append(mv)
        y_train = yt; y_val = d["y_val"].astype(np.float32)
    mu_t_mean = np.stack(mu_t, 1).mean(1)
    mu_v_mean = np.stack(mu_v, 1).mean(1)
    sc2, _ = choose_train_calibration(y_train, mu_t_mean, np.full_like(mu_t_mean, .02))
    return (y_train, y_val,
            (mu_t_mean * sc2).astype(np.float32),
            (mu_v_mean * sc2).astype(np.float32))


def load_dates():
    frame = load_training_frame(str(DATA), stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    scaler   = fit_feature_scaler(frame.loc[frame["Date"] <= "2020-03-31"].dropna(subset=features), features)
    scaled   = apply_feature_scaler(frame, scaler)
    _, _, meta = build_sequence_dataset(scaled, features, "target_next_return", 15, sequence_normalization="none")
    dummy = np.zeros((len(meta), 1), dtype=np.float32)
    splits = split_sequence_dataset(dummy, meta["target"].to_numpy(), meta, "2020-03-31", "2022-11-15")
    return (pd.to_datetime(splits["train"][2].reset_index(drop=True)["Date"]),
            pd.to_datetime(splits["val"][2].reset_index(drop=True)["Date"]))


# ── market features ──────────────────────────────────────────────────────────

def build_market_features(dates_train: pd.Series, dates_val: pd.Series,
                           y_train: np.ndarray, y_val: np.ndarray):
    raw = pd.read_csv(DATA, usecols=["Date", "code", "target_next_return"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    dm = raw.groupby("Date").agg(mkt_ret=("target_next_return", "mean"),
                                  mkt_std=("target_next_return", "std")).sort_index()
    dm["mkt_q90"] = raw.groupby("Date")["target_next_return"].apply(lambda x: np.quantile(np.abs(x), .9))
    dm["rv_lag5"]         = dm["mkt_std"].shift(5).rolling(5, min_periods=2).mean()
    dm["vol10"]           = dm["mkt_std"].shift(1).rolling(10, min_periods=3).mean()
    dm["q905"]            = dm["mkt_q90"].shift(1).rolling(5, min_periods=3).mean()
    dm["mkt_abs_lag3"]    = dm["mkt_ret"].abs().shift(3).rolling(3, min_periods=2).mean()
    dm = dm.fillna(dm.median())

    def lookup(dates, col):
        df = pd.DataFrame({"Date": dates}).merge(dm.reset_index()[["Date", col]], on="Date", how="left")
        return df[col].fillna(dm[col].median()).to_numpy(dtype=np.float32)

    return {
        "train": {
            "rv_lag5": lookup(dates_train, "rv_lag5"),
            "mkt_abs_lag3": lookup(dates_train, "mkt_abs_lag3"),
            "vol10": lookup(dates_train, "vol10"),
            "q905": lookup(dates_train, "q905"),
        },
        "val": {
            "rv_lag5": lookup(dates_val, "rv_lag5"),
            "mkt_abs_lag3": lookup(dates_val, "mkt_abs_lag3"),
            "vol10": lookup(dates_val, "vol10"),
            "q905": lookup(dates_val, "q905"),
        },
    }


# ── 1D regime calibration ─────────────────────────────────────────────────────

def fit_1d_scales(y: np.ndarray, p: np.ndarray, rv: np.ndarray, n: int = N_DECILES_1D):
    edges = np.percentile(rv, np.linspace(0, 100, n + 1))
    rows = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        mask = (rv >= lo) & (rv < hi if i < n - 1 else rv <= hi)
        ys, ps = y[mask], p[mask]
        if mask.sum() < 50:
            rows.append({"decile": i+1, "lo": float(lo), "hi": float(hi), "scale": 1.0,
                         "train_rs": rel_score(ys, ps), "n": int(mask.sum())}); continue
        best_s, best_r = 1.0, rel_score(ys, ps)
        for st in np.linspace(0.3, 1.5, 49):
            r = rel_score(ys, ps * st)
            if r > best_r:
                best_r, best_s = r, st
        rows.append({"decile": i+1, "lo": float(lo), "hi": float(hi), "scale": float(best_s),
                     "train_rs": float(best_r), "n": int(mask.sum())})
    return edges, rows


def apply_1d_scales(p: np.ndarray, rv: np.ndarray, edges, rows) -> np.ndarray:
    out = p.copy()
    for i, row in enumerate(rows):
        mask = (rv >= row["lo"]) & (rv < row["hi"] if i < len(rows)-1 else rv <= row["hi"])
        out[mask] *= row["scale"]
    return out.astype(np.float32)


# ── 2D regime calibration ─────────────────────────────────────────────────────

def fit_2d_scales(y, p, rv, mk, n=N_DECILES_2D):
    rv_e = np.percentile(rv, np.linspace(0, 100, n + 1))
    mk_e = np.percentile(mk, np.linspace(0, 100, n + 1))
    grid = {}
    for i in range(n):
        for j in range(n):
            rv_lo, rv_hi = rv_e[i], rv_e[i+1]
            mk_lo, mk_hi = mk_e[j], mk_e[j+1]
            mask = ((rv >= rv_lo) & (rv < rv_hi if i < n-1 else rv <= rv_hi) &
                    (mk >= mk_lo) & (mk < mk_hi if j < n-1 else mk <= mk_hi))
            ys, ps = y[mask], p[mask]
            if mask.sum() < 300:
                grid[(i,j)] = 1.0; continue
            def objective(pred: np.ndarray) -> float:
                err = ys - pred
                return (
                    rel_score(ys, pred)
                    - 0.8 * float(np.quantile(np.abs(err), 0.9))
                    - 0.25 * float((np.abs(err) > 0.05).mean())
                )
            best_s, best_r = 1.0, objective(ps)
            for st in np.linspace(0.4, 1.6, 49):
                r = objective(ps * st)
                if r > best_r:
                    best_r, best_s = r, st
            grid[(i,j)] = float(best_s)
    return rv_e, mk_e, grid


def apply_2d_scales(p, rv, mk, rv_e, mk_e, grid, n=N_DECILES_2D):
    out = p.copy()
    for i in range(n):
        for j in range(n):
            rv_lo, rv_hi = rv_e[i], rv_e[i+1]
            mk_lo, mk_hi = mk_e[j], mk_e[j+1]
            mask = ((rv >= rv_lo) & (rv < rv_hi if i < n-1 else rv <= rv_hi) &
                    (mk >= mk_lo) & (mk < mk_hi if j < n-1 else mk <= mk_hi))
            out[mask] *= grid[(i,j)]
    return out.astype(np.float32)


# ── conformal prediction intervals ──────────────────────────────────────────

def build_conformal_intervals(y_train, p_train, p_val, alpha=CONFORMAL_ALPHA):
    """Split conformal: use train residuals as calibration set."""
    residuals = np.abs(y_train - p_train)
    q = float(np.quantile(residuals, 1 - alpha))
    lower = p_val - q
    upper = p_val + q
    return lower, upper, q


def build_hetero_conformal_intervals(
    y_train,
    p_train,
    p_val,
    rv_train,
    rv_val,
    quantile=0.90,
    floor=0.015,
):
    residuals = np.abs(y_train - p_train)
    scale_train = np.maximum(rv_train, floor)
    scale_val = np.maximum(rv_val, floor)
    q = float(np.quantile(residuals / scale_train, quantile))
    width = q * scale_val
    return p_val - width, p_val + width, q


def conformal_metrics(y_val, lower, upper):
    covered = (y_val >= lower) & (y_val <= upper)
    return {
        "coverage": float(covered.mean()),
        "mean_width": float((upper - lower).mean()),
        "median_width": float(np.median(upper - lower)),
        "q90_width": float(np.quantile(upper - lower, 0.9)),
    }


# ── plots ────────────────────────────────────────────────────────────────────

def plot_histogram(y, p_base, p_1d, p_2d, out, gold):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    combos = [("Anchor baseline", p_base), ("1D-Regime (best RS)", p_1d), ("2D-Regime (best tail)", p_2d)]
    for ax, (title, pred) in zip(axes, combos):
        err = y - pred
        ax.hist(err * 100, bins=120, density=True, color="#4a90e2", alpha=0.65)
        for q, color, label in [(.75,"#9955cc","Q75"), (.9,"red","Q90"), (.95,"darkred","Q95")]:
            v = np.quantile(np.abs(err), q) * 100
            ax.axvline(v, color=color, linestyle=":", label=f"±{label}={v:.2f}%")
            ax.axvline(-v, color=color, linestyle=":")
        ax.axvline(3.5, color="black", linestyle="--", linewidth=1)
        ax.axvline(-3.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(title); ax.set_xlabel("Error (%)")
        ax.grid(alpha=0.25); ax.legend(fontsize=7); ax.set_xlim(-12, 12)
    axes[0].set_ylabel("Density")
    fig.suptitle("Regime Calibration Variants — Error Histogram", fontweight="bold")
    fig.tight_layout()
    for path in [out, gold]:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_q90_timeseries(y, p_base, p_1d, p_2d, dates, out, gold):
    def dq90(pred):
        return (pd.DataFrame({"Date": pd.to_datetime(dates), "ae": np.abs(y - pred)})
                .groupby("Date")["ae"].quantile(.9) * 100)
    b, r1, r2 = dq90(p_base), dq90(p_1d), dq90(p_2d)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(b.index, b.values, color="#aaaaaa", alpha=0.5, linewidth=0.7, label=f"Baseline viol>{(b>3.5).sum()}d")
    ax.plot(r1.index, r1.values, color="#1a73e8", alpha=0.75, linewidth=0.8, label=f"1D-Regime viol>{(r1>3.5).sum()}d")
    ax.plot(r2.index, r2.values, color="#f4511e", alpha=0.75, linewidth=0.8, label=f"2D-Regime viol>{(r2>3.5).sum()}d")
    ax.plot(b.index, b.rolling(21,min_periods=5).mean(), color="#888", linewidth=2.0, alpha=0.9)
    ax.plot(r1.index, r1.rolling(21,min_periods=5).mean(), color="#1a73e8", linewidth=2.0)
    ax.plot(r2.index, r2.rolling(21,min_periods=5).mean(), color="#f4511e", linewidth=2.0)
    ax.axhline(3.0, color="green", linestyle="--", label="Target 3.0%")
    ax.axhline(3.5, color="red",   linestyle="--", label="Violation 3.5%")
    ax.set_title("Daily Q90(|E|): Baseline vs 1D / 2D Regime", fontweight="bold")
    ax.set_ylabel("Q90(|E|) %"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    for path in [out, gold]:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_scales_bar(rows, out, gold):
    dn = [r["decile"] for r in rows]; sv = [r["scale"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#e8505b" if s < 1.0 else "#4caf50" for s in sv]
    ax.bar(dn, sv, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(dn); ax.set_xticklabels([f"D{d}" for d in dn], fontsize=9)
    ax.set_xlabel("Volatility decile (D1=low, D10=high)")
    ax.set_ylabel("Scale multiplier")
    ax.set_title("1D Regime: Per-Decile Scale Multipliers", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(ax.patches, sv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    for path in [out, gold]:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_interval_coverage(y, intervals, dates, out, gold):
    def daily_cov(lo, hi):
        return (pd.DataFrame({"Date": pd.to_datetime(dates),
                              "covered": (y >= lo) & (y <= hi),
                              "width":   hi - lo})
                .groupby("Date").agg(coverage=("covered","mean"), width=("width","mean")))

    fig, ax1 = plt.subplots(figsize=(15, 5))
    colors = ["#1a73e8", "#f4511e", "#34a853"]
    for idx, item in enumerate(intervals):
        daily = daily_cov(item["lower"], item["upper"])
        ax1.plot(
            daily.index,
            daily["coverage"].rolling(21, min_periods=5).mean() * 100,
            color=colors[idx % len(colors)],
            linewidth=2,
            linestyle=item.get("linestyle", "-"),
            label=f"{item['label']} coverage",
        )
    ax1.axhline(90, color="green", linestyle="--", label="Nominal 90%")
    ax1.axhline(95, color="#34a853", linestyle=":", label="Nominal 95%")
    ax1.set_ylabel("Coverage (%)", color="#1a73e8"); ax1.set_ylim(40, 102)
    ax2 = ax1.twinx()
    for idx, item in enumerate(intervals):
        daily = daily_cov(item["lower"], item["upper"])
        ax2.plot(
            daily.index,
            daily["width"] * 100,
            color=colors[idx % len(colors)],
            alpha=0.25,
            linewidth=0.8,
            linestyle=item.get("linestyle", "-"),
            label=f"{item['label']} width",
        )
    ax2.set_ylabel("Width (%)", color="#888")
    ax1.set_title("Conformal Prediction Intervals: Coverage & Width", fontweight="bold")
    ax1.grid(alpha=0.3)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lab1+lab2, fontsize=9, loc="lower left")
    fig.tight_layout()
    for path in [out, gold]:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)

    print("Loading predictions …")
    y_train, y_val, mu_train, mu_val = load_predictions()
    print("Loading dates …")
    dates_train, dates_val = load_dates()
    print("Building market features …")
    mf = build_market_features(dates_train, dates_val, y_train, y_val)
    rv_train = mf["train"]["rv_lag5"];  rv_val = mf["val"]["rv_lag5"]
    mk_train = mf["train"]["q905"]; mk_val = mf["val"]["q905"]

    # rv-error correlation (informational)
    daily = pd.DataFrame({"date": dates_val, "ae": np.abs(y_val - mu_val), "rv": rv_val})
    rv_corr = float(daily.groupby("date").agg(rv=("rv","first"), ae=("ae","mean"))
                    .corr().loc["rv","ae"])
    print(f"rv-error corr (val): {rv_corr:.4f}")

    # ── 1D regime ──────────────────────────────────────────────────────────
    print("Fitting 1D regime …")
    edges_1d, scales_1d = fit_1d_scales(y_train, mu_train, rv_train)
    mu_1d_val = apply_1d_scales(mu_val, rv_val, edges_1d, scales_1d)
    mu_1d_train = apply_1d_scales(mu_train, rv_train, edges_1d, scales_1d)

    # ── 2D regime ──────────────────────────────────────────────────────────
    print("Fitting 2D regime …")
    rv2_train = mf["train"]["vol10"]; rv2_val = mf["val"]["vol10"]
    rv_e2, mk_e2, grid_2d = fit_2d_scales(y_train, mu_train, rv2_train, mk_train)
    mu_2d_val   = apply_2d_scales(mu_val,   rv2_val,   mk_val,   rv_e2, mk_e2, grid_2d)
    mu_2d_train = apply_2d_scales(mu_train, rv2_train, mk_train, rv_e2, mk_e2, grid_2d)

    # ── conformal intervals ────────────────────────────────────────────────
    print("Building conformal intervals …")
    lo_plain90, hi_plain90, q_plain90 = build_conformal_intervals(y_train, mu_1d_train, mu_1d_val, alpha=0.10)
    lo_het90, hi_het90, q_het90 = build_hetero_conformal_intervals(
        y_train, mu_1d_train, mu_1d_val, rv_train, rv_val, quantile=0.90
    )
    lo_het95, hi_het95, q_het95 = build_hetero_conformal_intervals(
        y_train, mu_1d_train, mu_1d_val, rv_train, rv_val, quantile=0.95
    )
    conf_plain90 = conformal_metrics(y_val, lo_plain90, hi_plain90)
    conf_het90 = conformal_metrics(y_val, lo_het90, hi_het90)
    conf_het95 = conformal_metrics(y_val, lo_het95, hi_het95)

    # ── metrics ────────────────────────────────────────────────────────────
    base_m = metrics_row(y_val, mu_val,    dates_val); base_m["variant"] = "anchor_baseline"
    m_1d   = metrics_row(y_val, mu_1d_val, dates_val); m_1d["variant"]   = "1d_regime"
    m_2d   = metrics_row(y_val, mu_2d_val, dates_val); m_2d["variant"]   = "2d_regime"
    cmp = pd.DataFrame([base_m, m_1d, m_2d])
    cmp.to_csv(OUTPUT / "comparison.csv", index=False)
    cmp.to_csv(GOLD   / "comparison.csv", index=False)

    pd.DataFrame(scales_1d).to_csv(OUTPUT / "decile_scales_1d.csv", index=False)
    pd.DataFrame(scales_1d).to_csv(GOLD   / "decile_scales_1d.csv", index=False)

    conf_df = pd.DataFrame([
        {"variant": "plain_conformal_90pct", "train_quantile": q_plain90, **conf_plain90},
        {"variant": "hetero_rv_conformal_90pct", "train_quantile": q_het90, **conf_het90},
        {"variant": "hetero_rv_conformal_95pct", "train_quantile": q_het95, **conf_het95},
    ])
    conf_df.to_csv(OUTPUT / "conformal_metrics.csv", index=False)
    conf_df.to_csv(GOLD   / "conformal_metrics.csv", index=False)

    # ── plots ──────────────────────────────────────────────────────────────
    print("Plotting …")
    plot_histogram(y_val, mu_val, mu_1d_val, mu_2d_val,
                   OUTPUT/"regime_histogram.png", GOLD/"regime_histogram.png")
    plot_q90_timeseries(y_val, mu_val, mu_1d_val, mu_2d_val, dates_val,
                        OUTPUT/"regime_q90_timeseries.png", GOLD/"regime_q90_timeseries.png")
    plot_scales_bar(scales_1d,
                    OUTPUT/"regime_scales_bar.png", GOLD/"regime_scales_bar.png")
    plot_interval_coverage(
        y_val,
        [
            {"label": "Plain90", "lower": lo_plain90, "upper": hi_plain90, "linestyle": "-"},
            {"label": "Hetero90", "lower": lo_het90, "upper": hi_het90, "linestyle": "--"},
            {"label": "Hetero95", "lower": lo_het95, "upper": hi_het95, "linestyle": ":"},
        ],
        dates_val,
        OUTPUT/"conformal_coverage.png",
        GOLD/"conformal_coverage.png",
    )

    # ── summary md ─────────────────────────────────────────────────────────
    text = "\n".join([
        "# Regime-Conditional Calibration + Conformal Intervals",
        "",
        "Protocol: lagged features; validation readout only. Holdout/test not used.",
        "",
        f"rv–error correlation (val daily Q90): **{rv_corr:.4f}**",
        "",
        "## Variant Comparison",
        "",
        cmp[["variant","rel_score","q90_abs_e","q95_abs_e",
             "daily_violation_gt_035","daily_violation_share",
             "share_abs_e_gt_050","DA"]].round(6).to_markdown(index=False),
        "",
        "## 1D Per-Decile Scales",
        "",
        pd.DataFrame(scales_1d)[["decile","lo","hi","scale","train_rs","n"]].round(5).to_markdown(index=False),
        "",
        "## Conformal Interval Metrics",
        "",
        conf_df.to_markdown(index=False),
        "",
        json.dumps({"output_dir": str(OUTPUT), "gold_dir": str(GOLD)}, indent=2),
    ])
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD   / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
