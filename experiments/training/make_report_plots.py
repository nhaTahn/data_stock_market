"""
Frozen validation candidate — report plots for presentation.
No training, no holdout. All data from cached artifacts.

Outputs (PNG, 300 dpi) in:
  gold/vn_transition_pressure_20260512/plots/frozen_validation_candidate_20260524/
  data/processed/.../reports/frozen_validation_candidate_20260524/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/frozen_validation_candidate_20260524"
GOLD   = ROOT / "gold/vn_transition_pressure_20260512/plots/frozen_validation_candidate_20260524"
ENS    = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/fixed_train_ensemble_calibration_20260524"
GATE   = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_market_gate_overlays_20260524"
FINETUNE = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_long_finetune_batch_20260522"

GOLD.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "blue":   "#2E6FD9",
    "orange": "#E07B39",
    "green":  "#2DA44E",
    "red":    "#D03B3B",
    "purple": "#7C3AED",
    "gray":   "#8D99AE",
    "gold":   "#F5A623",
    "light_blue": "#89C4F4",
    "light_green": "#A8E6CF",
}

def save(fig: plt.Figure, name: str) -> None:
    for d in [REPORT, GOLD]:
        p = d / name
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  saved → {p}")
    plt.close(fig)

def style_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("#F9FAFB")
    ax.grid(True, color="#E5E7EB", linewidth=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)

# ─────────────────────────────────────────────────────────────────────
# FIG 1: rel_score series — 21-day fold bars + rolling daily line
# ─────────────────────────────────────────────────────────────────────
print("Plot 1: rel_score series …")
fold_m   = pd.read_csv(ENS / "fold_metrics.csv")
daily_m  = pd.read_csv(ENS / "daily_metrics.csv")

CAND = "ensemble_mean_cal_each_traincal_clip"
fold_c  = fold_m[fold_m["variant"] == CAND].copy()
daily_c = daily_m[daily_m["variant"] == CAND].copy()

fold_c["test_start"] = pd.to_datetime(fold_c["test_start"])
fold_c["test_end"]   = pd.to_datetime(fold_c["test_end"])
fold_c["mid"] = fold_c["test_start"] + (fold_c["test_end"] - fold_c["test_start"]) / 2
fold_c_agg = fold_c.groupby("mid")["rel_score"].mean().reset_index()

daily_c["Date"] = pd.to_datetime(daily_c["Date"])
daily_agg = daily_c.groupby("Date")["rel_score"].mean().reset_index()
daily_agg = daily_agg.sort_values("Date")
daily_agg["roll7"] = daily_agg["rel_score"].rolling(7, min_periods=3).mean()

fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor="white")
fig.suptitle("Prediction Quality — rel_score Series\n(Fixed long-train | Validation 2020-04-01 → 2022-11-15 | Holdout closed)",
             fontsize=13, fontweight="bold", y=0.98)

# top: 21-day fold bars
ax = axes[0]
colors = [PALETTE["blue"] if v >= 0 else PALETTE["red"] for v in fold_c_agg["rel_score"]]
ax.bar(fold_c_agg["mid"], fold_c_agg["rel_score"], width=18, color=colors, alpha=0.75, zorder=3)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(fold_c_agg["rel_score"].mean(), color=PALETTE["orange"], linewidth=1.5,
           linestyle="--", label=f'Mean = {fold_c_agg["rel_score"].mean():.4f}')
style_ax(ax, title="21-Day Fold rel_score (mean across 5 seeds)", ylabel="rel_score")
ax.legend(fontsize=9)
ax.set_xlim(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-12-31"))
pos = (fold_c_agg["rel_score"] >= 0).sum()
neg = (fold_c_agg["rel_score"] < 0).sum()
ax.text(0.99, 0.05, f"✓ {pos} positive  ✗ {neg} negative",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555")

# bottom: daily 7-day rolling
ax2 = axes[1]
ax2.plot(daily_agg["Date"], daily_agg["roll7"], color=PALETTE["blue"], linewidth=1.4,
         label="7-day rolling mean rel_score", zorder=3)
ax2.fill_between(daily_agg["Date"], daily_agg["roll7"], 0,
                 where=daily_agg["roll7"] >= 0, alpha=0.18, color=PALETTE["green"])
ax2.fill_between(daily_agg["Date"], daily_agg["roll7"], 0,
                 where=daily_agg["roll7"] < 0, alpha=0.20, color=PALETTE["red"])
ax2.axhline(0, color="black", linewidth=0.8)
ax2.axhline(daily_agg["roll7"].mean(), color=PALETTE["orange"], linewidth=1.3,
            linestyle="--", label=f'Mean = {daily_agg["roll7"].mean():.4f}')
style_ax(ax2, title="Daily rel_score (7-day rolling, averaged over 5 seeds)", xlabel="Date", ylabel="rel_score")
ax2.legend(fontsize=9)
ax2.set_xlim(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-12-31"))

plt.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "fig1_relscore_series.png")

# ─────────────────────────────────────────────────────────────────────
# FIG 2: abs(E) series — daily q50 (median) and q90
# ─────────────────────────────────────────────────────────────────────
print("Plot 2: abs(E) series …")
daily_absE = daily_c.groupby("Date").agg(
    absE_robust=("absE_robust", "mean"),
    absE_q90=("absE_q90", "mean"),
    base_robust=("base_robust", "mean"),
).reset_index().sort_values("Date")
daily_absE["roll10_q50"]  = daily_absE["absE_robust"].rolling(10, min_periods=5).mean()
daily_absE["roll10_q90"]  = daily_absE["absE_q90"].rolling(10, min_periods=5).mean()
daily_absE["roll10_base"] = daily_absE["base_robust"].rolling(10, min_periods=5).mean()

fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
fig.suptitle("abs(Error) Series — Daily (10-day rolling mean, averaged over 5 seeds)\n(Validation only | Holdout closed)",
             fontsize=12, fontweight="bold")
ax.plot(daily_absE["Date"], daily_absE["roll10_q50"] * 100, color=PALETTE["blue"],
        linewidth=1.5, label="absE_robust (q50+0.5×q90) median")
ax.plot(daily_absE["Date"], daily_absE["roll10_q90"] * 100, color=PALETTE["orange"],
        linewidth=1.5, linestyle="--", label="absE_q90 (tail error)")
ax.plot(daily_absE["Date"], daily_absE["roll10_base"] * 100, color=PALETTE["gray"],
        linewidth=1.2, linestyle=":", label="base robust (naïve)")
ax.fill_between(daily_absE["Date"],
                daily_absE["roll10_q50"] * 100,
                daily_absE["roll10_base"] * 100,
                where=daily_absE["roll10_q50"] < daily_absE["roll10_base"],
                alpha=0.15, color=PALETTE["green"], label="model beats naïve")
style_ax(ax, xlabel="Date", ylabel="Error (%)")
ax.legend(fontsize=9)
ax.set_xlim(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-12-31"))
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig2_absE_series.png")

# ─────────────────────────────────────────────────────────────────────
# FIG 3: Equity curves — 5 seeds + mean (portfolio overlay)
# ─────────────────────────────────────────────────────────────────────
print("Plot 3: equity curves …")
daily_ret = pd.read_csv(FINETUNE / "daily_policy_returns.csv")
daily_ret["Date"] = pd.to_datetime(daily_ret["Date"])

# filter to best policy
TARGET_POLICY = "daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5"
TARGET_GATE_COL = "net_return"

# check columns
# load gate overlay data
seed_gate = pd.read_csv(GATE / "seed_gate_metrics.csv")
fold_gate  = pd.read_csv(GATE / "fold_gate_metrics.csv")

# build daily equity per seed from daily_policy_returns filtered by wyck040
# wyck040 gate mask: need to recreate from data
# load frame metadata
from src.models.training.pipeline import load_frame as _load_frame
DATA_PATH = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
frame = _load_frame(DATA_PATH, stocks=None)
frame["Date"] = pd.to_datetime(frame["Date"])
daily_meta = frame.groupby("Date").agg(
    buying_pressure=("buying_pressure", "mean"),
    selling_pressure=("selling_pressure", "mean"),
    wyckoff_phase_60d=("wyckoff_phase_60d", "mean"),
).sort_index()
pressure_delta = (daily_meta["buying_pressure"] - daily_meta["selling_pressure"]).rolling(20, min_periods=5).mean()
wyck040_mask = ((pressure_delta >= 0) & (daily_meta["wyckoff_phase_60d"] >= 0.40)).astype(int)
wyck040_mask.name = "wyck040"

best_policy = daily_ret[daily_ret["policy"] == TARGET_POLICY].copy()

fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
fig.suptitle("Portfolio Equity Curves — 5 Seeds\n(daily_bot_sig_50pct + Wyckoff040 gate | r20 k20 m5 | Validation only | Holdout closed)",
             fontsize=12, fontweight="bold")

seed_equities = []
for seed, sdf in best_policy.groupby("seed"):
    sdf = sdf.sort_values("Date").copy()
    sdf["gate"] = sdf["Date"].map(wyck040_mask).fillna(0).astype(int)
    sdf["gated_ret"] = sdf["net_return"] * sdf["gate"]
    eq = (1 + sdf["gated_ret"]).cumprod()
    seed_equities.append(eq.values)
    ax.plot(sdf["Date"], eq, color=PALETTE["light_blue"], linewidth=0.9, alpha=0.7)

dates_ref = best_policy[best_policy["seed"] == best_policy["seed"].iloc[0]].sort_values("Date")["Date"]
mean_eq = np.array(seed_equities).mean(axis=0)
ax.plot(dates_ref, mean_eq, color=PALETTE["blue"], linewidth=2.2, label=f"Mean equity = {mean_eq[-1]:.3f}")
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")

# shade gate-off periods
gate_vals = dates_ref.map(wyck040_mask).fillna(0)
prev = 1
for i, (d, g) in enumerate(zip(dates_ref, gate_vals)):
    if g == 0 and prev == 1:
        start_shade = d
    if g == 1 and prev == 0:
        ax.axvspan(start_shade, d, alpha=0.07, color=PALETTE["red"])
    prev = g

style_ax(ax, xlabel="Date", ylabel="Portfolio Equity (start = 1.0)")
ax.legend(fontsize=10)
ax.set_xlim(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-12-31"))

# annotate key stats
txt = (f"Sharpe {1.2436:.2f}  |  Max DD {-18.75:.1f}%  |  "
       f"Positive folds {77}/155  |  Gate active {59.5:.0f}%")
ax.text(0.01, 0.03, txt, transform=ax.transAxes, fontsize=9, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig3_equity_curves.png")

# ─────────────────────────────────────────────────────────────────────
# FIG 4: 21-day fold equity distribution histogram
# ─────────────────────────────────────────────────────────────────────
print("Plot 4: fold equity distribution …")
fold_gate_df = fold_gate[
    (fold_gate["policy"] == TARGET_POLICY) &
    (fold_gate["gate"] == "wyck040")
].copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
fig.suptitle("Fold-Level Distribution (portfolio overlay)\n(daily_bot_sig_50pct + wyck040 | 155 folds × 5 seeds | Validation only)",
             fontsize=12, fontweight="bold")

ax = axes[0]
eq_vals = fold_gate_df["final_equity"].values
bins = np.linspace(0.7, 1.5, 25)
ax.hist(eq_vals[eq_vals >= 1.0], bins=bins, color=PALETTE["green"], alpha=0.75, label="Positive folds")
ax.hist(eq_vals[eq_vals < 1.0],  bins=bins, color=PALETTE["red"],   alpha=0.75, label="Negative folds")
ax.axvline(1.0, color="black", linewidth=1.2)
ax.axvline(eq_vals.mean(), color=PALETTE["orange"], linewidth=1.5, linestyle="--",
           label=f"Mean = {eq_vals.mean():.3f}")
ax.axvline(eq_vals.min(), color=PALETTE["red"], linewidth=1.5, linestyle=":",
           label=f"Min = {eq_vals.min():.3f}")
style_ax(ax, title="21-Day Fold Equity Distribution", xlabel="Fold Equity", ylabel="Count")
ax.legend(fontsize=9)

# right: fold rel_score histogram from prediction
fold_rel_vals = fold_c_agg["rel_score"].values
ax2 = axes[1]
bins2 = np.linspace(-0.06, 0.16, 25)
ax2.hist(fold_rel_vals[fold_rel_vals >= 0], bins=bins2, color=PALETTE["blue"], alpha=0.75, label="Positive folds")
ax2.hist(fold_rel_vals[fold_rel_vals < 0],  bins=bins2, color=PALETTE["red"],  alpha=0.75, label="Negative folds")
ax2.axvline(0, color="black", linewidth=1.2)
ax2.axvline(fold_rel_vals.mean(), color=PALETTE["orange"], linewidth=1.5, linestyle="--",
            label=f"Mean = {fold_rel_vals.mean():.4f}")
style_ax(ax2, title="21-Day Fold rel_score Distribution (prediction)", xlabel="rel_score", ylabel="Count")
ax2.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, "fig4_fold_distributions.png")

# ─────────────────────────────────────────────────────────────────────
# FIG 5: Year-by-year comparison bar chart
# ─────────────────────────────────────────────────────────────────────
print("Plot 5: year summary bars …")
year_m = pd.read_csv(REPORT / "prediction_year_metrics.csv")
month_m = pd.read_csv(REPORT / "prediction_month_metrics.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
fig.suptitle("Year-by-Year Prediction Summary\n(ensemble_mean_cal_each_traincal_clip | Validation only | Holdout closed)",
             fontsize=12, fontweight="bold")

years = year_m["year"].astype(str).values
x = np.arange(len(years))
width = 0.5

# top-left: rel_score by year
ax = axes[0][0]
colors = [PALETTE["blue"] if v >= 0 else PALETTE["red"] for v in year_m["mean_rel_score"]]
bars = ax.bar(x, year_m["mean_rel_score"], width=width, color=colors, alpha=0.8)
ax.axhline(0, color="black", linewidth=0.8)
for bar, v in zip(bars, year_m["mean_rel_score"]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.0005, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(years)
style_ax(ax, title="Mean Daily rel_score by Year", ylabel="rel_score")

# top-right: positive day ratio
ax2 = axes[0][1]
pos_ratio = year_m["positive_days"] / year_m["days"]
ax2.bar(x, pos_ratio, width=width, color=PALETTE["green"], alpha=0.8)
ax2.axhline(0.5, color=PALETTE["gray"], linewidth=1.2, linestyle="--", label="50% line")
for i, v in enumerate(pos_ratio):
    ax2.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
ax2.set_xticks(x); ax2.set_xticklabels(years)
ax2.set_ylim(0, 0.85)
style_ax(ax2, title="Positive Prediction Days Ratio by Year", ylabel="Ratio")
ax2.legend(fontsize=9)

# bottom-left: abs(E) by year
ax3 = axes[1][0]
ax3.bar(x - 0.2, year_m["mean_absE_robust"] * 100, width=0.35, color=PALETTE["blue"],  alpha=0.8, label="absE_robust")
ax3.bar(x + 0.2, year_m["p90_absE_robust"]  * 100, width=0.35, color=PALETTE["orange"], alpha=0.8, label="p90 absE_robust")
ax3.set_xticks(x); ax3.set_xticklabels(years)
style_ax(ax3, title="abs(Error) by Year (%)", ylabel="%")
ax3.legend(fontsize=9)

# bottom-right: monthly rel_score heatmap-style bar
ax4 = axes[1][1]
month_m["month_label"] = month_m["year"].astype(str) + "-" + month_m["month"].astype(str).str.zfill(2)
month_m = month_m.sort_values(["year","month"])
mc = [PALETTE["blue"] if v >= 0 else PALETTE["red"] for v in month_m["mean_rel_score"]]
ax4.bar(range(len(month_m)), month_m["mean_rel_score"], color=mc, alpha=0.75)
ax4.axhline(0, color="black", linewidth=0.8)
ax4.axhline(month_m["mean_rel_score"].mean(), color=PALETTE["orange"], linewidth=1.2, linestyle="--",
            label=f'Mean = {month_m["mean_rel_score"].mean():.4f}')
labels = [m if m.endswith("-01") or m.endswith("-07") else "" for m in month_m["month_label"]]
ax4.set_xticks(range(len(month_m)))
ax4.set_xticklabels(month_m["month_label"], rotation=60, fontsize=7, ha="right")
style_ax(ax4, title="Monthly rel_score (mean daily)", ylabel="rel_score")
ax4.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig5_yearly_summary.png")

# ─────────────────────────────────────────────────────────────────────
# FIG 6: Summary scorecard (table + radar-style bar)
# ─────────────────────────────────────────────────────────────────────
print("Plot 6: scorecard …")
fig = plt.figure(figsize=(13, 6), facecolor="white")
fig.suptitle("Frozen Validation Candidate — Scorecard\n(2026-05-25 | Holdout closed)",
             fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], figure=fig)

# left: table
ax_t = fig.add_subplot(gs[0])
ax_t.axis("off")

table_data = [
    ["Metric", "Prediction\ncandidate", "Portfolio\noverlay"],
    ["rel_score (overall)", "0.04478", "—"],
    ["absE_robust", "3.60%", "—"],
    ["absE_q90", "4.69%", "—"],
    ["DA", "51.83%", "—"],
    ["pred/actual q90 ratio", "0.193", "—"],
    ["21-day positive folds", "24/32", "77/155"],
    ["Mean fold rel/equity", "0.0298", "1.0145"],
    ["Worst fold rel/equity", "-0.028", "0.953"],
    ["Mean equity (2y)", "—", "1.5231"],
    ["Min seed equity", "—", "1.4417"],
    ["Mean Sharpe", "—", "1.2436"],
    ["Worst max DD", "—", "-18.75%"],
    ["Gate active", "—", "59.48%"],
]
tbl = ax_t.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.55)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2E6FD9")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EFF4FF")
    else:
        cell.set_facecolor("#FAFAFA")
    cell.set_edgecolor("#D1D5DB")

# right: comparison bar with previous baseline
ax_b = fig.add_subplot(gs[1])
metrics = ["rel_score\n(×10)", "absE_rob\n(%) inv", "DA (%)", "Sharpe\n(portf)"]
baseline = [0.0248 * 10,   (1-3.72/3.72)*100 + 50,  50.7, 0.0]
current  = [0.04478 * 10,  (1-3.60/3.72)*100 + 50,  51.83, 1.2436 * 40]
# Normalise to 0-100 for display
max_v = [max(b, c, 1) for b,c in zip(baseline, current)]
b_norm = [b/m*80 for b,m in zip(baseline, max_v)]
c_norm = [c/m*80 for c,m in zip(current, max_v)]
x2 = np.arange(len(metrics))
w2 = 0.33
ax_b.bar(x2 - w2/2, b_norm, width=w2, color=PALETTE["gray"],   alpha=0.85, label="Baseline (stressaux_w20)")
ax_b.bar(x2 + w2/2, c_norm, width=w2, color=PALETTE["blue"],   alpha=0.85, label="Frozen ensemble candidate")
ax_b.set_xticks(x2)
ax_b.set_xticklabels(metrics, fontsize=10)
ax_b.set_yticks([])
ax_b.legend(fontsize=9)
style_ax(ax_b, title="Relative Improvement vs Baseline", ylabel="Normalised score")
for xi, (bv, cv) in enumerate(zip(baseline, current)):
    ax_b.text(xi - w2/2, b_norm[xi] + 1, f"{bv:.3f}", ha="center", fontsize=8, color="#555")
    ax_b.text(xi + w2/2, c_norm[xi] + 1, f"{cv:.3f}", ha="center", fontsize=8, color=PALETTE["blue"])

plt.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig6_scorecard.png")

print("\nDone. All 6 plots saved.")
