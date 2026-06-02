"""Apply VN transition pressure gate to hetero holding-period candidates.

Offline simulation using 5-seed ensemble predictions from hetero_combined_full5.
Gate definition follows advisor report:
  pressure_delta_20(t) = MA20(mean_i(buying_pressure - selling_pressure))
  trade only when pressure_delta_20 >= 0

Evaluate selected candidate families:
  - daily_bot_sig_50pct, conf_ratio_q80, conf_ratio_q70, full
  - rebalance_every: 10, 20
  - top_k: 8, 10, 15
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.run_hetero_holding_period_grid import (  # noqa: E402
    ROOT as _ROOT,
    COST_BPS,
    PRED_DIR,
    SEEDS,
    build_val_meta,
    load_ensemble_predictions,
    apply_train_thresholds,
    simulate_portfolio,
    summarize,
)
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402

OUTPUT_DIR = (
    ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports"
    / "hetero_pressure_gate_grid_20260521"
)
GOLD_DIR = ROOT / "gold/vn_transition_pressure_20260512/plots/hetero_pressure_gate_grid_20260521"

RULES = ["daily_bot_sig_50pct", "conf_ratio_q80", "conf_ratio_q70", "full"]
REBALANCE = [10, 20]
TOP_K = [8, 10, 15]


def pressure_gate_by_date() -> pd.Series:
    data_path = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
    frame = load_training_frame(data_path, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    if not {"buying_pressure", "selling_pressure"}.issubset(frame.columns):
        raise KeyError("Need buying_pressure and selling_pressure columns")
    tmp = frame[["Date", "buying_pressure", "selling_pressure"]].dropna().copy()
    tmp["pressure_delta"] = tmp["buying_pressure"].astype(float) - tmp["selling_pressure"].astype(float)
    daily = tmp.groupby("Date")["pressure_delta"].mean().sort_index()
    pd20 = daily.rolling(20, min_periods=5).mean()
    gate = (pd20 >= 0).astype(bool)
    return gate


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    y_val, meta_val = build_val_meta()
    dates_val = pd.to_datetime(meta_val["Date"]).values
    codes_val = meta_val["code"].values if "code" in meta_val.columns else np.arange(len(y_val))
    mu_tr, sig_tr, mu_v, sig_v, y_v = load_ensemble_predictions(SEEDS)
    signals = apply_train_thresholds(mu_tr, sig_tr, mu_v, sig_v)

    # add daily_bot_sig_50pct
    df_tmp = pd.DataFrame({"date": pd.to_datetime(dates_val), "sigma": sig_v})
    df_tmp.index = np.arange(len(df_tmp))
    mask_arr = np.zeros(len(mu_v), dtype=bool)
    for _, grp in df_tmp.groupby("date"):
        thresh = grp["sigma"].quantile(0.50)
        mask_arr[grp.index[grp["sigma"] <= thresh]] = True
    signals["daily_bot_sig_50pct"] = np.where(mask_arr, np.abs(mu_v), 0.0).astype(np.float32)

    gate = pressure_gate_by_date()
    date_series = pd.Series(pd.to_datetime(dates_val))
    gate_mask = date_series.map(gate).fillna(False).to_numpy(dtype=bool)
    gate_days = pd.Series(gate_mask, index=date_series).groupby(level=0).first()
    print(f"Pressure gate active days: {gate_days.mean():.1%}")

    rows = []
    for rule in RULES:
        base_sig = signals[rule]
        gated_sig = np.where(gate_mask, base_sig, 0.0).astype(np.float32)
        for gate_name, sig in [("none", base_sig), ("pressure_nonneg", gated_sig)]:
            for reb in REBALANCE:
                for topk in TOP_K:
                    sim = simulate_portfolio(
                        y_v, sig, dates_val, codes_val,
                        rebalance_every=reb, top_k=topk,
                    )
                    s = summarize(sim)
                    s.update({"rule": rule, "gate": gate_name, "rebalance_every": reb, "top_k": topk})
                    rows.append(s)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "grid_summary.csv", index=False)
    df.to_csv(GOLD_DIR / "grid_summary.csv", index=False)

    # Gate candidates: compare risk threshold relaxed and production-like
    cols = ["rule", "gate", "rebalance_every", "top_k", "final_equity", "ann_ret", "sharpe", "max_dd", "mean_turnover", "worst_quarter", "mean_positions"]
    top = df.sort_values(["sharpe", "final_equity"], ascending=False).head(20)
    risk = df[(df["sharpe"] > 0.5) & (df["max_dd"] > -0.30) & (df["mean_turnover"] < 0.20) & (df["final_equity"] > 1.2)].sort_values("sharpe", ascending=False)

    lines = [
        "# Hetero + Transition Pressure Gate Grid",
        "",
        f"Pressure gate active days: {gate_days.mean():.1%}",
        "Gate: pressure_delta_20 >= 0. Holdout/test not used.",
        "",
        "## Top 20 by Sharpe",
        "",
        top[cols].round(4).to_markdown(index=False),
        "",
        "## Production-like risk gate (Sharpe>0.5, MaxDD>-30%, TO<0.20, Eq>1.2)",
        f"Count: {len(risk)}",
        "",
        risk[cols].round(4).to_markdown(index=False) if len(risk) else "None passed.",
        "",
        "## Best pressure-gated candidates",
        "",
        df[df["gate"] == "pressure_nonneg"].sort_values("sharpe", ascending=False).head(15)[cols].round(4).to_markdown(index=False),
    ]
    text = "\n".join(lines)
    (OUTPUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    (GOLD_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text[:3000])
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "gold_dir": str(GOLD_DIR)}, indent=2))


if __name__ == "__main__":
    main()
