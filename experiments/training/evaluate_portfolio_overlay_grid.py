"""Portfolio overlay grid on existing daily policy returns.

Tests risk overlays on top of the best baseline policies from
hetero_long_finetune_batch. No new model training; pure post-processing.

Overlays:
  1. rolling_dd_stop:  go to cash (return=0) for N days after drawdown < -thresh
  2. vol_scale:        scale daily return by target_vol / realised_vol_20
  3. regime_gate:      only active when 5-day fold equity trend >= 0

Scope: VN validation 2020-04-01..2022-11-15.  Holdout/test not used.
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

DAILY_CSV = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_long_finetune_batch_20260522/daily_policy_returns.csv"
POLICY_CSV = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_long_finetune_batch_20260522/policy_summary.csv"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/portfolio_overlay_grid_20260526"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/portfolio_overlay_grid_20260526"

# Focus on policies with mean_sharpe > 1.0 to keep runtime manageable
MIN_BASE_SHARPE = 1.0
# Look at mean across seeds for each (policy, date)
SEED_AGG = "mean"


def sharpe_ann(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252))


def max_dd(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1).min())


def cagr(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return float("nan")
    years = len(eq) / 252
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)


def summarize(net_rets: pd.Series, label: str, policy: str) -> dict[str, object]:
    eq = (1 + net_rets).cumprod()
    return {
        "label": label,
        "policy": policy,
        "sharpe": sharpe_ann(net_rets),
        "max_dd": max_dd(eq),
        "final_equity": float(eq.iloc[-1]) if not eq.empty else float("nan"),
        "cagr": cagr(eq),
        "positive_days": int((net_rets > 0).sum()),
        "n_days": int(len(net_rets)),
        "n_active_days": int((net_rets != 0).sum()),
    }


def rolling_dd_stop(rets: pd.Series, dd_thresh: float = -0.10, pause_days: int = 10) -> pd.Series:
    """Zero out returns for `pause_days` after rolling equity drops below dd_thresh."""
    eq = (1 + rets).cumprod()
    in_drawdown_stop = pd.Series(False, index=rets.index)
    pause_counter = 0
    for i in range(len(rets)):
        if pause_counter > 0:
            in_drawdown_stop.iloc[i] = True
            pause_counter -= 1
        else:
            rolling_eq = eq.iloc[max(0, i - 20):i + 1]
            dd = (eq.iloc[i] / rolling_eq.max() - 1) if len(rolling_eq) > 0 else 0.0
            if dd < dd_thresh:
                in_drawdown_stop.iloc[i] = True
                pause_counter = pause_days
    return rets.where(~in_drawdown_stop, 0.0)


def vol_scale(rets: pd.Series, target_vol: float = 0.10, window: int = 20, cap: float = 2.0) -> pd.Series:
    """Scale returns to target daily annualised volatility."""
    ann_vol = rets.rolling(window, min_periods=5).std() * np.sqrt(252)
    scale = (target_vol / ann_vol.replace(0, np.nan)).clip(0.1, cap)
    return (rets * scale.shift(1)).fillna(rets)


def regime_gate_score(rets: pd.Series, lookback: int = 10, min_active: float = 0.5) -> pd.Series:
    """Only active when fraction of positive days over last `lookback` days >= min_active."""
    pos = (rets > 0).rolling(lookback, min_periods=3).mean().shift(1)
    return rets.where(pos >= min_active, 0.0)


def apply_overlays(base_rets: pd.Series, policy: str) -> list[dict[str, object]]:
    rows = [summarize(base_rets, "baseline", policy)]
    # rolling_dd_stop variants
    for thresh, pause in [(-0.08, 5), (-0.10, 10), (-0.12, 15), (-0.15, 10)]:
        label = f"dd_stop_t{int(abs(thresh)*100)}_p{pause}"
        rows.append(summarize(rolling_dd_stop(base_rets, thresh, pause), label, policy))
    # vol_scale variants
    for tvol in [0.07, 0.10, 0.12]:
        label = f"vol_scale_tv{int(tvol*100)}"
        rows.append(summarize(vol_scale(base_rets, tvol), label, policy))
    # regime_gate variants
    for lb, mact in [(10, 0.5), (10, 0.55), (15, 0.55)]:
        label = f"regime_gate_lb{lb}_m{int(mact*100)}"
        rows.append(summarize(regime_gate_score(base_rets, lb, mact), label, policy))
    # combined: dd_stop + vol_scale
    for thresh, pause, tvol in [(-0.10, 10, 0.10), (-0.08, 5, 0.07)]:
        r = rolling_dd_stop(base_rets, thresh, pause)
        r = vol_scale(r, tvol)
        label = f"dd_vol_t{int(abs(thresh)*100)}_p{pause}_tv{int(tvol*100)}"
        rows.append(summarize(r, label, policy))
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)

    policy_df = pd.read_csv(POLICY_CSV)
    top_policies = policy_df.loc[policy_df["mean_sharpe"] >= MIN_BASE_SHARPE, "policy"].unique().tolist()
    print(f"Top policies to test: {len(top_policies)}")

    daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
    # Aggregate across seeds
    grp = daily.groupby(["Date", "policy"])["net_return"]
    if SEED_AGG == "mean":
        daily_agg = grp.mean().reset_index()
    else:
        daily_agg = grp.median().reset_index()

    all_rows: list[dict] = []
    for policy in top_policies:
        pdata = daily_agg.loc[daily_agg["policy"] == policy].sort_values("Date")
        if pdata.empty:
            continue
        rets = pdata.set_index("Date")["net_return"]
        rows = apply_overlays(rets, policy)
        all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    result.to_csv(OUTPUT / "overlay_grid.csv", index=False)
    result.to_csv(GOLD / "overlay_grid.csv", index=False)

    # Find top candidates: sharpe > baseline, max_dd > -0.25
    baseline = result[result["label"] == "baseline"].rename(columns={"sharpe": "base_sharpe", "max_dd": "base_dd"})
    merged = result.merge(baseline[["policy", "base_sharpe", "base_dd"]], on="policy")
    improved = merged[
        (merged["label"] != "baseline")
        & (merged["sharpe"] > merged["base_sharpe"])
        & (merged["max_dd"] > -0.25)
    ].sort_values("sharpe", ascending=False)

    improved.to_csv(OUTPUT / "improved_candidates.csv", index=False)
    improved.to_csv(GOLD / "improved_candidates.csv", index=False)

    # Summary table: for each policy, compare best overlay vs baseline
    summary_rows: list[dict] = []
    for policy in top_policies:
        sub = result[result["policy"] == policy]
        base_row = sub[sub["label"] == "baseline"].iloc[0] if not sub[sub["label"] == "baseline"].empty else None
        best_row = sub[sub["label"] != "baseline"].sort_values("sharpe", ascending=False).iloc[0] if len(sub) > 1 else None
        if base_row is None:
            continue
        row: dict[str, object] = {
            "policy": policy,
            "base_sharpe": base_row["sharpe"],
            "base_max_dd": base_row["max_dd"],
            "base_final_equity": base_row["final_equity"],
        }
        if best_row is not None:
            row.update({
                "best_overlay": best_row["label"],
                "best_sharpe": best_row["sharpe"],
                "best_max_dd": best_row["max_dd"],
                "best_final_equity": best_row["final_equity"],
                "sharpe_delta": float(best_row["sharpe"]) - float(base_row["sharpe"]),
                "dd_delta": float(best_row["max_dd"]) - float(base_row["max_dd"]),
            })
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("best_sharpe", ascending=False)
    summary.to_csv(OUTPUT / "policy_overlay_summary.csv", index=False)
    summary.to_csv(GOLD / "policy_overlay_summary.csv", index=False)

    text = "\n".join([
        "# Portfolio Overlay Grid",
        "",
        "Protocol: applied on existing fold-averaged daily returns. Holdout/test not used.",
        f"Base policies tested: {len(top_policies)}. Overlays per policy: {len(all_rows) // len(top_policies) if top_policies else 0}.",
        "",
        "## Top Improved Candidates (sharpe > baseline, max_dd > -0.25)",
        "",
        (improved[["policy", "label", "sharpe", "max_dd", "final_equity", "cagr", "base_sharpe", "base_dd"]].head(20).round(4).to_markdown(index=False)
         if not improved.empty else "_No policy passes both screens._"),
        "",
        "## Policy Overlay Summary (best overlay per policy)",
        "",
        summary.head(20).round(4).to_markdown(index=False),
        "",
        json.dumps({"output_dir": str(OUTPUT), "gold_dir": str(GOLD)}, indent=2),
    ])
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
