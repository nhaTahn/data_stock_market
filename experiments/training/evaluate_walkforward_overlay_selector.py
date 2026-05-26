"""Walk-forward portfolio overlay selector.

Uses existing VN validation daily policy returns.  For each 21-trading-day block,
selects the best (policy, overlay) using only prior validation blocks, then applies
that choice to the next block. This is not holdout, but avoids full-validation
look-ahead for overlay selection.
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
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/walkforward_overlay_selector_20260526"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/walkforward_overlay_selector_20260526"

MIN_BASE_SHARPE = 1.0
FOLD_DAYS = 21
MIN_HISTORY_DAYS = 84


def sharpe_ann(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 10 or r.std(ddof=1) <= 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def max_dd_from_rets(rets: pd.Series) -> float:
    if rets.empty:
        return float("nan")
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def summarize(rets: pd.Series) -> dict[str, float]:
    eq = (1 + rets).cumprod()
    return {
        "n_days": int(len(rets)),
        "active_days": int((rets != 0).sum()),
        "final_equity": float(eq.iloc[-1]) if len(eq) else float("nan"),
        "sharpe": sharpe_ann(rets),
        "max_dd": max_dd_from_rets(rets),
        "mean_daily_return": float(rets.mean()) if len(rets) else float("nan"),
        "daily_vol": float(rets.std(ddof=1)) if len(rets) > 1 else float("nan"),
    }


def dd_stop(rets: pd.Series, thresh: float, pause_days: int) -> pd.Series:
    out = rets.copy()
    eq = (1 + rets).cumprod()
    pause = 0
    for i in range(len(out)):
        if pause > 0:
            out.iloc[i] = 0.0
            pause -= 1
            continue
        window = eq.iloc[max(0, i - 20): i + 1]
        dd = eq.iloc[i] / window.max() - 1 if len(window) else 0.0
        if dd < thresh:
            out.iloc[i] = 0.0
            pause = pause_days
    return out


def vol_scale(rets: pd.Series, target_vol: float, cap: float = 2.0) -> pd.Series:
    vol = rets.rolling(20, min_periods=5).std() * np.sqrt(252)
    scale = (target_vol / vol.replace(0, np.nan)).clip(0.1, cap).shift(1)
    return (rets * scale).fillna(rets)


def regime_gate(rets: pd.Series, lookback: int, min_pos: float) -> pd.Series:
    signal = (rets > 0).rolling(lookback, min_periods=3).mean().shift(1)
    return rets.where(signal >= min_pos, 0.0)


def apply_overlay(rets: pd.Series, label: str) -> pd.Series:
    if label == "baseline":
        return rets
    if label == "dd_stop_t8_p5":
        return dd_stop(rets, -0.08, 5)
    if label == "dd_stop_t10_p10":
        return dd_stop(rets, -0.10, 10)
    if label == "dd_vol_t8_p5_tv7":
        return vol_scale(dd_stop(rets, -0.08, 5), 0.07)
    if label == "dd_vol_t10_p10_tv10":
        return vol_scale(dd_stop(rets, -0.10, 10), 0.10)
    if label == "regime_gate_lb10_m55":
        return regime_gate(rets, 10, 0.55)
    if label == "regime_gate_lb15_m55":
        return regime_gate(rets, 15, 0.55)
    raise ValueError(label)


def objective(rets: pd.Series) -> float:
    stats = summarize(rets)
    if not np.isfinite(stats["sharpe"]):
        return float("-inf")
    # Penalize large drawdowns and low activity.
    activity = stats["active_days"] / max(stats["n_days"], 1)
    return float(stats["sharpe"] + 2.0 * stats["max_dd"] + 0.25 * activity)


def make_fold_id(dates: pd.Series) -> pd.Series:
    unique = pd.Series(pd.to_datetime(dates).unique()).sort_values().reset_index(drop=True)
    mapping = {date: idx // FOLD_DAYS for idx, date in enumerate(unique)}
    return pd.to_datetime(dates).map(mapping)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)

    policy_df = pd.read_csv(POLICY_CSV)
    policies = policy_df.loc[policy_df["mean_sharpe"] >= MIN_BASE_SHARPE, "policy"].tolist()
    daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
    daily = daily[daily["policy"].isin(policies)]
    daily = daily.groupby(["Date", "policy"], as_index=False)["net_return"].mean()
    daily["fold_id"] = make_fold_id(daily["Date"])
    overlays = [
        "baseline", "dd_stop_t8_p5", "dd_stop_t10_p10", "dd_vol_t8_p5_tv7",
        "dd_vol_t10_p10_tv10", "regime_gate_lb10_m55", "regime_gate_lb15_m55",
    ]

    # Precompute full overlay series for each candidate to avoid recompute in loop.
    series_map: dict[tuple[str, str], pd.Series] = {}
    for policy in policies:
        base = daily[daily["policy"] == policy].sort_values("Date").set_index("Date")["net_return"]
        for overlay in overlays:
            series_map[(policy, overlay)] = apply_overlay(base, overlay)

    all_dates = sorted(daily["Date"].unique())
    fold_ids = sorted(daily["fold_id"].unique())
    rows: list[dict[str, object]] = []
    selected_parts: list[pd.Series] = []

    fallback_policy = "full_none_r10_k10_m1" if "full_none_r10_k10_m1" in policies else policies[0]
    fallback_overlay = "dd_stop_t8_p5"

    for fold_id in fold_ids:
        block_dates = sorted(daily.loc[daily["fold_id"] == fold_id, "Date"].unique())
        if not block_dates:
            continue
        start = pd.Timestamp(block_dates[0])
        history_dates = [d for d in all_dates if d < np.datetime64(start)]
        if len(history_dates) < MIN_HISTORY_DAYS:
            policy, overlay = fallback_policy, fallback_overlay
            train_obj = float("nan")
        else:
            best = (float("-inf"), fallback_policy, fallback_overlay)
            hist_index = pd.DatetimeIndex(history_dates)
            for candidate_policy in policies:
                for candidate_overlay in overlays:
                    hist_returns = series_map[(candidate_policy, candidate_overlay)].reindex(hist_index).fillna(0.0)
                    score = objective(hist_returns)
                    if score > best[0]:
                        best = (score, candidate_policy, candidate_overlay)
            train_obj, policy, overlay = best
        block_index = pd.DatetimeIndex(block_dates)
        block_rets = series_map[(policy, overlay)].reindex(block_index).fillna(0.0)
        selected_parts.append(block_rets)
        block_stats = summarize(block_rets)
        rows.append({
            "fold_id": int(fold_id),
            "start": pd.Timestamp(block_dates[0]).date().isoformat(),
            "end": pd.Timestamp(block_dates[-1]).date().isoformat(),
            "selected_policy": policy,
            "selected_overlay": overlay,
            "train_objective": train_obj,
            **block_stats,
        })

    selected_returns = pd.concat(selected_parts).sort_index()
    selected_returns.name = "net_return"
    selected_daily = selected_returns.reset_index().rename(columns={"index": "Date"})
    selected_daily.to_csv(OUTPUT / "walkforward_selected_daily_returns.csv", index=False)
    selected_daily.to_csv(GOLD / "walkforward_selected_daily_returns.csv", index=False)
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUTPUT / "walkforward_fold_choices.csv", index=False)
    fold_df.to_csv(GOLD / "walkforward_fold_choices.csv", index=False)

    final_stats = summarize(selected_returns)
    # Benchmarks from important static policies/overlays
    bench_rows: list[dict[str, object]] = [{"candidate": "walkforward_selector", **final_stats}]
    for policy, overlay in [
        ("full_none_r10_k10_m1", "baseline"),
        ("full_none_r10_k10_m1", "dd_stop_t8_p5"),
        ("full_none_r20_k10_m1", "dd_vol_t8_p5_tv7"),
        ("abs_mu_q60_pressure_nonneg_r10_k8_m1", "dd_stop_t8_p5"),
    ]:
        if (policy, overlay) in series_map:
            bench_rows.append({"candidate": f"{policy}__{overlay}", **summarize(series_map[(policy, overlay)])})
    bench = pd.DataFrame(bench_rows).sort_values("sharpe", ascending=False)
    bench.to_csv(OUTPUT / "walkforward_benchmark.csv", index=False)
    bench.to_csv(GOLD / "walkforward_benchmark.csv", index=False)

    text = "\n".join([
        "# Walk-Forward Overlay Selector",
        "",
        "Protocol: for each 21-day validation block, select policy/overlay using prior validation days only. Holdout/test not used.",
        "",
        "## Final Benchmark",
        "",
        bench.round(4).to_markdown(index=False),
        "",
        "## Fold Choices",
        "",
        fold_df.head(40).round(4).to_markdown(index=False),
        "",
        json.dumps({"output_dir": str(OUTPUT), "gold_dir": str(GOLD)}, indent=2),
    ])
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
