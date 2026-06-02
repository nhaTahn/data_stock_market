"""P3: σ-based selection layer probe.

Uses saved predictions (mu, sigma) from P2 hetero_combined runs to test
selection rules that cover 30–70% of stock-days. Goal: find a rule that
simultaneously achieves rel_score > 0.037 (full coverage) on the selected
subset AND reduces spike days.

Selection rules evaluated (all calibrated on TRAIN, applied on VAL):
  - sigma_q{20,30,40,50}     : keep rows where sigma <= sigma_q{X} (low uncertainty)
  - abs_mu_q{50,60,70}       : keep rows where |mu| >= abs_mu_q{X} (strong signal)
  - combo_s{q}m{q}           : sigma <= sigma_q{X} AND |mu| >= abs_mu_q{Y}
  - daily_top_n_{5,10,15}    : per day, keep top N stocks ranked by |mu|/sigma
  - daily_bottom_sigma_{25,50}: per day, keep stocks with lowest sigma (25th/50th pct daily)
  - daily_top_mu_low_sigma   : per day, keep stocks with |mu| >= mu_q70_daily AND sigma <= sigma_q40_daily

Metrics: rel_score, daily_q90_max, spike_days_ge_8pct, DA, pred/act ratio, coverage.
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

PRED_DIR = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history"
    / "training_runs" / "reports" / "sigma_shrinkage_probe_20260521"
)
OUTPUT_DIR = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history"
    / "training_runs" / "reports" / "sigma_selection_probe_20260521"
)
GOLD_DIR = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "sigma_selection_probe_20260521"


def robust_loss(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    return float(np.quantile(np.abs(v), 0.5) + 0.5 * np.quantile(np.abs(v), 0.9))


def rel_score_fn(actual: np.ndarray, pred: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float(1.0 - robust_loss(actual - pred) / base)


def evaluate(actual: np.ndarray, pred: np.ndarray, dates: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() == 0:
        return {"rel_score": float("nan"), "coverage": 0.0, "spike_days_ge_8pct": 0,
                "daily_q90_max": float("nan"), "directional_accuracy": float("nan"),
                "pred_actual_q90_ratio": float("nan")}
    a = actual[mask]
    p = pred[mask]
    d = dates[mask]
    ae = np.abs(a - p)
    ser = pd.Series(ae, index=d)
    dq90 = ser.groupby(level=0).quantile(0.90)
    return {
        "rel_score": rel_score_fn(a, p),
        "daily_q90_max": float(dq90.max()),
        "daily_q90_p90": float(dq90.quantile(0.90)),
        "spike_days_ge_8pct": int((dq90 >= 0.08).sum()),
        "spike_days_ge_5pct": int((dq90 >= 0.05).sum()),
        "directional_accuracy": float(np.mean(np.sign(a) == np.sign(p))),
        "pred_actual_q90_ratio": float(
            np.quantile(np.abs(p), 0.9) / max(np.quantile(np.abs(a), 0.9), 1e-8)
        ),
        "median_abs_error": float(np.quantile(ae, 0.5)),
        "coverage": float(mask.mean()),
    }


def load_seed(seed: int) -> dict:
    npz = np.load(PRED_DIR / f"predictions_seed_{seed}.npz")
    return {k: npz[k] for k in npz.files}


def load_val_meta(seed: int) -> pd.DataFrame:
    """Reconstruct meta_val from predictions CSV (for dates)."""
    # Dates are in per_seed results but we need per-sample dates.
    # We need to load them from the P2 artifact — which did NOT save meta.
    # Instead, regenerate them the same way as in P2 training script.
    # We stored predictions but not dates — use the static dataset rebuild.
    return None  # will be handled via per-sample rebuild below


def build_date_index_for_val(train_end: str, val_end: str, data_path: Path) -> np.ndarray:
    """Return dates array aligned with the val predictions npz."""
    from src.models.config import DEFAULT_FEATURE_COLUMNS
    from src.models.training.datasets import build_sequence_dataset, split_frame_by_date, split_sequence_dataset
    from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler
    from src.models.training.pipeline import load_frame as load_training_frame
    feature_columns = DEFAULT_FEATURE_COLUMNS
    frame = load_training_frame(data_path, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["__tn__"] = frame["volatility_20"].astype(float)
    train_df, _, _ = split_frame_by_date(frame, train_end, val_end)
    scaler = fit_feature_scaler(train_df.dropna(subset=feature_columns), feature_columns)
    scaled = apply_feature_scaler(frame, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled, feature_columns, "target_next_return", 15,
        extra_meta_columns=("__tn__",), sequence_normalization="none",
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, train_end, val_end)
    _, y_val, meta_val = splits["val"]
    return meta_val["Date"].values, y_val


def apply_selection_rules(
    mu_train: np.ndarray, sigma_train: np.ndarray,
    mu_val: np.ndarray, sigma_val: np.ndarray,
    dates_val: np.ndarray,
) -> dict[str, np.ndarray]:
    """Returns dict of rule_name -> bool mask over val rows."""
    n = len(mu_val)
    full = np.ones(n, dtype=bool)

    # Train-derived thresholds
    sq = {q: float(np.quantile(sigma_train, q/100)) for q in [20, 30, 40, 50, 60, 70]}
    aq = {q: float(np.quantile(np.abs(mu_train), q/100)) for q in [40, 50, 60, 70, 80]}

    rules: dict[str, np.ndarray] = {"full_coverage": full}
    for pct in [20, 30, 40, 50]:
        rules[f"sigma_q{pct}"] = sigma_val <= sq[pct]
    for pct in [50, 60, 70, 80]:
        rules[f"abs_mu_q{pct}"] = np.abs(mu_val) >= aq[pct]

    # Combo rules
    for spct in [30, 40]:
        for mpct in [60, 70]:
            name = f"combo_s{spct}_m{mpct}"
            rules[name] = (sigma_val <= sq[spct]) & (np.abs(mu_val) >= aq[mpct])

    # Confidence ratio: |mu| / sigma
    conf_ratio = np.abs(mu_val) / np.maximum(sigma_val, 1e-8)
    conf_ratio_train = np.abs(mu_train) / np.maximum(sigma_train, 1e-8)
    for pct in [50, 60, 70]:
        thr = float(np.quantile(conf_ratio_train, pct/100))
        rules[f"conf_ratio_q{pct}"] = conf_ratio >= thr

    # Per-day top-N by |mu|/sigma (confidence ratio descending)
    val_df = pd.DataFrame({
        "date": dates_val, "mu": mu_val, "sigma": sigma_val, "conf_ratio": conf_ratio
    })
    val_df.index = np.arange(n)
    for top_n in [5, 10, 15, 20]:
        mask_arr = np.zeros(n, dtype=bool)
        for _, grp in val_df.groupby("date"):
            keep = grp.nlargest(top_n, "conf_ratio").index
            mask_arr[keep] = True
        rules[f"daily_top{top_n}_conf_ratio"] = mask_arr

    # Per-day bottom-sigma (keep stocks with lowest sigma each day)
    for pct in [25, 50]:
        mask_arr = np.zeros(n, dtype=bool)
        for _, grp in val_df.groupby("date"):
            thresh = grp["sigma"].quantile(pct/100)
            keep = grp.index[grp["sigma"] <= thresh]
            mask_arr[keep] = True
        rules[f"daily_bottom_sigma_{pct}pct"] = mask_arr

    return rules


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
    print("Loading val dates (one-time)...")
    dates_val, y_val = build_date_index_for_val("2020-03-31", "2022-11-15", data_path)
    print(f"Val samples: {len(y_val)}")

    seeds = [43, 52, 71]
    all_rows: list[dict] = []

    for seed in seeds:
        print(f"Seed {seed}")
        preds = load_seed(seed)
        mu_train = preds["mu_train"]
        sigma_train = preds["sigma_train"]
        mu_val = preds["mu_val"]
        sigma_val = preds["sigma_val"]
        y_val_seed = preds["y_val"]
        # Sanity check
        assert len(y_val_seed) == len(y_val), f"Mismatch: {len(y_val_seed)} vs {len(y_val)}"
        rules = apply_selection_rules(mu_train, sigma_train, mu_val, sigma_val, dates_val)
        for name, mask in rules.items():
            m = evaluate(y_val_seed, mu_val, dates_val, mask)
            m.update({"seed": seed, "rule": name})
            all_rows.append(m)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "results_per_seed.csv", index=False)

    agg_rows = []
    for rule, g in df.groupby("rule"):
        row = {"rule": rule, "n_seeds": len(g)}
        for c in ["rel_score","daily_q90_max","spike_days_ge_8pct","spike_days_ge_5pct",
                   "directional_accuracy","pred_actual_q90_ratio","coverage","median_abs_error"]:
            if c in g.columns:
                v = g[c].astype(float)
                row[f"{c}_mean"] = float(v.mean())
                row[f"{c}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows).sort_values("rel_score_mean", ascending=False)
    agg.to_csv(OUTPUT_DIR / "results_aggregate.csv", index=False)
    agg.to_csv(GOLD_DIR / "results_aggregate.csv", index=False)

    # Write readable table
    key_cols = ["rule","rel_score_mean","rel_score_std","daily_q90_max_mean",
                "spike_days_ge_8pct_mean","directional_accuracy_mean",
                "pred_actual_q90_ratio_mean","coverage_mean"]
    key_cols = [c for c in key_cols if c in agg.columns]
    md = agg[key_cols].round(4).to_markdown(index=False)
    text = "\n".join([
        "# σ-Selection Layer Probe (P3) Readout",
        "",
        "Scope: VN val only. Holdout/test not used.",
        "Uses saved (mu, sigma) from hetero_combined P2 seeds 43,52,71.",
        "",
        "## Rules evaluated (all thresholds calibrated on TRAIN)",
        "- sigma_q{X}: keep rows sigma ≤ train-sigma percentile X",
        "- abs_mu_q{X}: keep rows |mu| ≥ train-|mu| percentile X",
        "- combo_s{X}_m{Y}: sigma_qX AND abs_mu_qY",
        "- conf_ratio_q{X}: |mu|/sigma ≥ train-conf_ratio percentile X",
        "- daily_top{N}_conf_ratio: per-day top-N stocks by |mu|/sigma",
        "- daily_bottom_sigma_{X}pct: per-day bottom X% sigma stocks",
        "",
        "## Aggregate (3 seeds, sorted by rel_score)",
        "",
        md,
        "",
        "## Reference baselines",
        "- stressaux_w20 full: rel_score 0.0248, spike_8pct 7.7, daily_q90_max 9.44%",
        "- hetero_combined full: rel_score 0.0372, spike_8pct 13.7, daily_q90_max 11.23%",
    ])
    (OUTPUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    (GOLD_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "gold_dir": str(GOLD_DIR)}, indent=2))


if __name__ == "__main__":
    main()
