"""Train-only calibration/gating diagnostics for dynamic VN30 panel runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    "vn30_dynamic_panel_run",
    "vn30_dynamic_panel_tuned_5seed",
]
OUT = ROOT / "gold/vn_transition_pressure_20260512/plots/vn30_dynamic_panel_tuned_5seed"


def robust_loss(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def rel_score(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    base = robust_loss(actual[mask])
    return float(1.0 - robust_loss((actual - pred)[mask]) / base) if base > 0 else float("nan")


def metric_row(frame: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    actual = frame["actual"].to_numpy(float)
    pred = frame["prediction"].to_numpy(float)
    err = actual - pred
    daily_q90 = frame.assign(abs_error=np.abs(err)).groupby("Date")["abs_error"].quantile(0.9)
    return {
        "label": label,
        "n_obs": int(len(frame)),
        "n_days": int(frame["Date"].nunique()),
        "rel_score": rel_score(actual, pred),
        "mae": float(np.abs(err).mean()),
        "q90_abs_e": float(np.quantile(np.abs(err), 0.9)),
        "q95_abs_e": float(np.quantile(np.abs(err), 0.95)),
        "share_abs_e_gt_035": float((np.abs(err) > 0.035).mean()),
        "daily_q90_gt_035": int((daily_q90 > 0.035).sum()),
        "da": float((np.sign(actual) == np.sign(pred)).mean()),
    }


def load_run(run_name: str) -> pd.DataFrame:
    path = ROOT / f"data/processed/assets/data_info_vn/history/training_runs/{run_name}/reports/core/predictions.csv"
    return pd.read_csv(path, parse_dates=["Date"])


def calibrate_scale(train: pd.DataFrame, val: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    scales = np.linspace(0.2, 1.4, 61)
    actual = train["actual"].to_numpy(float)
    pred = train["prediction"].to_numpy(float)
    best_scale = max(scales, key=lambda s: rel_score(actual, pred * s))
    out = val.copy()
    out["prediction"] = out["prediction"] * float(best_scale)
    return float(best_scale), out


def build_wide(df: pd.DataFrame, split: str) -> pd.DataFrame:
    part = df[df["split"].eq(split)].copy()
    keys = ["code", "Date"]
    actual = part.drop_duplicates(keys)[[*keys, "actual"]]
    wide = part.pivot_table(index=keys, columns="model", values="prediction", aggfunc="first").reset_index()
    return actual.merge(wide, on=keys, how="inner")


def evaluate_selective(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    train_wide = build_wide(df, "train")
    val_wide = build_wide(df, "val")
    seed_cols = [c for c in train_wide.columns if str(c).startswith("panel_lstm_seed_")]
    for frame in [train_wide, val_wide]:
        frame["prediction"] = frame["panel_lstm_ensemble"]
        frame["seed_std"] = frame[seed_cols].std(axis=1)
        frame["abs_pred"] = frame["prediction"].abs()
    rows = []
    best_thr = 1.0
    best_score = -np.inf
    for q in np.linspace(0.3, 1.0, 15):
        thr = float(train_wide["seed_std"].quantile(q))
        tr_sel = train_wide[train_wide["seed_std"] <= thr]
        score = rel_score(tr_sel["actual"].to_numpy(float), tr_sel["prediction"].to_numpy(float)) if len(tr_sel) else -np.inf
        if score > best_score and len(tr_sel) >= 1000:
            best_score = score
            best_thr = thr
    val_sel = val_wide[val_wide["seed_std"] <= best_thr].copy()
    rows.append(metric_row(val_wide.rename(columns={"panel_lstm_ensemble":"_unused"}), "ensemble_full_val"))
    rows.append(metric_row(val_sel, f"ensemble_selective_seedstd_le_{best_thr:.5f}"))
    return pd.DataFrame(rows), best_thr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for run in RUNS:
        df = load_run(run)
        with open(ROOT / f"data/processed/assets/data_info_vn/history/training_runs/{run}/reports/core/metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append({"run": run, "variant": "best_by_val", **metrics["panel_lstm_best_by_val"]["val"]})
        rows.append({"run": run, "variant": "ensemble", **metrics["panel_lstm_ensemble"]["val"]})
        train = df[(df["split"].eq("train")) & (df["model"].eq("panel_lstm_ensemble"))].copy()
        val = df[(df["split"].eq("val")) & (df["model"].eq("panel_lstm_ensemble"))].copy()
        scale, cal_val = calibrate_scale(train, val)
        cal_row = metric_row(cal_val, f"ensemble_scaled_{scale:.3f}")
        rows.append({"run": run, "variant": "train_scaled_ensemble", "scale": scale, **{k:v for k,v in cal_row.items() if k != "label"}})
        if run == "vn30_dynamic_panel_tuned_5seed":
            sel, thr = evaluate_selective(df)
            sel["run"] = run
            sel["threshold"] = thr
            sel.to_csv(OUT / "selective_seedstd_validation.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "vn30_dynamic_run_comparison.csv", index=False)
    print(summary.round(6).to_markdown(index=False))
    print(OUT / "vn30_dynamic_run_comparison.csv")
    if (OUT / "selective_seedstd_validation.csv").exists():
        print(pd.read_csv(OUT / "selective_seedstd_validation.csv").round(6).to_markdown(index=False))

if __name__ == "__main__":
    main()
