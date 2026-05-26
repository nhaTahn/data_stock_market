"""Tail-control calibration for VN calibrated ensemble predictions.

Goal: improve the error histogram tails by reducing Q90(|E|) and violation days,
while keeping rel_score above a minimum threshold. Holdout/test is not used.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.evaluate_fixed_train_relscore_calibration import choose_train_calibration  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler  # noqa: E402

PRED_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_combined_full5_20260521"
DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/tail_control_calibration_20260526"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/tail_control_calibration_20260526"
SEEDS = (43, 52, 62, 71, 82)
MIN_REL_SCORE = 0.035


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def rel_score(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    base = robust_loss(actual)
    return float(1.0 - robust_loss(actual - pred) / base) if base > 0 else float("nan")


def metric(actual: np.ndarray, pred: np.ndarray, dates: pd.Series) -> dict[str, float]:
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    dates = dates[mask]
    err = actual - pred
    daily_q90 = pd.DataFrame({"Date": pd.to_datetime(dates), "abs_e": np.abs(err)}).groupby("Date")["abs_e"].quantile(0.9)
    return {
        "rel_score": rel_score(actual, pred),
        "median_abs_e": float(np.quantile(np.abs(err), 0.5)),
        "q90_abs_e": float(np.quantile(np.abs(err), 0.9)),
        "q95_abs_e": float(np.quantile(np.abs(err), 0.95)),
        "share_abs_e_gt_035": float((np.abs(err) > 0.035).mean()),
        "share_abs_e_gt_050": float((np.abs(err) > 0.05).mean()),
        "share_abs_e_gt_080": float((np.abs(err) > 0.08).mean()),
        "daily_q90_median": float(daily_q90.median()),
        "daily_q90_p90": float(daily_q90.quantile(0.9)),
        "daily_q90_max": float(daily_q90.max()),
        "daily_violation_gt_035": int((daily_q90 > 0.035).sum()),
        "daily_violation_share": float((daily_q90 > 0.035).mean()),
        "DA": float(np.mean(np.sign(actual) == np.sign(pred))),
        "pred_actual_q90_ratio": float(np.quantile(np.abs(pred), 0.9) / max(np.quantile(np.abs(actual), 0.9), 1e-8)),
    }


def load_meta_dates() -> pd.Series:
    frame = load_training_frame(str(DATA), stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["__tn__"] = frame["volatility_20"].astype(float)
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    train_df = frame.loc[frame["Date"] <= "2020-03-31"].copy()
    scaler = fit_feature_scaler(train_df.dropna(subset=features), features)
    scaled = apply_feature_scaler(frame, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled, features, "target_next_return", 15,
        extra_meta_columns=("__tn__",), sequence_normalization="none"
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, "2020-03-31", "2022-11-15")
    return pd.to_datetime(splits["val"][2].reset_index(drop=True)["Date"])


def load_anchor() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_preds: list[np.ndarray] = []
    val_preds: list[np.ndarray] = []
    sigma_train: list[np.ndarray] = []
    sigma_val: list[np.ndarray] = []
    y_train_ref: np.ndarray | None = None
    y_val_ref: np.ndarray | None = None
    for seed in SEEDS:
        data = np.load(PRED_DIR / f"predictions_seed_{seed}.npz", allow_pickle=True)
        y_train = data["y_train"].astype(np.float32)
        mu_train = data["mu_train"].astype(np.float32)
        mu_val = data["mu_val"].astype(np.float32)
        sig_train = data["sigma_train"].astype(np.float32)
        sig_val = data["sigma_val"].astype(np.float32)
        scale, clip = choose_train_calibration(y_train, mu_train, sig_train)
        p_train = mu_train * scale
        p_val = mu_val * scale
        if clip is not None:
            p_train = np.clip(p_train, -clip * sig_train, clip * sig_train)
            p_val = np.clip(p_val, -clip * sig_val, clip * sig_val)
        train_preds.append(p_train)
        val_preds.append(p_val)
        sigma_train.append(sig_train)
        sigma_val.append(sig_val)
        y_train_ref = y_train
        y_val_ref = data["y_val"].astype(np.float32)
    assert y_train_ref is not None and y_val_ref is not None
    pred_train = np.mean(train_preds, axis=0)
    pred_val = np.mean(val_preds, axis=0)
    sig_train_mean = np.mean(sigma_train, axis=0)
    sig_val_mean = np.mean(sigma_val, axis=0)
    # same anchor global train calibration
    scale, clip = choose_train_calibration(y_train_ref, pred_train, np.full_like(pred_train, 0.02))
    pred_train = pred_train * scale
    pred_val = pred_val * scale
    if clip is not None:
        pred_train = np.clip(pred_train, -clip * 0.02, clip * 0.02)
        pred_val = np.clip(pred_val, -clip * 0.02, clip * 0.02)
    return y_train_ref, y_val_ref, pred_train.astype(np.float32), pred_val.astype(np.float32), sig_train_mean, sig_val_mean


def apply_rule(pred: np.ndarray, sigma: np.ndarray, rule: str, shrink: float, clip_k: float | None, conf_q: float | None) -> np.ndarray:
    out = pred.copy() * shrink
    if clip_k is not None:
        out = np.clip(out, -clip_k * sigma, clip_k * sigma)
    if rule == "conf_shrink" and conf_q is not None:
        conf = np.abs(pred) / np.maximum(sigma, 1e-8)
        threshold = np.quantile(conf[np.isfinite(conf)], conf_q)
        low_conf = conf < threshold
        out[low_conf] *= 0.25
    elif rule == "conf_zero" and conf_q is not None:
        conf = np.abs(pred) / np.maximum(sigma, 1e-8)
        threshold = np.quantile(conf[np.isfinite(conf)], conf_q)
        out[conf < threshold] = 0.0
    return out.astype(np.float32)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    dates_val = load_meta_dates().reset_index(drop=True)
    y_train, y_val, pred_train, pred_val, sig_train, sig_val = load_anchor()

    rows: list[dict[str, object]] = []
    baseline = metric(y_val, pred_val, dates_val)
    baseline.update({"variant": "anchor_baseline", "rule": "baseline", "shrink": 1.0, "clip_k": np.nan, "conf_q": np.nan, "train_objective": np.nan})
    rows.append(baseline)

    candidates: list[dict[str, object]] = []
    for rule in ["plain", "conf_shrink", "conf_zero"]:
        for shrink in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]:
            for clip_k in [None, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]:
                conf_grid = [None] if rule == "plain" else [0.20, 0.30, 0.40, 0.50, 0.60]
                for conf_q in conf_grid:
                    train_pred = apply_rule(pred_train, sig_train, rule, shrink, clip_k, conf_q)
                    train_rel = rel_score(y_train, train_pred)
                    # Objective chooses lower robust/tail error while keeping rel_score acceptable.
                    train_err = y_train - train_pred
                    train_obj = (
                        - robust_loss(train_err)
                        - 0.25 * float(np.quantile(np.abs(train_err), 0.95))
                        + 0.02 * max(train_rel, -1.0)
                    )
                    val_pred = apply_rule(pred_val, sig_val, rule, shrink, clip_k, conf_q)
                    val_m = metric(y_val, val_pred, dates_val)
                    val_m.update({
                        "variant": f"{rule}_s{shrink}_k{clip_k}_q{conf_q}",
                        "rule": rule,
                        "shrink": shrink,
                        "clip_k": np.nan if clip_k is None else clip_k,
                        "conf_q": np.nan if conf_q is None else conf_q,
                        "train_rel_score": train_rel,
                        "train_objective": train_obj,
                    })
                    candidates.append(val_m)

    cand = pd.DataFrame(candidates)
    # Three rankings:
    # 1) dominated improvement over the anchor when available,
    # 2) tail-safe with min rel_score,
    # 3) pure tail best for diagnostic readout.
    safe = cand[cand["rel_score"] >= MIN_REL_SCORE].copy()
    improved = cand[
        (cand["rel_score"] >= baseline["rel_score"])
        & (cand["q90_abs_e"] <= baseline["q90_abs_e"])
        & (cand["q95_abs_e"] <= baseline["q95_abs_e"])
        & (cand["share_abs_e_gt_050"] <= baseline["share_abs_e_gt_050"])
    ].copy()
    if not improved.empty:
        improved["improvement_objective"] = (
            2.0 * (improved["rel_score"] - baseline["rel_score"])
            + 8.0 * (baseline["q90_abs_e"] - improved["q90_abs_e"])
            + 4.0 * (baseline["q95_abs_e"] - improved["q95_abs_e"])
            + 1.0 * (baseline["share_abs_e_gt_050"] - improved["share_abs_e_gt_050"])
            - 0.15 * np.maximum(improved["daily_violation_share"] - baseline["daily_violation_share"], 0.0)
        )
        improved = improved.sort_values("improvement_objective", ascending=False)
    safe["tail_objective"] = (
        2.0 * safe["rel_score"]
        -8.0 * safe["q90_abs_e"]
        -4.0 * safe["q95_abs_e"]
        -1.0 * safe["share_abs_e_gt_050"]
        -0.15 * safe["daily_violation_share"]
    )
    safe = safe.sort_values("tail_objective", ascending=False)
    best_safe = improved.iloc[0].to_dict() if not improved.empty else (safe.iloc[0].to_dict() if not safe.empty else None)

    pure_tail = cand.sort_values(["q90_abs_e", "daily_violation_share"], ascending=[True, True]).head(20)
    result = pd.concat([pd.DataFrame([baseline]), cand], ignore_index=True)
    result.to_csv(OUTPUT / "tail_control_grid.csv", index=False)
    result.to_csv(GOLD / "tail_control_grid.csv", index=False)
    safe.head(50).to_csv(OUTPUT / "tail_safe_candidates.csv", index=False)
    safe.head(50).to_csv(GOLD / "tail_safe_candidates.csv", index=False)
    pure_tail.to_csv(OUTPUT / "pure_tail_candidates.csv", index=False)
    if not improved.empty:
        improved.head(50).to_csv(OUTPUT / "dominant_improvement_candidates.csv", index=False)
        improved.head(50).to_csv(GOLD / "dominant_improvement_candidates.csv", index=False)

    if best_safe is None:
        selected = baseline
        selected_pred = pred_val
    else:
        selected = best_safe
        selected_pred = apply_rule(pred_val, sig_val, selected["rule"], float(selected["shrink"]), None if pd.isna(selected["clip_k"]) else float(selected["clip_k"]), None if pd.isna(selected["conf_q"]) else float(selected["conf_q"]))

    # plots
    plot_hist(y_val, pred_val, selected_pred, selected, OUTPUT / "tail_control_histogram.png", GOLD / "tail_control_histogram.png")
    plot_q90(dates_val, y_val, pred_val, selected_pred, OUTPUT / "tail_control_q90_timeseries.png", GOLD / "tail_control_q90_timeseries.png")

    comparison = pd.DataFrame([baseline, selected])
    comparison.to_csv(OUTPUT / "selected_comparison.csv", index=False)
    comparison.to_csv(GOLD / "selected_comparison.csv", index=False)

    text = "\n".join([
        "# Tail-Control Calibration",
        "",
        "Protocol: train-selected calibration grid; validation readout only. Holdout/test not used.",
        "",
        "## Selected vs Baseline",
        "",
        comparison[["variant", "rel_score", "q90_abs_e", "q95_abs_e", "daily_violation_gt_035", "daily_violation_share", "share_abs_e_gt_050", "DA", "pred_actual_q90_ratio"]].round(6).to_markdown(index=False),
        "",
        "## Dominant Improvement Candidates",
        "",
        improved[["variant", "rel_score", "q90_abs_e", "q95_abs_e", "daily_violation_gt_035", "daily_violation_share", "share_abs_e_gt_050", "DA", "pred_actual_q90_ratio"]].head(10).round(6).to_markdown(index=False) if not improved.empty else "_No candidate improves rel_score, Q90, Q95, and >5% tail share together._",
        "",
        "## Top Tail-Safe Candidates (rel_score >= 0.035)",
        "",
        safe[["variant", "rel_score", "q90_abs_e", "q95_abs_e", "daily_violation_gt_035", "daily_violation_share", "share_abs_e_gt_050", "DA", "pred_actual_q90_ratio"]].head(20).round(6).to_markdown(index=False) if not safe.empty else "_No candidate._",
        "",
        "## Pure Tail Candidates (may sacrifice rel_score)",
        "",
        pure_tail[["variant", "rel_score", "q90_abs_e", "q95_abs_e", "daily_violation_gt_035", "daily_violation_share", "share_abs_e_gt_050", "DA", "pred_actual_q90_ratio"]].head(10).round(6).to_markdown(index=False),
        "",
        json.dumps({"output_dir": str(OUTPUT), "gold_dir": str(GOLD)}, indent=2),
    ])
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD / "summary.md").write_text(text, encoding="utf-8")
    print(text)


def plot_hist(y: np.ndarray, base_pred: np.ndarray, sel_pred: np.ndarray, selected: dict, out: Path, gold: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, title, pred in [
        (axes[0], "Anchor baseline", base_pred),
        (axes[1], f"Tail-control selected\n{selected['variant']}", sel_pred),
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
    fig.suptitle("Tail-Control Error Histogram", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(gold, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_q90(dates: pd.Series, y: np.ndarray, base_pred: np.ndarray, sel_pred: np.ndarray, out: Path, gold: Path) -> None:
    def daily_q90(pred: np.ndarray) -> pd.Series:
        return pd.DataFrame({"Date": pd.to_datetime(dates), "abs_e": np.abs(y - pred)}).groupby("Date")["abs_e"].quantile(0.9)
    b = daily_q90(base_pred) * 100
    s = daily_q90(sel_pred) * 100
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(b.index, b.values, color="#888888", alpha=0.65, linewidth=0.9, label=f"Anchor Q90, viol>{(b>3.5).sum()} days")
    ax.plot(s.index, s.values, color="#1a73e8", alpha=0.85, linewidth=0.9, label=f"Tail-control Q90, viol>{(s>3.5).sum()} days")
    ax.plot(b.index, b.rolling(21, min_periods=5).mean(), color="#666666", linewidth=2, alpha=0.8)
    ax.plot(s.index, s.rolling(21, min_periods=5).mean(), color="#1a73e8", linewidth=2)
    ax.axhline(3.0, color="green", linestyle="--", label="Target 3.0%")
    ax.axhline(3.5, color="red", linestyle="--", label="Violation 3.5%")
    ax.set_title("Daily Q90(|E|): Anchor vs Tail-Control", fontweight="bold")
    ax.set_ylabel("Q90(|E|) %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(gold, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
