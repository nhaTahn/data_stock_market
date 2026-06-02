"""Tail-risk aware gating for VN calibrated ensemble predictions.

This diagnostic trains a small classifier on train-period prediction errors to
identify observations likely to have large absolute errors. It then evaluates
validation-only abstain/shrink gates. Holdout/test is not used.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/tail_risk_aware_gate_20260526"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/tail_risk_aware_gate_20260526"
SEEDS = (43, 52, 62, 71, 82)
TAIL_THRESHOLD = 0.035


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


def metric(actual: np.ndarray, pred: np.ndarray, dates: pd.Series, risk: np.ndarray | None = None) -> dict[str, float]:
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    dates = dates[mask]
    err = actual - pred
    daily_q90 = pd.DataFrame({"Date": pd.to_datetime(dates), "abs_e": np.abs(err)}).groupby("Date")["abs_e"].quantile(0.9)
    row = {
        "rel_score": rel_score(actual, pred),
        "median_abs_e": float(np.quantile(np.abs(err), 0.5)),
        "q90_abs_e": float(np.quantile(np.abs(err), 0.9)),
        "q95_abs_e": float(np.quantile(np.abs(err), 0.95)),
        "share_abs_e_gt_035": float((np.abs(err) > 0.035).mean()),
        "share_abs_e_gt_050": float((np.abs(err) > 0.05).mean()),
        "share_abs_e_gt_080": float((np.abs(err) > 0.08).mean()),
        "daily_q90_median": float(daily_q90.median()),
        "daily_q90_p90": float(daily_q90.quantile(0.9)),
        "daily_violation_gt_035": int((daily_q90 > 0.035).sum()),
        "daily_violation_share": float((daily_q90 > 0.035).mean()),
        "DA": float(np.mean(np.sign(actual) == np.sign(pred))),
        "pred_actual_q90_ratio": float(np.quantile(np.abs(pred), 0.9) / max(np.quantile(np.abs(actual), 0.9), 1e-8)),
    }
    if risk is not None:
        risk = risk[mask]
        tail = (np.abs(err) > TAIL_THRESHOLD).astype(int)
        if len(np.unique(tail)) > 1:
            row["tail_auc"] = float(roc_auc_score(tail, risk))
            row["tail_ap"] = float(average_precision_score(tail, risk))
        else:
            row["tail_auc"] = float("nan")
            row["tail_ap"] = float("nan")
    return row


def load_meta() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_training_frame(str(DATA), stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["__tn__"] = frame["volatility_20"].astype(float)
    frame["__ret5__"] = frame.groupby("code")["target_next_return"].shift(1).rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    train_df = frame.loc[frame["Date"] <= "2020-03-31"].copy()
    scaler = fit_feature_scaler(train_df.dropna(subset=features), features)
    scaled = apply_feature_scaler(frame, scaler)
    _, _, meta_all = build_sequence_dataset(
        scaled,
        features,
        "target_next_return",
        15,
        extra_meta_columns=("__tn__", "__ret5__"),
        sequence_normalization="none",
    )
    splits = split_sequence_dataset(np.zeros((len(meta_all), 1), dtype=np.float32), meta_all["target"].to_numpy(), meta_all, "2020-03-31", "2022-11-15")
    return splits["train"][2].reset_index(drop=True), splits["val"][2].reset_index(drop=True)


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
    scale, clip = choose_train_calibration(y_train_ref, pred_train, np.full_like(pred_train, 0.02))
    pred_train = pred_train * scale
    pred_val = pred_val * scale
    if clip is not None:
        pred_train = np.clip(pred_train, -clip * 0.02, clip * 0.02)
        pred_val = np.clip(pred_val, -clip * 0.02, clip * 0.02)
    return y_train_ref, y_val_ref, pred_train.astype(np.float32), pred_val.astype(np.float32), sig_train_mean, sig_val_mean


def make_features(meta: pd.DataFrame, pred: np.ndarray, sigma: np.ndarray) -> pd.DataFrame:
    work = meta[["Date", "code", "__tn__", "__ret5__"]].copy()
    work["pred"] = pred
    work["sigma"] = sigma
    work["abs_pred"] = np.abs(pred)
    work["conf"] = work["abs_pred"] / np.maximum(work["sigma"], 1e-8)
    grouped = work.groupby("Date", observed=True)
    work["daily_abs_pred_rank"] = grouped["abs_pred"].rank(pct=True)
    work["daily_sigma_rank"] = grouped["sigma"].rank(pct=True)
    work["daily_abs_pred_mean"] = grouped["abs_pred"].transform("mean")
    work["daily_abs_pred_q90"] = grouped["abs_pred"].transform(lambda values: values.quantile(0.9))
    work["daily_sigma_mean"] = grouped["sigma"].transform("mean")
    work["daily_conf_mean"] = grouped["conf"].transform("mean")
    return work[
        [
            "__tn__",
            "__ret5__",
            "pred",
            "sigma",
            "abs_pred",
            "conf",
            "daily_abs_pred_rank",
            "daily_sigma_rank",
            "daily_abs_pred_mean",
            "daily_abs_pred_q90",
            "daily_sigma_mean",
            "daily_conf_mean",
        ]
    ].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_risk_models(x_train: pd.DataFrame, y_tail_train: np.ndarray) -> dict[str, object]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3, random_state=43)
    logistic.fit(x_scaled, y_tail_train)
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.035,
        max_leaf_nodes=12,
        l2_regularization=0.08,
        random_state=43,
        class_weight="balanced",
    )
    hgb.fit(x_train, y_tail_train)
    return {"logistic": (logistic, scaler), "hgb": hgb}


def predict_risk(models: dict[str, object], x_train: pd.DataFrame, x_val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    logistic, scaler = models["logistic"]
    train_log = logistic.predict_proba(scaler.transform(x_train))[:, 1]
    val_log = logistic.predict_proba(scaler.transform(x_val))[:, 1]
    hgb = models["hgb"]
    train_hgb = hgb.predict_proba(x_train)[:, 1]
    val_hgb = hgb.predict_proba(x_val)[:, 1]
    return (0.5 * train_log + 0.5 * train_hgb).astype(np.float32), (0.5 * val_log + 0.5 * val_hgb).astype(np.float32)


def apply_gate(pred: np.ndarray, risk: np.ndarray, q: float, mode: str, shrink: float) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(risk[np.isfinite(risk)], q))
    high_risk = risk >= threshold
    out = pred.copy()
    if mode == "zero":
        out[high_risk] = 0.0
    elif mode == "shrink":
        out[high_risk] *= shrink
    elif mode == "soft":
        scale = np.clip(1.0 - shrink * risk, 0.0, 1.0)
        out = out * scale
    else:
        raise ValueError(f"Unknown gate mode: {mode}")
    return out.astype(np.float32), float(high_risk.mean())


def plot_hist(y: np.ndarray, base_pred: np.ndarray, gate_pred: np.ndarray, selected: dict[str, object], out: Path, gold: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, title, pred in [
        (axes[0], "Anchor baseline", base_pred),
        (axes[1], f"Tail-risk aware gate\n{selected['variant']}", gate_pred),
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
    fig.suptitle("Tail-Risk Aware Error Histogram", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(gold, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_risk_deciles(y: np.ndarray, pred: np.ndarray, risk: np.ndarray, out: Path, gold: Path) -> None:
    err = np.abs(y - pred)
    df = pd.DataFrame({"risk": risk, "abs_e": err, "tail": err > TAIL_THRESHOLD})
    df["decile"] = pd.qcut(df["risk"], 10, labels=False, duplicates="drop") + 1
    agg = df.groupby("decile", observed=True).agg(abs_e_q90=("abs_e", lambda values: values.quantile(0.9)), tail_rate=("tail", "mean"), risk_mean=("risk", "mean"))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(agg.index.astype(str), agg["tail_rate"] * 100, color="#d93025", alpha=0.72, label="Tail rate |E|>3.5%")
    ax1.set_ylabel("Tail rate (%)", color="#d93025")
    ax2 = ax1.twinx()
    ax2.plot(agg.index.astype(str), agg["abs_e_q90"] * 100, color="#1a73e8", marker="o", label="Q90(|E|)")
    ax2.set_ylabel("Q90(|E|) %", color="#1a73e8")
    ax1.set_xlabel("Predicted tail-risk decile")
    ax1.set_title("Tail-Risk Classifier Monotonicity", fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(gold, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_daily_q90(dates: pd.Series, y: np.ndarray, base_pred: np.ndarray, gate_pred: np.ndarray, out: Path, gold: Path) -> None:
    def daily_q90(pred: np.ndarray) -> pd.Series:
        return pd.DataFrame({"Date": pd.to_datetime(dates), "abs_e": np.abs(y - pred)}).groupby("Date")["abs_e"].quantile(0.9)

    b = daily_q90(base_pred) * 100
    g = daily_q90(gate_pred) * 100
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(b.index, b.values, color="#888888", alpha=0.55, linewidth=0.8, label=f"Anchor, viol>{(b > 3.5).sum()} days")
    ax.plot(g.index, g.values, color="#1a73e8", alpha=0.78, linewidth=0.8, label=f"Tail-risk gate, viol>{(g > 3.5).sum()} days")
    ax.plot(b.index, b.rolling(21, min_periods=5).mean(), color="#666666", linewidth=2)
    ax.plot(g.index, g.rolling(21, min_periods=5).mean(), color="#1a73e8", linewidth=2)
    ax.axhline(3.0, color="green", linestyle="--", label="Target 3.0%")
    ax.axhline(3.5, color="red", linestyle="--", label="Violation 3.5%")
    ax.set_title("Daily Q90(|E|): Anchor vs Tail-Risk Gate", fontweight="bold")
    ax.set_ylabel("Q90(|E|) %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(gold, dpi=130, bbox_inches="tight")
    plt.close(fig)


def selective_coverage_table(actual: np.ndarray, pred: np.ndarray, risk: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for coverage in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]:
        threshold = float(np.quantile(risk, coverage))
        keep = risk <= threshold
        kept_actual = actual[keep]
        kept_pred = pred[keep]
        err = kept_actual - kept_pred
        rows.append({
            "coverage": float(keep.mean()),
            "risk_threshold": threshold,
            "rel_score_subset": rel_score(kept_actual, kept_pred),
            "base_loss_subset": robust_loss(kept_actual),
            "abs_loss_subset": robust_loss(err),
            "q90_abs_e": float(np.quantile(np.abs(err), 0.9)),
            "q95_abs_e": float(np.quantile(np.abs(err), 0.95)),
            "share_abs_e_gt_035": float((np.abs(err) > TAIL_THRESHOLD).mean()),
            "share_abs_e_gt_050": float((np.abs(err) > 0.05).mean()),
            "DA": float(np.mean(np.sign(kept_actual) == np.sign(kept_pred))),
        })
    return pd.DataFrame(rows)


def plot_selective_coverage(coverage: pd.DataFrame, out: Path, gold: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = coverage["coverage"] * 100
    ax1.plot(x, coverage["q90_abs_e"] * 100, color="#1a73e8", marker="o", label="Q90(|E|)")
    ax1.plot(x, coverage["q95_abs_e"] * 100, color="#174ea6", marker="o", label="Q95(|E|)")
    ax1.axhline(3.5, color="red", linestyle="--", linewidth=1, label="3.5% tail line")
    ax1.set_xlabel("Prediction coverage kept: low-risk subset (%)")
    ax1.set_ylabel("Absolute error quantile (%)")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, coverage["share_abs_e_gt_050"] * 100, color="#d93025", marker="s", label="Share |E|>5%")
    ax2.set_ylabel("Share |E|>5% (%)", color="#d93025")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.set_title("Selective Prediction: Abstain on High Tail-Risk", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(gold, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    meta_train, meta_val = load_meta()
    y_train, y_val, pred_train, pred_val, sig_train, sig_val = load_anchor()
    if len(meta_train) != len(y_train) or len(meta_val) != len(y_val):
        raise RuntimeError(f"Meta/prediction length mismatch: train {len(meta_train)} vs {len(y_train)}, val {len(meta_val)} vs {len(y_val)}")

    x_train = make_features(meta_train, pred_train, sig_train)
    x_val = make_features(meta_val, pred_val, sig_val)
    train_err = y_train - pred_train
    y_tail_train = (np.abs(train_err) > TAIL_THRESHOLD).astype(int)
    models = fit_risk_models(x_train, y_tail_train)
    risk_train, risk_val = predict_risk(models, x_train, x_val)

    rows: list[dict[str, object]] = []
    baseline = metric(y_val, pred_val, meta_val["Date"], risk_val)
    baseline.update({"variant": "anchor_baseline", "mode": "baseline", "risk_q": np.nan, "shrink": np.nan, "gated_share": 0.0})
    rows.append(baseline)
    for mode in ["zero", "shrink", "soft"]:
        for risk_q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95]:
            shrink_grid = [0.0] if mode == "zero" else ([0.25, 0.50, 0.75] if mode == "shrink" else [0.35, 0.50, 0.70, 0.90])
            for shrink in shrink_grid:
                gated_pred, gated_share = apply_gate(pred_val, risk_val, risk_q, mode, shrink)
                row = metric(y_val, gated_pred, meta_val["Date"], risk_val)
                row.update({
                    "variant": f"{mode}_riskq{risk_q}_s{shrink}",
                    "mode": mode,
                    "risk_q": risk_q,
                    "shrink": shrink,
                    "gated_share": gated_share,
                })
                rows.append(row)

    result = pd.DataFrame(rows)
    train_tail = (np.abs(train_err) > TAIL_THRESHOLD).astype(int)
    train_auc = roc_auc_score(train_tail, risk_train)
    train_ap = average_precision_score(train_tail, risk_train)

    candidates = result[result["variant"] != "anchor_baseline"].copy()
    improved = candidates[
        (candidates["rel_score"] >= baseline["rel_score"])
        & (candidates["q90_abs_e"] <= baseline["q90_abs_e"])
        & (candidates["q95_abs_e"] <= baseline["q95_abs_e"])
        & (candidates["share_abs_e_gt_050"] <= baseline["share_abs_e_gt_050"])
    ].copy()
    if not improved.empty:
        improved["objective"] = (
            2.0 * (improved["rel_score"] - baseline["rel_score"])
            + 8.0 * (baseline["q90_abs_e"] - improved["q90_abs_e"])
            + 4.0 * (baseline["q95_abs_e"] - improved["q95_abs_e"])
            + 1.0 * (baseline["share_abs_e_gt_050"] - improved["share_abs_e_gt_050"])
            - 0.10 * np.maximum(improved["daily_violation_share"] - baseline["daily_violation_share"], 0.0)
        )
        improved = improved.sort_values("objective", ascending=False)
        selected = improved.iloc[0].to_dict()
    else:
        candidates["objective"] = (
            2.0 * candidates["rel_score"]
            -8.0 * candidates["q90_abs_e"]
            -4.0 * candidates["q95_abs_e"]
            -1.0 * candidates["share_abs_e_gt_050"]
            -0.10 * candidates["daily_violation_share"]
        )
        selected = candidates.sort_values("objective", ascending=False).iloc[0].to_dict()

    selected_pred, _ = apply_gate(pred_val, risk_val, float(selected["risk_q"]), str(selected["mode"]), float(selected["shrink"]))
    comparison = pd.DataFrame([baseline, selected])
    selective = selective_coverage_table(y_val, pred_val, risk_val)
    risk_readout = {
        "train_tail_rate": float(train_tail.mean()),
        "train_tail_auc": float(train_auc),
        "train_tail_ap": float(train_ap),
        "val_tail_auc_anchor_errors": float(baseline["tail_auc"]),
        "val_tail_ap_anchor_errors": float(baseline["tail_ap"]),
    }

    result.to_csv(OUTPUT / "tail_risk_gate_grid.csv", index=False)
    result.to_csv(GOLD / "tail_risk_gate_grid.csv", index=False)
    comparison.to_csv(OUTPUT / "selected_comparison.csv", index=False)
    comparison.to_csv(GOLD / "selected_comparison.csv", index=False)
    selective.to_csv(OUTPUT / "selective_coverage.csv", index=False)
    selective.to_csv(GOLD / "selective_coverage.csv", index=False)
    if not improved.empty:
        improved.head(50).to_csv(OUTPUT / "dominant_improvement_candidates.csv", index=False)
        improved.head(50).to_csv(GOLD / "dominant_improvement_candidates.csv", index=False)
    pd.DataFrame({"risk": risk_val, "actual": y_val, "anchor_pred": pred_val, "selected_pred": selected_pred, "date": meta_val["Date"], "code": meta_val["code"]}).to_csv(OUTPUT / "validation_risk_scores.csv", index=False)

    plot_hist(y_val, pred_val, selected_pred, selected, OUTPUT / "tail_risk_histogram.png", GOLD / "tail_risk_histogram.png")
    plot_risk_deciles(y_val, pred_val, risk_val, OUTPUT / "tail_risk_deciles.png", GOLD / "tail_risk_deciles.png")
    plot_daily_q90(meta_val["Date"], y_val, pred_val, selected_pred, OUTPUT / "tail_risk_daily_q90.png", GOLD / "tail_risk_daily_q90.png")
    plot_selective_coverage(selective, OUTPUT / "selective_coverage.png", GOLD / "selective_coverage.png")

    text = "\n".join([
        "# Tail-Risk Aware Gate",
        "",
        "Protocol: train-period tail classifier; validation readout only. Holdout/test not used.",
        "",
        "## Risk Classifier Readout",
        "",
        json.dumps(risk_readout, indent=2),
        "",
        "## Selected vs Baseline",
        "",
        comparison[["variant", "rel_score", "q90_abs_e", "q95_abs_e", "daily_violation_gt_035", "daily_violation_share", "share_abs_e_gt_050", "DA", "gated_share", "tail_auc", "tail_ap"]].round(6).to_markdown(index=False),
        "",
        "## Dominant Improvement Candidates",
        "",
        improved[["variant", "rel_score", "q90_abs_e", "q95_abs_e", "daily_violation_gt_035", "daily_violation_share", "share_abs_e_gt_050", "DA", "gated_share"]].head(20).round(6).to_markdown(index=False) if not improved.empty else "_No gate improves rel_score, Q90, Q95, and >5% tail share together._",
        "",
        "## Selective Prediction Coverage",
        "",
        selective[["coverage", "rel_score_subset", "base_loss_subset", "abs_loss_subset", "q90_abs_e", "q95_abs_e", "share_abs_e_gt_035", "share_abs_e_gt_050", "DA"]].round(6).to_markdown(index=False),
        "",
        "## Top Gate Candidates",
        "",
        result[result["variant"] != "anchor_baseline"].sort_values(["rel_score", "q90_abs_e"], ascending=[False, True])[["variant", "rel_score", "q90_abs_e", "q95_abs_e", "daily_violation_gt_035", "daily_violation_share", "share_abs_e_gt_050", "DA", "gated_share"]].head(20).round(6).to_markdown(index=False),
        "",
        json.dumps({"output_dir": str(OUTPUT), "gold_dir": str(GOLD)}, indent=2),
    ])
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
