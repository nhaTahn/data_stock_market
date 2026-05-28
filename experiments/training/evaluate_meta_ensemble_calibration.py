"""Meta-ensemble calibration over the frozen 5-seed hetero anchor.

This trains lightweight train-only meta learners using seed-level predictions,
uncertainty disagreement, and lagged market-regime features. Validation is used
only for readout. Holdout/test is not used.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.evaluate_fixed_train_relscore_calibration import choose_train_calibration  # noqa: E402
from experiments.training import evaluate_regime_calibration as regime  # noqa: E402

PRED_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_combined_full5_20260521"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/meta_ensemble_calibration_20260528"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/meta_ensemble_calibration_20260528"
SEEDS = (43, 52, 62, 71, 82)


def robust_loss(values: np.ndarray) -> float:
    return float(np.quantile(np.abs(values), 0.5) + 0.5 * np.quantile(np.abs(values), 0.9))


def rel_score(actual: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - robust_loss(actual - pred) / robust_loss(actual)


def metrics(actual: np.ndarray, pred: np.ndarray, dates: pd.Series) -> dict[str, float]:
    err = actual - pred
    daily_q90 = pd.DataFrame({"Date": pd.to_datetime(dates), "abs_e": np.abs(err)}).groupby("Date")["abs_e"].quantile(0.9)
    return {
        "rel_score": rel_score(actual, pred),
        "q90_abs_e": float(np.quantile(np.abs(err), 0.9)),
        "q95_abs_e": float(np.quantile(np.abs(err), 0.95)),
        "share_abs_e_gt_035": float((np.abs(err) > 0.035).mean()),
        "share_abs_e_gt_050": float((np.abs(err) > 0.05).mean()),
        "daily_violation_gt_035": int((daily_q90 > 0.035).sum()),
        "daily_violation_share": float((daily_q90 > 0.035).mean()),
        "DA": float((np.sign(actual) == np.sign(pred)).mean()),
    }


def load_anchor_seed_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_preds: list[np.ndarray] = []
    val_preds: list[np.ndarray] = []
    train_sigmas: list[np.ndarray] = []
    val_sigmas: list[np.ndarray] = []
    y_train_ref = y_val_ref = None
    for seed in SEEDS:
        data = np.load(PRED_DIR / f"predictions_seed_{seed}.npz", allow_pickle=True)
        y_train = data["y_train"].astype(np.float32)
        mu_train = data["mu_train"].astype(np.float32)
        sigma_train = data["sigma_train"].astype(np.float32)
        mu_val = data["mu_val"].astype(np.float32)
        sigma_val = data["sigma_val"].astype(np.float32)
        scale, clip = choose_train_calibration(y_train, mu_train, sigma_train)
        mu_train = mu_train * scale
        mu_val = mu_val * scale
        if clip is not None:
            mu_train = np.clip(mu_train, -clip * sigma_train, clip * sigma_train)
            mu_val = np.clip(mu_val, -clip * sigma_val, clip * sigma_val)
        train_preds.append(mu_train)
        val_preds.append(mu_val)
        train_sigmas.append(sigma_train)
        val_sigmas.append(sigma_val)
        y_train_ref = y_train
        y_val_ref = data["y_val"].astype(np.float32)
    assert y_train_ref is not None and y_val_ref is not None
    train_matrix = np.stack(train_preds, axis=1).astype(np.float32)
    val_matrix = np.stack(val_preds, axis=1).astype(np.float32)
    train_sigma = np.stack(train_sigmas, axis=1).astype(np.float32)
    val_sigma = np.stack(val_sigmas, axis=1).astype(np.float32)
    base_train = train_matrix.mean(axis=1)
    base_val = val_matrix.mean(axis=1)
    scale, _ = choose_train_calibration(y_train_ref, base_train, np.full_like(base_train, 0.02))
    return y_train_ref, y_val_ref, train_matrix * scale, val_matrix * scale, train_sigma, val_sigma


def make_meta_features(pred_matrix: np.ndarray, sigma_matrix: np.ndarray, rv: np.ndarray, q905: np.ndarray) -> np.ndarray:
    mean_pred = pred_matrix.mean(axis=1)
    return np.column_stack([
        pred_matrix,
        mean_pred,
        pred_matrix.std(axis=1),
        np.median(pred_matrix, axis=1),
        pred_matrix.max(axis=1),
        pred_matrix.min(axis=1),
        sigma_matrix.mean(axis=1),
        sigma_matrix.std(axis=1),
        rv,
        q905,
        rv * q905,
        mean_pred * rv,
        mean_pred * q905,
    ]).astype(np.float32)


def train_selected_blend(y_train: np.ndarray, anchor_train: np.ndarray, meta_train: np.ndarray, y_val: np.ndarray, anchor_val: np.ndarray, meta_val: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    best_alpha = 0.0
    best_score = rel_score(y_train, anchor_train)
    for alpha in np.linspace(-0.5, 1.5, 81):
        candidate = (1.0 - alpha) * anchor_train + alpha * meta_train
        score = rel_score(y_train, candidate)
        if score > best_score:
            best_alpha = float(alpha)
            best_score = score
    return best_alpha, (1.0 - best_alpha) * anchor_train + best_alpha * meta_train, (1.0 - best_alpha) * anchor_val + best_alpha * meta_val


def plot_hist(actual: np.ndarray, series: dict[str, np.ndarray], out: Path, gold: Path) -> None:
    fig, axes = plt.subplots(1, len(series), figsize=(5.5 * len(series), 5), sharey=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (name, pred) in zip(axes, series.items()):
        err = actual - pred
        ax.hist(err * 100, bins=120, density=True, color="#4a90e2", alpha=0.65)
        for quantile, color, label in [(0.75, "#9955cc", "Q75"), (0.9, "red", "Q90"), (0.95, "darkred", "Q95")]:
            value = np.quantile(np.abs(err), quantile) * 100
            ax.axvline(value, color=color, linestyle=":", label=f"±{label}={value:.2f}%")
            ax.axvline(-value, color=color, linestyle=":")
        ax.axvline(3.5, color="black", linestyle="--", linewidth=1)
        ax.axvline(-3.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Error (%)")
        ax.set_xlim(-12, 12)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density")
    fig.suptitle("Meta-Ensemble Calibration Error Histogram", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(gold, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(comparison: pd.DataFrame, out: Path, gold: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = comparison["variant"].to_list()
    for ax, col, title in [
        (axes[0], "rel_score", "rel_score ↑"),
        (axes[1], "q90_abs_e", "Q90(|E|) ↓"),
        (axes[2], "share_abs_e_gt_050", "Share |E|>5% ↓"),
    ]:
        ax.bar(labels, comparison[col], color=["#999999", "#1a73e8", "#34a853", "#f4511e"][: len(labels)])
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(gold, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    y_train, y_val, pred_train, pred_val, sigma_train, sigma_val = load_anchor_seed_predictions()
    dates_train, dates_val = regime.load_dates()
    market = regime.build_market_features(dates_train, dates_val, y_train, y_val)
    rv_train = market["train"]["vol10"]
    rv_val = market["val"]["vol10"]
    q_train = market["train"]["q905"]
    q_val = market["val"]["q905"]
    base_train = pred_train.mean(axis=1)
    base_val = pred_val.mean(axis=1)
    rv_edges, q_edges, grid = regime.fit_2d_scales(y_train, base_train, rv_train, q_train)
    regime_train = regime.apply_2d_scales(base_train, rv_train, q_train, rv_edges, q_edges, grid)
    regime_val = regime.apply_2d_scales(base_val, rv_val, q_val, rv_edges, q_edges, grid)

    x_train = make_meta_features(pred_train, sigma_train, rv_train, q_train)
    x_val = make_meta_features(pred_val, sigma_val, rv_val, q_val)
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "ridge_poly2": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1000.0)),
        "enet": make_pipeline(StandardScaler(), ElasticNet(alpha=1e-4, l1_ratio=0.2, max_iter=5000, random_state=43)),
        "hgb_abs": HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, learning_rate=0.03, max_leaf_nodes=8, l2_regularization=0.2, random_state=43),
        "et_tail": ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_leaf=300, random_state=43, n_jobs=-1),
    }

    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {
        "anchor": base_val,
        "2d_regime": regime_val,
    }
    rows.append({"variant": "anchor", "model": "anchor", "alpha": 0.0, **metrics(y_val, base_val, dates_val)})
    rows.append({"variant": "2d_regime", "model": "2d_regime", "alpha": 0.0, **metrics(y_val, regime_val, dates_val)})

    for name, model in models.items():
        model.fit(x_train, y_train)
        meta_train = np.asarray(model.predict(x_train), dtype=np.float32)
        meta_val = np.asarray(model.predict(x_val), dtype=np.float32)
        alpha, blended_train, blended_val = train_selected_blend(y_train, regime_train, meta_train, y_val, regime_val, meta_val)
        variant = f"{name}_blend"
        predictions[variant] = blended_val
        row = {"variant": variant, "model": name, "alpha": alpha, "train_rel_score": rel_score(y_train, blended_train)}
        row.update(metrics(y_val, blended_val, dates_val))
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values("rel_score", ascending=False)
    comparison.to_csv(OUTPUT / "comparison.csv", index=False)
    comparison.to_csv(GOLD / "comparison.csv", index=False)
    best_rs = comparison.iloc[0]["variant"]
    best_tail = comparison.sort_values(["q90_abs_e", "share_abs_e_gt_050", "rel_score"], ascending=[True, True, False]).iloc[0]["variant"]

    plot_hist(
        y_val,
        {
            "Anchor": predictions["anchor"],
            "2D-Regime": predictions["2d_regime"],
            f"Best RS\n{best_rs}": predictions[str(best_rs)],
            f"Best Tail\n{best_tail}": predictions[str(best_tail)],
        },
        OUTPUT / "meta_ensemble_histogram.png",
        GOLD / "meta_ensemble_histogram.png",
    )
    plot_metric_bars(
        comparison[comparison["variant"].isin(["anchor", "2d_regime", str(best_rs), str(best_tail)])].drop_duplicates("variant"),
        OUTPUT / "meta_ensemble_metric_bars.png",
        GOLD / "meta_ensemble_metric_bars.png",
    )
    text = "\n".join([
        "# Meta-Ensemble Calibration",
        "",
        "Protocol: train-only meta learners over frozen 5-seed predictions + lagged market regime. Validation readout only. Holdout/test not used.",
        "",
        "## Comparison",
        "",
        comparison[["variant", "model", "alpha", "train_rel_score", "rel_score", "q90_abs_e", "q95_abs_e", "share_abs_e_gt_050", "daily_violation_gt_035", "DA"]].round(6).to_markdown(index=False),
        "",
        json.dumps({"output_dir": str(OUTPUT), "gold_dir": str(GOLD)}, indent=2),
    ])
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
