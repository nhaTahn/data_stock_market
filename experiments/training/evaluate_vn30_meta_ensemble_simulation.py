"""VN30 Point-in-Time meta-ensemble simulation over the frozen 5-seed hetero anchor.

This compares broad-market calibration versus VN30-specific calibration on the
VN30 validation subset, supporting Point-in-Time constituents via vn30_historical.csv.
Holdout/test is not used.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training import evaluate_regime_calibration as regime  # noqa: E402
from experiments.training.evaluate_fixed_train_relscore_calibration import choose_train_calibration  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler  # noqa: E402
from src.utils.universe import load_historical_universe_mask  # noqa: E402

PRED_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_combined_full5_20260521"
DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
VN30_PATH = ROOT / "market_lists/vn30.txt"
VN30_HIST_PATH = ROOT / "market_lists/vn30_historical.csv"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/vn30_meta_ensemble_simulation_20260529"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/vn30_meta_ensemble_simulation_20260529"
SEEDS = (43, 52, 62, 71, 82)


def robust_loss(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def rel_score(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    base = robust_loss(actual[mask])
    if base <= 0:
        return float("nan")
    return float(1.0 - robust_loss((actual - pred)[mask]) / base)


def metrics(actual: np.ndarray, pred: np.ndarray, dates: pd.Series) -> dict[str, float]:
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    dates = dates[mask]
    err = actual - pred
    daily_q90 = (
        pd.DataFrame({"Date": pd.to_datetime(dates), "abs_e": np.abs(err)})
        .groupby("Date")["abs_e"]
        .quantile(0.9)
    )
    return {
        "n_obs": int(len(actual)),
        "n_days": int(daily_q90.shape[0]),
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
    y_train_ref: np.ndarray | None = None
    y_val_ref: np.ndarray | None = None

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
    scale, _ = choose_train_calibration(y_train_ref, base_train, np.full_like(base_train, 0.02))
    return y_train_ref, y_val_ref, train_matrix * scale, val_matrix * scale, train_sigma, val_sigma


def load_meta() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_training_frame(str(DATA), stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    scaler = fit_feature_scaler(frame.loc[frame["Date"] <= "2020-03-31"].dropna(subset=features), features)
    scaled = apply_feature_scaler(frame, scaler)
    _, _, meta = build_sequence_dataset(scaled, features, "target_next_return", 15, sequence_normalization="none")
    dummy = np.zeros((len(meta), 1), dtype=np.float32)
    splits = split_sequence_dataset(dummy, meta["target"].to_numpy(), meta, "2020-03-31", "2022-11-15")
    return splits["train"][2].reset_index(drop=True), splits["val"][2].reset_index(drop=True)


def make_meta_features(pred_matrix: np.ndarray, sigma_matrix: np.ndarray, rv: np.ndarray, q905: np.ndarray) -> np.ndarray:
    mean_pred = pred_matrix.mean(axis=1)
    return np.column_stack(
        [
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
        ]
    ).astype(np.float32)


def train_selected_blend(
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    meta_train: np.ndarray,
    anchor_val: np.ndarray,
    meta_val: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    best_alpha = 0.0
    best_score = rel_score(y_train, anchor_train)
    for alpha in np.linspace(-0.5, 1.5, 81):
        candidate = (1.0 - alpha) * anchor_train + alpha * meta_train
        score = rel_score(y_train, candidate)
        if score > best_score:
            best_alpha = float(alpha)
            best_score = score
    return (
        best_alpha,
        (1.0 - best_alpha) * anchor_train + best_alpha * meta_train,
        (1.0 - best_alpha) * anchor_val + best_alpha * meta_val,
    )


def plot_hist(actual: np.ndarray, series: dict[str, np.ndarray], out: Path, gold: Path) -> None:
    fig, axes = plt.subplots(1, len(series), figsize=(5.2 * len(series), 5), sharey=True)
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
    fig.suptitle("VN30 Meta-Ensemble Calibration Error Histogram", fontweight="bold")
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
        ax.bar(labels, comparison[col], color="#1a73e8")
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(gold, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)

    y_train_full, y_val_full, pred_train_full, pred_val_full, sigma_train_full, sigma_val_full = load_anchor_seed_predictions()
    meta_train, meta_val = load_meta()
    dates_train_full = pd.to_datetime(meta_train["Date"])
    dates_val_full = pd.to_datetime(meta_val["Date"])
    codes_train = meta_train["code"].astype(str).str.upper()
    codes_val = meta_val["code"].astype(str).str.upper()

    # Load point-in-time masks
    train_mask = load_historical_universe_mask(meta_train, VN30_HIST_PATH)
    val_mask = load_historical_universe_mask(meta_val, VN30_HIST_PATH)

    market = regime.build_market_features(dates_train_full, dates_val_full, y_train_full, y_val_full)
    rv_train_full = market["train"]["vol10"]
    rv_val_full = market["val"]["vol10"]
    q_train_full = market["train"]["q905"]
    q_val_full = market["val"]["q905"]

    base_train_full = pred_train_full.mean(axis=1)
    base_val_full = pred_val_full.mean(axis=1)
    rv_edges_broad, q_edges_broad, grid_broad = regime.fit_2d_scales(y_train_full, base_train_full, rv_train_full, q_train_full)
    regime_train_broad = regime.apply_2d_scales(base_train_full, rv_train_full, q_train_full, rv_edges_broad, q_edges_broad, grid_broad)
    regime_val_broad = regime.apply_2d_scales(base_val_full, rv_val_full, q_val_full, rv_edges_broad, q_edges_broad, grid_broad)

    y_train_vn30 = y_train_full[train_mask]
    y_val_vn30 = y_val_full[val_mask]
    base_train_vn30 = base_train_full[train_mask]
    base_val_vn30 = base_val_full[val_mask]
    dates_val_vn30 = dates_val_full[val_mask].reset_index(drop=True)

    rv_edges_vn30, q_edges_vn30, grid_vn30 = regime.fit_2d_scales(
        y_train_vn30,
        base_train_vn30,
        rv_train_full[train_mask],
        q_train_full[train_mask],
    )
    regime_train_vn30 = regime.apply_2d_scales(
        base_train_vn30,
        rv_train_full[train_mask],
        q_train_full[train_mask],
        rv_edges_vn30,
        q_edges_vn30,
        grid_vn30,
    )
    regime_val_vn30 = regime.apply_2d_scales(
        base_val_vn30,
        rv_val_full[val_mask],
        q_val_full[val_mask],
        rv_edges_vn30,
        q_edges_vn30,
        grid_vn30,
    )

    x_train_full = make_meta_features(pred_train_full, sigma_train_full, rv_train_full, q_train_full)
    x_val_full = make_meta_features(pred_val_full, sigma_val_full, rv_val_full, q_val_full)
    x_train_vn30 = x_train_full[train_mask]
    x_val_vn30 = x_val_full[val_mask]

    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "enet": make_pipeline(StandardScaler(), ElasticNet(alpha=1e-4, l1_ratio=0.2, max_iter=5000, random_state=43)),
        "hgb_abs": HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, learning_rate=0.03, max_leaf_nodes=8, l2_regularization=0.2, random_state=43),
        "et_tail": ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_leaf=300, random_state=43, n_jobs=-1),
    }

    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {
        "anchor": base_val_vn30,
        "2d_regime_broad": regime_val_broad[val_mask],
        "2d_regime_vn30": regime_val_vn30,
    }

    rows.append({"variant": "anchor", "model": "anchor", "alpha": 0.0, "train_rel_score": rel_score(y_train_vn30, base_train_vn30), **metrics(y_val_vn30, base_val_vn30, dates_val_vn30)})
    rows.append({"variant": "2d_regime_broad", "model": "2d_regime", "alpha": 0.0, "train_rel_score": rel_score(y_train_vn30, regime_train_broad[train_mask]), **metrics(y_val_vn30, regime_val_broad[val_mask], dates_val_vn30)})
    rows.append({"variant": "2d_regime_vn30", "model": "2d_regime", "alpha": 0.0, "train_rel_score": rel_score(y_train_vn30, regime_train_vn30), **metrics(y_val_vn30, regime_val_vn30, dates_val_vn30)})

    for name, model in models.items():
        model.fit(x_train_full, y_train_full)
        meta_train_full = np.asarray(model.predict(x_train_full), dtype=np.float32)
        meta_val_full = np.asarray(model.predict(x_val_full), dtype=np.float32)
        alpha_broad, blend_train_broad, blend_val_broad = train_selected_blend(
            y_train_full,
            regime_train_broad,
            meta_train_full,
            regime_val_broad,
            meta_val_full,
        )
        variant_broad = f"{name}_blend_broad"
        predictions[variant_broad] = blend_val_broad[val_mask]
        rows.append({
            "variant": variant_broad,
            "model": name,
            "alpha": alpha_broad,
            "train_rel_score": rel_score(y_train_vn30, blend_train_broad[train_mask]),
            **metrics(y_val_vn30, blend_val_broad[val_mask], dates_val_vn30),
        })

        model.fit(x_train_vn30, y_train_vn30)
        meta_train_vn30 = np.asarray(model.predict(x_train_vn30), dtype=np.float32)
        meta_val_vn30 = np.asarray(model.predict(x_val_vn30), dtype=np.float32)
        alpha_vn30, blend_train_vn30, blend_val_vn30 = train_selected_blend(
            y_train_vn30,
            regime_train_vn30,
            meta_train_vn30,
            regime_val_vn30,
            meta_val_vn30,
        )
        variant_vn30 = f"{name}_blend_vn30"
        predictions[variant_vn30] = blend_val_vn30
        rows.append({
            "variant": variant_vn30,
            "model": name,
            "alpha": alpha_vn30,
            "train_rel_score": rel_score(y_train_vn30, blend_train_vn30),
            **metrics(y_val_vn30, blend_val_vn30, dates_val_vn30),
        })

    comparison = pd.DataFrame(rows).sort_values("rel_score", ascending=False)
    comparison.to_csv(OUTPUT / "comparison.csv", index=False)
    comparison.to_csv(GOLD / "comparison.csv", index=False)

    best_broad = comparison[comparison["variant"].str.endswith("_broad")].iloc[0]["variant"]
    best_vn30 = comparison[comparison["variant"].str.endswith("_vn30")].iloc[0]["variant"]
    plot_hist(
        y_val_vn30,
        {
            "Anchor": predictions["anchor"],
            "2D Broad": predictions["2d_regime_broad"],
            "2D VN30": predictions["2d_regime_vn30"],
            f"Best Broad\n{best_broad}": predictions[str(best_broad)],
            f"Best VN30\n{best_vn30}": predictions[str(best_vn30)],
        },
        OUTPUT / "vn30_simulation_histogram.png",
        GOLD / "vn30_simulation_histogram.png",
    )
    plot_metric_bars(
        comparison[comparison["variant"].isin(["anchor", "2d_regime_broad", "2d_regime_vn30", str(best_broad), str(best_vn30)])].drop_duplicates("variant"),
        OUTPUT / "vn30_simulation_metric_bars.png",
        GOLD / "vn30_simulation_metric_bars.png",
    )

    summary = "\n".join([
        "# VN30 Meta-Ensemble Simulation Report",
        "",
        "Date: 2026-05-29",
        "Scope: VN30 constituents during VN validation period only. Holdout/test not used.",
        "",
        f"Constituents loaded from: `{VN30_HIST_PATH.name}`.",
        f"VN30 train observations: {int(train_mask.sum())}; VN30 validation observations: {int(val_mask.sum())}.",
        "",
        "## Protocol",
        "",
        "- `_broad`: calibration/meta learner trained on the full VN training set, then evaluated only on VN30 validation rows.",
        "- `_vn30`: calibration/meta learner trained only on VN30 training rows, then evaluated on VN30 validation rows.",
        "- Market-regime features remain lagged full-market features; holdout/test is not used.",
        "",
        "## Performance Table",
        "",
        comparison[["variant", "model", "alpha", "n_obs", "n_days", "train_rel_score", "rel_score", "q90_abs_e", "q95_abs_e", "share_abs_e_gt_050", "daily_violation_gt_035", "DA"]].round(6).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- Best broad-trained variant: `{best_broad}`.",
        f"- Best VN30-trained variant: `{best_vn30}`.",
        "- If broad-trained variants win, the calibration is benefiting from larger cross-sectional training support.",
        "- If VN30-trained variants win, the VN30 universe has enough distinct structure to justify a dedicated calibration layer.",
        "",
        json.dumps({"output_dir": str(OUTPUT), "gold_dir": str(GOLD)}, indent=2),
    ])
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    (GOLD / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
