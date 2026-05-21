from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
)
DEFAULT_OUTPUT_DIR = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "tail_calibration_probe_20260519"

PREDICTION_SPECS: dict[str, tuple[str, str]] = {
    "base": ("tail_aware_lstm_multiseed_20260519", "plain_global_rel"),
    "weighted": ("tail_aware_lstm_multiseed_20260519", "plain_global_weighted_mild"),
    "instance": ("tail_aware_lstm_multiseed_20260519", "plain_global_instance_rel"),
    "tail_loss": ("tail_loss_lstm_probe_20260519", "plain_global_weighted_mild_tail35_p05"),
    "tailstress": ("tailstress_lstm_probe_20260519", "plain_global_weighted_mild_tail35_p05_tailstress"),
    "stressaux_w20": ("stressaux_lstm_probe_20260519", "plain_global_weighted_mild_tail35_stressaux_w20"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate train-fitted calibration layers for tail-aware LSTM candidates.",
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--scale-step", type=float, default=0.05)
    parser.add_argument("--weight-step", type=float, default=0.10)
    parser.add_argument("--tail-penalty", type=float, default=0.50)
    return parser.parse_args(argv)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def daily_q90(frame: pd.DataFrame, prediction: np.ndarray) -> pd.Series:
    actual = frame["actual"].to_numpy(dtype=float)
    return (
        pd.DataFrame({"Date": pd.to_datetime(frame["Date"]), "abs_error": np.abs(actual - prediction)})
        .groupby("Date", sort=True)["abs_error"]
        .quantile(0.90)
    )


def tail_objective(frame: pd.DataFrame, prediction: np.ndarray, tail_penalty: float) -> float:
    error_loss = robust_loss(frame["actual"].to_numpy(dtype=float) - prediction)
    daily = daily_q90(frame, prediction)
    daily_tail = float(daily.quantile(0.90)) if not daily.empty else 0.0
    return error_loss + tail_penalty * daily_tail


def summarize_prediction(frame: pd.DataFrame, prediction: np.ndarray, candidate: str, seed: int, split: str) -> dict[str, object]:
    actual = frame["actual"].to_numpy(dtype=float)
    error = actual - prediction
    abs_error = np.abs(error)
    daily = daily_q90(frame, prediction)
    actual_abs_q90 = float(np.quantile(np.abs(actual), 0.90)) if len(actual) else float("nan")
    prediction_abs_q90 = float(np.quantile(np.abs(prediction), 0.90)) if len(prediction) else float("nan")
    return {
        "seed": seed,
        "candidate": candidate,
        "split": split,
        "n_obs": int(len(frame)),
        "n_days": int(daily.shape[0]),
        "rel_score": rel_score(actual, prediction),
        "median_abs_error": float(np.quantile(abs_error, 0.50)) if len(abs_error) else float("nan"),
        "q90_abs_error": float(np.quantile(abs_error, 0.90)) if len(abs_error) else float("nan"),
        "daily_q90_abs_error_median": float(daily.median()) if not daily.empty else float("nan"),
        "daily_q90_abs_error_q90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "daily_q90_abs_error_max": float(daily.max()) if not daily.empty else float("nan"),
        "spike_days_ge_5pct": int(daily.ge(0.05).sum()) if not daily.empty else 0,
        "spike_days_ge_7pct": int(daily.ge(0.07).sum()) if not daily.empty else 0,
        "spike_days_ge_8pct": int(daily.ge(0.08).sum()) if not daily.empty else 0,
        "prediction_actual_abs_q90_ratio": (
            prediction_abs_q90 / actual_abs_q90 if np.isfinite(actual_abs_q90) and actual_abs_q90 > 0.0 else float("nan")
        ),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(prediction))) if len(actual) else float("nan"),
    }


def prediction_path(report_root: Path, seed: int, short_name: str) -> Path:
    family, variant = PREDICTION_SPECS[short_name]
    return report_root / family / f"seed_{seed}" / f"predictions_{variant}.csv"


def load_seed_predictions(report_root: Path, seed: int) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for short_name in PREDICTION_SPECS:
        path = prediction_path(report_root, seed, short_name)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["Date"])
        keep = frame.loc[:, ["code", "Date", "split", "actual", "prediction"]].copy()
        keep = keep.rename(columns={"prediction": f"pred_{short_name}"})
        if merged is None:
            merged = keep
        else:
            merged = merged.merge(keep, on=["code", "Date", "split", "actual"], how="inner")
    if merged is None:
        raise ValueError(f"No predictions found for seed {seed}.")
    return merged


def fit_best_scale(actual: np.ndarray, prediction: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    best_scale = 1.0
    best_score = -np.inf
    for scale in grid:
        score = rel_score(actual, prediction * float(scale))
        if score > best_score:
            best_score = float(score)
            best_scale = float(scale)
    return best_scale, best_score


def fit_sign_scales(train: pd.DataFrame, source: str, grid: np.ndarray) -> dict[str, float]:
    actual = train["actual"].to_numpy(dtype=float)
    prediction = train[f"pred_{source}"].to_numpy(dtype=float)
    positive = prediction >= 0.0
    negative = ~positive
    pos_scale = fit_best_scale(actual[positive], prediction[positive], grid)[0] if np.any(positive) else 1.0
    neg_scale = fit_best_scale(actual[negative], prediction[negative], grid)[0] if np.any(negative) else 1.0
    return {"source": source, "positive_scale": pos_scale, "negative_scale": neg_scale}


def apply_sign_scales(frame: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    source = str(params["source"])
    prediction = frame[f"pred_{source}"].to_numpy(dtype=float)
    return np.where(prediction >= 0.0, prediction * float(params["positive_scale"]), prediction * float(params["negative_scale"]))


def fit_daily_scale(train: pd.DataFrame, source: str, grid: np.ndarray, tail_penalty: float) -> dict[str, float | str]:
    prediction = train[f"pred_{source}"].to_numpy(dtype=float)
    daily_signal = (
        pd.DataFrame({"Date": pd.to_datetime(train["Date"]), "abs_prediction": np.abs(prediction)})
        .groupby("Date", sort=True)["abs_prediction"]
        .quantile(0.90)
    )
    thresholds = np.quantile(daily_signal.to_numpy(dtype=float), [1.0 / 3.0, 2.0 / 3.0])
    signal_map = daily_signal.to_dict()
    buckets = pd.to_datetime(train["Date"]).map(signal_map).to_numpy(dtype=float)
    bucket_id = np.digitize(buckets, thresholds, right=False)
    scales: list[float] = []
    for idx in range(3):
        mask = bucket_id == idx
        if np.sum(mask) < 100:
            scales.append(1.0)
            continue
        scale = fit_best_scale(
            train.loc[mask, "actual"].to_numpy(dtype=float),
            train.loc[mask, f"pred_{source}"].to_numpy(dtype=float),
            grid,
        )[0]
        scales.append(float(scale))
    adjusted = apply_daily_scale(train, {"source": source, "threshold_1": thresholds[0], "threshold_2": thresholds[1], "scale_low": scales[0], "scale_mid": scales[1], "scale_high": scales[2]})
    base_score = tail_objective(train, prediction, tail_penalty)
    adjusted_score = tail_objective(train, adjusted, tail_penalty)
    if adjusted_score > base_score:
        scales = [1.0, 1.0, 1.0]
    return {
        "source": source,
        "threshold_1": float(thresholds[0]),
        "threshold_2": float(thresholds[1]),
        "scale_low": float(scales[0]),
        "scale_mid": float(scales[1]),
        "scale_high": float(scales[2]),
    }


def apply_daily_scale(frame: pd.DataFrame, params: dict[str, float | str]) -> np.ndarray:
    source = str(params["source"])
    prediction = frame[f"pred_{source}"].to_numpy(dtype=float)
    daily_signal = (
        pd.DataFrame({"Date": pd.to_datetime(frame["Date"]), "abs_prediction": np.abs(prediction)})
        .groupby("Date", sort=True)["abs_prediction"]
        .quantile(0.90)
    )
    signal_map = daily_signal.to_dict()
    values = pd.to_datetime(frame["Date"]).map(signal_map).to_numpy(dtype=float)
    thresholds = np.array([float(params["threshold_1"]), float(params["threshold_2"])], dtype=float)
    bucket_id = np.digitize(values, thresholds, right=False)
    scales = np.array([float(params["scale_low"]), float(params["scale_mid"]), float(params["scale_high"])], dtype=float)
    return prediction * scales[bucket_id]


def simplex_weights(names: list[str], step: float) -> Iterable[dict[str, float]]:
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for values in product(grid, repeat=len(names)):
        total = float(sum(values))
        if abs(total - 1.0) > step / 2.0 or total <= 0.0:
            continue
        yield {name: float(value) / total for name, value in zip(names, values)}


def apply_simplex(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    prediction = np.zeros(len(frame), dtype=float)
    for name, weight in weights.items():
        prediction += float(weight) * frame[f"pred_{name}"].to_numpy(dtype=float)
    return prediction


def fit_simplex(train: pd.DataFrame, names: list[str], weight_step: float, tail_penalty: float, objective_name: str) -> dict[str, float]:
    best_weights = {names[0]: 1.0}
    best_score = np.inf if objective_name == "tail" else -np.inf
    for weights in simplex_weights(names, weight_step):
        prediction = apply_simplex(train, weights)
        score = tail_objective(train, prediction, tail_penalty) if objective_name == "tail" else rel_score(
            train["actual"].to_numpy(dtype=float),
            prediction,
        )
        if (objective_name == "tail" and score < best_score) or (objective_name != "tail" and score > best_score):
            best_score = float(score)
            best_weights = weights
    return best_weights


def fit_params(train: pd.DataFrame, scale_grid: np.ndarray, weight_step: float, tail_penalty: float) -> dict[str, object]:
    actual = train["actual"].to_numpy(dtype=float)
    params: dict[str, object] = {}
    for source in PREDICTION_SPECS:
        prediction = train[f"pred_{source}"].to_numpy(dtype=float)
        scale, score = fit_best_scale(actual, prediction, scale_grid)
        params[f"{source}_scale"] = {"source": source, "scale": scale, "train_rel_score": score}
    for source in ("tail_loss", "tailstress", "stressaux_w20"):
        params[f"{source}_sign_scale"] = fit_sign_scales(train, source, scale_grid)
        params[f"{source}_daily_scale"] = fit_daily_scale(train, source, scale_grid, tail_penalty)
    core_names = ["tail_loss", "tailstress", "stressaux_w20", "base", "instance"]
    params["simplex_rel"] = fit_simplex(train, core_names, weight_step, tail_penalty, "rel")
    params["simplex_tail"] = fit_simplex(train, core_names, weight_step, tail_penalty, "tail")
    return params


def candidate_predictions(frame: pd.DataFrame, params: dict[str, object]) -> dict[str, np.ndarray]:
    candidates = {name: frame[f"pred_{name}"].to_numpy(dtype=float) for name in PREDICTION_SPECS}
    for source in PREDICTION_SPECS:
        scale_params = params[f"{source}_scale"]
        assert isinstance(scale_params, dict)
        candidates[f"{source}_scale"] = frame[f"pred_{source}"].to_numpy(dtype=float) * float(scale_params["scale"])
    for source in ("tail_loss", "tailstress", "stressaux_w20"):
        sign_params = params[f"{source}_sign_scale"]
        daily_params = params[f"{source}_daily_scale"]
        assert isinstance(sign_params, dict)
        assert isinstance(daily_params, dict)
        candidates[f"{source}_sign_scale"] = apply_sign_scales(frame, sign_params)  # type: ignore[arg-type]
        candidates[f"{source}_daily_scale"] = apply_daily_scale(frame, daily_params)  # type: ignore[arg-type]
    candidates["simplex_rel"] = apply_simplex(frame, params["simplex_rel"])  # type: ignore[arg-type]
    candidates["simplex_tail"] = apply_simplex(frame, params["simplex_tail"])  # type: ignore[arg-type]
    return candidates


def aggregate_validation(by_seed: pd.DataFrame) -> pd.DataFrame:
    val = by_seed[by_seed["split"].eq("val")].copy()
    return (
        val.groupby("candidate", sort=True)
        .agg(
            rel_score_mean=("rel_score", "mean"),
            rel_score_std=("rel_score", "std"),
            q90_abs_error_mean=("q90_abs_error", "mean"),
            daily_q90_abs_error_q90_mean=("daily_q90_abs_error_q90", "mean"),
            daily_q90_abs_error_max_mean=("daily_q90_abs_error_max", "mean"),
            spike_days_ge_5pct_mean=("spike_days_ge_5pct", "mean"),
            spike_days_ge_7pct_mean=("spike_days_ge_7pct", "mean"),
            spike_days_ge_8pct_mean=("spike_days_ge_8pct", "mean"),
            prediction_actual_abs_q90_ratio_mean=("prediction_actual_abs_q90_ratio", "mean"),
            directional_accuracy_mean=("directional_accuracy", "mean"),
        )
        .reset_index()
    )


def format_percent(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * float(value):.3f}%"


def write_plot(output_dir: Path, aggregate: pd.DataFrame) -> None:
    selected = [
        "base",
        "tail_loss",
        "tailstress",
        "stressaux_w20",
        "simplex_rel",
        "simplex_tail",
        "stressaux_w20_daily_scale",
    ]
    plot = aggregate[aggregate["candidate"].isin(selected)].copy()
    plot["candidate"] = pd.Categorical(plot["candidate"], categories=selected, ordered=True)
    plot = plot.sort_values("candidate")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].barh(plot["candidate"].astype(str), plot["rel_score_mean"], xerr=plot["rel_score_std"], color="#2563eb", alpha=0.85)
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set_title("Mean validation rel_score")
    axes[1].barh(plot["candidate"].astype(str), 100.0 * plot["daily_q90_abs_error_max_mean"], color="#f59e0b", alpha=0.85)
    axes[1].axvline(5.0, color="#16a34a", linestyle="--", linewidth=1, label="5% target")
    axes[1].set_title("Mean max daily q90 error")
    axes[2].barh(plot["candidate"].astype(str), plot["spike_days_ge_8pct_mean"], color="#dc2626", alpha=0.85)
    axes[2].set_title("Mean days q90 error >= 8%")
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.25)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "tail_calibration_tradeoff.png", dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, by_seed: pd.DataFrame, aggregate: pd.DataFrame, params: pd.DataFrame) -> None:
    by_seed_val = by_seed[by_seed["split"].eq("val")].copy().sort_values(["seed", "rel_score"], ascending=[True, False])
    aggregate_ranked = aggregate.copy().sort_values("rel_score_mean", ascending=False)
    top_seed = by_seed_val.groupby("seed", group_keys=False).head(8)
    lines = [
        "# Tail Calibration Probe",
        "",
        "Calibration is fitted only on train and evaluated on validation. Holdout/test is not used.",
        "",
        "Purpose: check whether the remaining LSTM issue is mainly output under-amplification or missing tail/regime signal.",
        "",
        "## Top Validation By Seed",
        "",
        "| seed | candidate | rel_score | q90 error | daily q90 p90 | daily max | days >=5% | days >=7% | days >=8% | pred/actual q90 |",
        "|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in top_seed.iterrows():
        lines.append(
            f"| {int(row.seed)} | `{row.candidate}` | {float(row.rel_score):.5f} | "
            f"{format_percent(row.q90_abs_error)} | {format_percent(row.daily_q90_abs_error_q90)} | "
            f"{format_percent(row.daily_q90_abs_error_max)} | {float(row.spike_days_ge_5pct):.1f} | "
            f"{float(row.spike_days_ge_7pct):.1f} | {float(row.spike_days_ge_8pct):.1f} | "
            f"{float(row.prediction_actual_abs_q90_ratio):.3f} |"
        )
    lines += [
        "",
        "## Aggregate Validation",
        "",
        "| candidate | rel_score mean | rel_score std | q90 error mean | daily q90 p90 mean | daily max mean | days >=5% mean | days >=7% mean | days >=8% mean | pred/actual q90 mean |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in aggregate_ranked.iterrows():
        lines.append(
            f"| `{row.candidate}` | {float(row.rel_score_mean):.5f} | {float(row.rel_score_std):.5f} | "
            f"{format_percent(row.q90_abs_error_mean)} | {format_percent(row.daily_q90_abs_error_q90_mean)} | "
            f"{format_percent(row.daily_q90_abs_error_max_mean)} | {float(row.spike_days_ge_5pct_mean):.1f} | "
            f"{float(row.spike_days_ge_7pct_mean):.1f} | {float(row.spike_days_ge_8pct_mean):.1f} | "
            f"{float(row.prediction_actual_abs_q90_ratio_mean):.3f} |"
        )
    lines += [
        "",
        "## Fitted Parameters",
        "",
        params.to_markdown(index=False),
        "",
        "## Read",
        "",
        "- If scaling wins, the model mainly under-amplifies return magnitude.",
        "- If simplex/gating wins, different LSTM variants capture complementary regimes.",
        "- If none of the calibration candidates lowers spikes without sacrificing `rel_score`, the next improvement must come from input processing or a better tail-aware training objective.",
    ]
    (output_dir / "tail_calibration_readout.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scale_grid = np.arange(0.0, args.scale_max + 1e-9, args.scale_step)
    rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    for seed in seeds:
        frame = load_seed_predictions(args.report_root, seed)
        train = frame[frame["split"].eq("train")].copy()
        params = fit_params(train, scale_grid, args.weight_step, args.tail_penalty)
        param_rows.append({"seed": seed, "params": json.dumps(params, sort_keys=True)})
        for split, split_frame in frame.groupby("split", sort=True):
            for candidate, prediction in candidate_predictions(split_frame, params).items():
                rows.append(summarize_prediction(split_frame, prediction, candidate, seed, str(split)))
    by_seed = pd.DataFrame(rows)
    aggregate = aggregate_validation(by_seed)
    params_frame = pd.DataFrame(param_rows)
    by_seed.to_csv(args.output_dir / "tail_calibration_by_seed.csv", index=False)
    aggregate.to_csv(args.output_dir / "tail_calibration_aggregate.csv", index=False)
    params_frame.to_csv(args.output_dir / "tail_calibration_params.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "scale_max": args.scale_max,
                "scale_step": args.scale_step,
                "weight_step": args.weight_step,
                "tail_penalty": args.tail_penalty,
                "holdout_test_used": False,
                "prediction_specs": PREDICTION_SPECS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_plot(args.output_dir, aggregate)
    write_report(args.output_dir, by_seed, aggregate, params_frame)
    print((args.output_dir / "tail_calibration_readout.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
