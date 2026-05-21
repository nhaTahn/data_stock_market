from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "tailstress_gate_probe_20260519"
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate train-fitted gates for tailstress LSTM predictions.")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-step", type=float, default=0.10)
    parser.add_argument("--tail-penalty", type=float, default=0.50)
    return parser.parse_args(argv)


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
    daily = (
        pd.DataFrame({"Date": pd.to_datetime(frame["Date"]), "abs_error": np.abs(actual - prediction)})
        .groupby("Date", sort=True)["abs_error"]
        .quantile(0.90)
    )
    return daily


def objective(frame: pd.DataFrame, prediction: np.ndarray, tail_penalty: float) -> float:
    actual = frame["actual"].to_numpy(dtype=float)
    daily = daily_q90(frame, prediction)
    tail_term = float(daily.quantile(0.90)) if not daily.empty else 0.0
    return robust_loss(actual - prediction) + tail_penalty * tail_term


def summarize(frame: pd.DataFrame, prediction: np.ndarray, candidate: str, split: str) -> dict[str, object]:
    actual = frame["actual"].to_numpy(dtype=float)
    error = actual - prediction
    abs_error = np.abs(error)
    daily = daily_q90(frame, prediction)
    actual_abs_q90 = float(np.quantile(np.abs(actual), 0.90))
    pred_abs_q90 = float(np.quantile(np.abs(prediction), 0.90))
    return {
        "candidate": candidate,
        "split": split,
        "n_obs": int(len(frame)),
        "n_days": int(daily.shape[0]),
        "rel_score": rel_score(actual, prediction),
        "q90_abs_error": float(np.quantile(abs_error, 0.90)),
        "daily_q90_abs_error_q90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "daily_q90_abs_error_max": float(daily.max()) if not daily.empty else float("nan"),
        "spike_days_ge_7pct": int(daily.ge(0.07).sum()) if not daily.empty else 0,
        "spike_days_ge_8pct": int(daily.ge(0.08).sum()) if not daily.empty else 0,
        "prediction_actual_abs_q90_ratio": pred_abs_q90 / actual_abs_q90 if actual_abs_q90 > 0.0 else float("nan"),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(prediction))),
    }


def prediction_path(report_root: Path, seed: int, family: str, variant: str) -> Path:
    return report_root / family / f"seed_{seed}" / f"predictions_{variant}.csv"


def load_predictions(report_root: Path, seed: int) -> pd.DataFrame:
    specs = {
        "base": ("tail_aware_lstm_multiseed_20260519", "plain_global_rel"),
        "weighted": ("tail_aware_lstm_multiseed_20260519", "plain_global_weighted_mild"),
        "instance": ("tail_aware_lstm_multiseed_20260519", "plain_global_instance_rel"),
        "tail_loss": ("tail_loss_lstm_probe_20260519", "plain_global_weighted_mild_tail35_p05"),
        "tailstress": ("tailstress_lstm_probe_20260519", "plain_global_weighted_mild_tail35_p05_tailstress"),
        "tailstress_past": (
            "tailstress_lstm_probe_20260519",
            "plain_global_weighted_mild_tail35_p05_tailstress_past",
        ),
    }
    merged: pd.DataFrame | None = None
    for short_name, (family, variant) in specs.items():
        path = prediction_path(report_root, seed, family, variant)
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
        raise ValueError("No predictions loaded.")
    return merged


def simplex_weights(names: list[str], step: float) -> list[dict[str, float]]:
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    out: list[dict[str, float]] = []
    for values in product(grid, repeat=len(names)):
        total = float(sum(values))
        if abs(total - 1.0) > step / 2:
            continue
        out.append({name: float(value) / total for name, value in zip(names, values) if total > 0.0})
    return out


def apply_simplex(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=float)
    for name, weight in weights.items():
        pred += weight * frame[f"pred_{name}"].to_numpy(dtype=float)
    return pred


def fit_simplex(train: pd.DataFrame, names: list[str], step: float, tail_penalty: float, mode: str) -> dict[str, float]:
    best_weights: dict[str, float] = {names[0]: 1.0}
    best_score = np.inf if mode == "tail_objective" else -np.inf
    for weights in simplex_weights(names, step):
        prediction = apply_simplex(train, weights)
        score = objective(train, prediction, tail_penalty) if mode == "tail_objective" else rel_score(
            train["actual"].to_numpy(dtype=float),
            prediction,
        )
        if (mode == "tail_objective" and score < best_score) or (mode != "tail_objective" and score > best_score):
            best_score = float(score)
            best_weights = weights
    return best_weights


def daily_gate_signal(frame: pd.DataFrame, mode: str) -> pd.Series:
    work = frame.copy()
    if mode == "disagreement":
        work["signal"] = (work["pred_tailstress"] - work["pred_tail_loss"]).abs()
    elif mode == "tailstress_abs":
        work["signal"] = work["pred_tailstress"].abs()
    elif mode == "mean_positive_gap":
        work["signal"] = work["pred_tailstress"] - work["pred_tail_loss"]
    else:
        raise ValueError(f"Unknown gate mode: {mode}")
    if mode == "mean_positive_gap":
        return work.groupby("Date", sort=True)["signal"].mean()
    return work.groupby("Date", sort=True)["signal"].quantile(0.90)


def apply_daily_gate(
    frame: pd.DataFrame,
    mode: str,
    threshold: float,
    high_candidate: str,
    low_candidate: str,
) -> np.ndarray:
    signal = daily_gate_signal(frame, mode)
    high_days = set(signal[signal >= threshold].index)
    is_high = pd.to_datetime(frame["Date"]).isin(high_days).to_numpy()
    high = frame[f"pred_{high_candidate}"].to_numpy(dtype=float)
    low = frame[f"pred_{low_candidate}"].to_numpy(dtype=float)
    return np.where(is_high, high, low)


def fit_daily_gate(
    train: pd.DataFrame,
    mode: str,
    high_candidate: str,
    low_candidate: str,
    tail_penalty: float,
) -> tuple[float, str, str]:
    signal = daily_gate_signal(train, mode)
    thresholds = np.quantile(signal.to_numpy(dtype=float), np.linspace(0.50, 0.95, 10))
    best = (float(thresholds[0]), high_candidate, low_candidate, np.inf)
    for threshold in thresholds:
        for high, low in [(high_candidate, low_candidate), (low_candidate, high_candidate)]:
            prediction = apply_daily_gate(train, mode, float(threshold), high, low)
            score = objective(train, prediction, tail_penalty)
            if score < best[3]:
                best = (float(threshold), high, low, float(score))
    return best[0], best[1], best[2]


def fit_daily_gate_fixed_orientation(
    train: pd.DataFrame,
    mode: str,
    high_candidate: str,
    low_candidate: str,
    tail_penalty: float,
) -> tuple[float, str, str]:
    signal = daily_gate_signal(train, mode)
    thresholds = np.quantile(signal.to_numpy(dtype=float), np.linspace(0.50, 0.95, 10))
    best = (float(thresholds[0]), high_candidate, low_candidate, np.inf)
    for threshold in thresholds:
        prediction = apply_daily_gate(train, mode, float(threshold), high_candidate, low_candidate)
        score = objective(train, prediction, tail_penalty)
        if score < best[3]:
            best = (float(threshold), high_candidate, low_candidate, float(score))
    return best[0], best[1], best[2]


def build_candidates(frame: pd.DataFrame, params: dict[str, object]) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    for name in ["base", "weighted", "instance", "tail_loss", "tailstress", "tailstress_past"]:
        candidates[name] = frame[f"pred_{name}"].to_numpy(dtype=float)
    candidates["simplex_rel"] = apply_simplex(frame, params["simplex_rel"])  # type: ignore[arg-type]
    candidates["simplex_tail"] = apply_simplex(frame, params["simplex_tail"])  # type: ignore[arg-type]
    for mode in ["disagreement", "tailstress_abs", "mean_positive_gap"]:
        threshold, high, low = params[f"gate_{mode}"]  # type: ignore[misc]
        candidates[f"gate_{mode}"] = apply_daily_gate(frame, mode, float(threshold), str(high), str(low))
        conservative_threshold, conservative_high, conservative_low = params[f"conservative_gate_{mode}"]  # type: ignore[misc]
        candidates[f"conservative_{mode}"] = apply_daily_gate(
            frame,
            mode,
            float(conservative_threshold),
            str(conservative_high),
            str(conservative_low),
        )
    return candidates


def write_report(output_dir: Path, by_seed: pd.DataFrame, params: dict[str, object]) -> None:
    val = by_seed[by_seed["split"].eq("val")].copy().sort_values("rel_score", ascending=False)
    lines = [
        "# Tailstress Gate Probe",
        "",
        "Rules are fitted on train and evaluated on validation. Holdout/test is not used.",
        "",
        "## Validation",
        "",
        "| candidate | rel_score | q90 error | daily q90 p90 | daily q90 max | days >=7% | days >=8% | pred/actual abs q90 | dir_acc |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in val.iterrows():
        lines.append(
            f"| `{row.candidate}` | {row.rel_score:.5f} | {100 * row.q90_abs_error:.3f}% | "
            f"{100 * row.daily_q90_abs_error_q90:.3f}% | {100 * row.daily_q90_abs_error_max:.3f}% | "
            f"{int(row.spike_days_ge_7pct)} | {int(row.spike_days_ge_8pct)} | "
            f"{row.prediction_actual_abs_q90_ratio:.3f} | {row.directional_accuracy:.3f} |"
        )
    lines += [
        "",
        "## Fitted Parameters",
        "",
        "```json",
        json.dumps(params, indent=2),
        "```",
        "",
        "## Read",
        "",
        "- A gate is useful only if it improves `rel_score` without increasing daily max/spike count.",
        "- Direct `tailstress` remains a diagnostic signal if it wins `rel_score` but worsens spikes.",
        "- If a train-fitted gate cannot beat `tail_loss`, tailstress should not be injected directly into the production LSTM yet.",
    ]
    (output_dir / "tailstress_gate_readout.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_predictions(args.report_root, args.seed)
    train = frame[frame["split"].eq("train")].copy()
    params: dict[str, object] = {
        "simplex_rel": fit_simplex(
            train,
            ["tail_loss", "tailstress", "tailstress_past", "base", "instance"],
            args.grid_step,
            args.tail_penalty,
            mode="rel_score",
        ),
        "simplex_tail": fit_simplex(
            train,
            ["tail_loss", "tailstress", "tailstress_past", "base", "instance"],
            args.grid_step,
            args.tail_penalty,
            mode="tail_objective",
        ),
    }
    for mode in ["disagreement", "tailstress_abs", "mean_positive_gap"]:
        params[f"gate_{mode}"] = fit_daily_gate(train, mode, "tail_loss", "tailstress", args.tail_penalty)
        params[f"conservative_gate_{mode}"] = fit_daily_gate_fixed_orientation(
            train,
            mode,
            "tail_loss",
            "tailstress",
            args.tail_penalty,
        )

    rows: list[dict[str, object]] = []
    for split, split_frame in frame.groupby("split", sort=True):
        candidates = build_candidates(split_frame, params)
        for name, prediction in candidates.items():
            rows.append(summarize(split_frame, prediction, name, str(split)))
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "tailstress_gate_summary.csv", index=False)
    (output_dir / "tailstress_gate_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "tail_penalty": args.tail_penalty,
                "grid_step": args.grid_step,
                "holdout_test_used": False,
                "report_root": str(args.report_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(output_dir, summary, params)
    print((output_dir / "tailstress_gate_readout.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
