"""LSTM reliability diagnostics: calibration + bootstrap CI + seed stability.

Mục đích: bổ sung báo cáo độ tin cậy thống kê cho một training run/predictions.csv:

1. **Reliability diagram (magnitude calibration)** — chia |prediction| theo
   decile, so sánh với mean |actual| trong từng bucket. Tốt = đường y=x.
2. **Sign reliability** — chia P(sign) theo bucket (cho signmag), so sánh hit
   rate thực tế. Tốt = monotone increasing.
3. **Bootstrap CI** cho rel_score, IC, hit-rate, top-bottom equity. 1000
   resample mặc định.
4. **Seed stability** — nếu predictions.csv có nhiều model_name dạng
   `lstm_signmag_seed_*`, in bảng per-seed rel_score/IC để biết phương sai.
5. **IC by year, by regime** — nếu có regime column.

Output: thư mục `reports/core/reliability/` với:
- `summary.md`
- `magnitude_calibration.csv`
- `sign_calibration.csv` (nếu có sign_prob)
- `bootstrap_metrics.csv`
- `seed_stability.csv` (nếu nhiều seed)
- `ic_by_year.csv`
- `ic_by_regime.csv` (nếu có regime)

Cách chạy:

```
python experiments/analysis/lstm_reliability_diagnostics.py \\
    --predictions-csv data/.../run/predictions.csv \\
    --model lstm_signmag_mean_ensemble \\
    --split val \\
    --output-dir data/.../run/reports/core/reliability \\
    --bootstrap-iterations 1000 \\
    --regime-csv data/.../run/regimes.csv
```
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse rel_score evaluation from production metric module.
from src.evaluation.metric import evaluate  # noqa: E402


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> float:
    series_x = pd.Series(x)
    series_y = pd.Series(y)
    valid = series_x.notna() & series_y.notna()
    if valid.sum() < 5:
        return float("nan")
    return float(series_x[valid].corr(series_y[valid], method=method))


def magnitude_calibration(prediction: np.ndarray, actual: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    abs_pred = np.abs(prediction)
    abs_actual = np.abs(actual)
    mask = np.isfinite(abs_pred) & np.isfinite(abs_actual)
    abs_pred = abs_pred[mask]
    abs_actual = abs_actual[mask]
    if len(abs_pred) < n_buckets * 5:
        return pd.DataFrame(columns=["bucket", "n", "mean_pred_abs", "mean_actual_abs", "ratio"])
    edges = np.quantile(abs_pred, np.linspace(0.0, 1.0, n_buckets + 1))
    # Make edges strictly increasing
    for i in range(1, len(edges)):
        edges[i] = max(edges[i], edges[i - 1] + 1e-12)
    buckets = np.digitize(abs_pred, edges[1:-1], right=True)
    rows = []
    for b in range(n_buckets):
        sel = buckets == b
        if sel.sum() == 0:
            continue
        mean_pred = float(np.mean(abs_pred[sel]))
        mean_actual = float(np.mean(abs_actual[sel]))
        rows.append(
            {
                "bucket": b,
                "n": int(sel.sum()),
                "mean_pred_abs": mean_pred,
                "mean_actual_abs": mean_actual,
                "ratio": float(mean_actual / mean_pred) if mean_pred > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def sign_calibration(sign_prob: np.ndarray, actual_sign: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    """sign_prob in [0, 1], actual_sign in {-1, 0, +1} (we treat +1 as positive class)."""
    p = np.asarray(sign_prob, dtype=float)
    a = np.asarray(actual_sign, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a)
    p = p[mask]
    a = (a[mask] > 0).astype(int)
    if len(p) < n_buckets * 5:
        return pd.DataFrame(columns=["bucket", "n", "mean_prob", "hit_rate"])
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_buckets + 1))
    for i in range(1, len(edges)):
        edges[i] = max(edges[i], edges[i - 1] + 1e-12)
    buckets = np.digitize(p, edges[1:-1], right=True)
    rows = []
    for b in range(n_buckets):
        sel = buckets == b
        if sel.sum() == 0:
            continue
        rows.append(
            {
                "bucket": b,
                "n": int(sel.sum()),
                "mean_prob": float(np.mean(p[sel])),
                "hit_rate": float(np.mean(a[sel])),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric(
    fn,
    *args,
    n_iter: int = 1000,
    rng: np.random.Generator | None = None,
    block_length: int | None = None,
) -> dict[str, float]:
    """Generic bootstrap CI helper.

    fn must accept the same positional arrays as the call site, and return float.
    Resampling is row-level by default; pass `block_length` for block bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng(20260514)
    arrays = [np.asarray(a, dtype=float) for a in args]
    if not arrays:
        raise ValueError("Need at least one array")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All arrays must have same length")
    if n < 20:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "std": float("nan")}
    point = float(fn(*arrays))

    boots = np.empty(n_iter, dtype=float)
    for b in range(n_iter):
        if block_length is None or block_length <= 1:
            idx = rng.integers(0, n, size=n)
        else:
            # Stationary-bootstrap-lite: contiguous blocks with random starts
            starts = rng.integers(0, n, size=math.ceil(n / block_length))
            chunks = [np.arange(s, s + block_length) % n for s in starts]
            idx = np.concatenate(chunks)[:n]
        resampled = [a[idx] for a in arrays]
        try:
            boots[b] = float(fn(*resampled))
        except Exception:  # noqa: BLE001
            boots[b] = float("nan")
    boots = boots[np.isfinite(boots)]
    if len(boots) < 10:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"), "std": float("nan")}
    return {
        "point": point,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "std": float(np.std(boots, ddof=1)),
    }


def compute_rel_score(prediction: np.ndarray, actual: np.ndarray) -> float:
    """Reuse evaluate() but skip its 1-day shift — call directly on aligned arrays.

    Note: this is intentional. We assume `prediction` and `actual` are already
    aligned 1:1 by the caller; evaluate() does an internal alignment that we
    bypass here for bootstrap convenience.
    """
    err = actual - prediction
    abs_err = np.abs(err)
    abs_base = np.abs(actual)
    if len(abs_err) < 3:
        return float("nan")
    loss_err = float(np.quantile(abs_err, 0.5) + 0.5 * np.quantile(abs_err, 0.9))
    loss_base = float(np.quantile(abs_base, 0.5) + 0.5 * np.quantile(abs_base, 0.9))
    if loss_base <= 0.0:
        return float("nan")
    return 1.0 - loss_err / loss_base


def compute_hit_rate(prediction: np.ndarray, actual: np.ndarray) -> float:
    pred_sign = np.sign(prediction)
    actual_sign = np.sign(actual)
    mask = (pred_sign != 0) & np.isfinite(prediction) & np.isfinite(actual)
    if mask.sum() < 5:
        return float("nan")
    return float(np.mean(pred_sign[mask] == actual_sign[mask]))


def compute_spearman_ic_daily(df: pd.DataFrame, prediction_col: str = "prediction", actual_col: str = "actual") -> pd.Series:
    out = []
    for date, day in df.groupby("Date", sort=True):
        if len(day) < 5:
            continue
        ic = _safe_corr(day[prediction_col].to_numpy(dtype=float), day[actual_col].to_numpy(dtype=float))
        out.append({"Date": date, "ic": ic})
    if not out:
        return pd.Series([], name="ic")
    out_df = pd.DataFrame(out).set_index("Date")["ic"]
    return out_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None, help="Specific model name to filter by.")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--n-buckets", type=int, default=10)
    parser.add_argument("--regime-csv", type=Path, default=None,
                        help="Optional CSV with Date, regime columns for breakdown.")
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv, parse_dates=["Date"])
    if "split" in df.columns:
        df = df.loc[df["split"] == args.split].copy()
    if args.model is not None:
        if "model" not in df.columns:
            raise SystemExit("--model requires a 'model' column in predictions CSV")
        df = df.loc[df["model"] == args.model].copy()

    if df.empty:
        raise SystemExit(f"No rows after filtering split={args.split} model={args.model}")

    needed = {"Date", "code", "prediction", "actual"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["prediction", "actual"]).copy()
    pred = df["prediction"].to_numpy(dtype=float)
    act = df["actual"].to_numpy(dtype=float)

    rng = np.random.default_rng(args.seed)

    # 1. Magnitude calibration
    mag = magnitude_calibration(pred, act, n_buckets=args.n_buckets)
    mag.to_csv(args.output_dir / "magnitude_calibration.csv", index=False)

    # 2. Sign calibration (if available)
    sign_cal = pd.DataFrame()
    if "sign_prob" in df.columns:
        sign_cal = sign_calibration(df["sign_prob"].to_numpy(dtype=float), np.sign(act), n_buckets=args.n_buckets)
        sign_cal.to_csv(args.output_dir / "sign_calibration.csv", index=False)

    # 3. Bootstrap CIs
    rel_ci = bootstrap_metric(compute_rel_score, pred, act, n_iter=args.bootstrap_iterations, rng=rng, block_length=args.block_length)
    hit_ci = bootstrap_metric(compute_hit_rate, pred, act, n_iter=args.bootstrap_iterations, rng=rng, block_length=args.block_length)

    ic_series = compute_spearman_ic_daily(df)
    ic_arr = ic_series.to_numpy(dtype=float)
    ic_ci = (
        bootstrap_metric(lambda x: float(np.mean(x[np.isfinite(x)])), ic_arr,
                         n_iter=args.bootstrap_iterations, rng=rng, block_length=args.block_length)
        if len(ic_arr) >= 20 else {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "std": float("nan")}
    )

    metrics_df = pd.DataFrame(
        [
            {"metric": "rel_score", **rel_ci},
            {"metric": "hit_rate", **hit_ci},
            {"metric": "mean_daily_ic", **ic_ci},
        ]
    )
    metrics_df.to_csv(args.output_dir / "bootstrap_metrics.csv", index=False)

    # 4. Seed stability
    seed_table = pd.DataFrame()
    if "model" in df.columns and args.model is None:
        # Aggregate per seed-like model name
        original = pd.read_csv(args.predictions_csv, parse_dates=["Date"])
        if "split" in original.columns:
            original = original.loc[original["split"] == args.split].copy()
        seed_rows = []
        for model_name, group in original.groupby("model"):
            if "_seed_" not in str(model_name):
                continue
            g = group.dropna(subset=["prediction", "actual"])
            seed_rows.append(
                {
                    "model": model_name,
                    "n": len(g),
                    "rel_score": compute_rel_score(g["prediction"].to_numpy(dtype=float), g["actual"].to_numpy(dtype=float)),
                    "hit_rate": compute_hit_rate(g["prediction"].to_numpy(dtype=float), g["actual"].to_numpy(dtype=float)),
                    "mean_daily_ic": float(compute_spearman_ic_daily(g).mean()),
                }
            )
        if seed_rows:
            seed_table = pd.DataFrame(seed_rows).sort_values("rel_score", ascending=False)
            seed_table.to_csv(args.output_dir / "seed_stability.csv", index=False)

    # 5. IC by year
    df_ic = df.copy()
    df_ic["year"] = df_ic["Date"].dt.year
    ic_by_year_rows = []
    for year, year_df in df_ic.groupby("year"):
        series = compute_spearman_ic_daily(year_df)
        if series.empty:
            continue
        ic_by_year_rows.append(
            {
                "year": year,
                "n_days": int(len(series)),
                "mean_daily_ic": float(series.mean()),
                "t_stat": float(series.mean() / (series.std(ddof=1) / math.sqrt(len(series)))) if len(series) >= 2 and series.std(ddof=1) > 0 else float("nan"),
                "positive_days_pct": float((series > 0).mean()),
            }
        )
    ic_by_year = pd.DataFrame(ic_by_year_rows)
    ic_by_year.to_csv(args.output_dir / "ic_by_year.csv", index=False)

    # 6. IC by regime (optional)
    ic_by_regime = pd.DataFrame()
    if args.regime_csv is not None and args.regime_csv.exists():
        regimes = pd.read_csv(args.regime_csv, parse_dates=["Date"])
        if "Date" in regimes.columns and "regime" in regimes.columns:
            merged = df.merge(regimes[["Date", "regime"]], on="Date", how="left")
            ic_rows = []
            for regime, regime_df in merged.dropna(subset=["regime"]).groupby("regime"):
                series = compute_spearman_ic_daily(regime_df)
                if series.empty:
                    continue
                ic_rows.append(
                    {
                        "regime": regime,
                        "n_days": int(len(series)),
                        "mean_daily_ic": float(series.mean()),
                        "t_stat": float(series.mean() / (series.std(ddof=1) / math.sqrt(len(series)))) if len(series) >= 2 and series.std(ddof=1) > 0 else float("nan"),
                    }
                )
            if ic_rows:
                ic_by_regime = pd.DataFrame(ic_rows).sort_values("mean_daily_ic", ascending=False)
                ic_by_regime.to_csv(args.output_dir / "ic_by_regime.csv", index=False)

    # Markdown summary
    lines = [
        f"# LSTM Reliability Diagnostics — {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        f"- predictions: `{args.predictions_csv}`",
        f"- split: `{args.split}`",
        f"- model filter: `{args.model or 'all'}`",
        f"- n_observations: `{len(df)}`",
        f"- bootstrap iterations: `{args.bootstrap_iterations}`",
        f"- block length: `{args.block_length}`",
        "",
        "## Headline Metrics (with bootstrap 95% CI)",
        "",
        "| metric | point | ci_low | ci_high | std |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['metric']} | {row['point']:.5f} | {row['ci_low']:.5f} | {row['ci_high']:.5f} | {row['std']:.5f} |"
        )

    lines += [
        "",
        "## Magnitude Calibration",
        "",
        f"See [magnitude_calibration.csv](magnitude_calibration.csv). Perfect calibration: `ratio = 1.0` in every bucket. ",
        f"Mean ratio across buckets: `{mag['ratio'].mean():.3f}` (std `{mag['ratio'].std(ddof=1):.3f}`)" if not mag.empty else "_(insufficient data)_",
    ]

    if not sign_cal.empty:
        lines += [
            "",
            "## Sign Calibration",
            "",
            "See [sign_calibration.csv](sign_calibration.csv). Perfect: hit_rate ≈ mean_prob across buckets.",
        ]

    if not seed_table.empty:
        lines += [
            "",
            "## Per-Seed Stability",
            "",
            "| seed model | n | rel_score | hit_rate | mean_daily_ic |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for _, row in seed_table.iterrows():
            lines.append(
                f"| {row['model']} | {row['n']} | {row['rel_score']:.5f} | {row['hit_rate']:.4f} | {row['mean_daily_ic']:.5f} |"
            )

    if not ic_by_year.empty:
        lines += [
            "",
            "## IC by Year",
            "",
            "| year | n_days | mean_daily_ic | t_stat | positive_days_pct |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for _, row in ic_by_year.iterrows():
            lines.append(
                f"| {row['year']} | {row['n_days']} | {row['mean_daily_ic']:.5f} | {row['t_stat']:.3f} | {row['positive_days_pct']:.3f} |"
            )

    if not ic_by_regime.empty:
        lines += [
            "",
            "## IC by Regime",
            "",
            "| regime | n_days | mean_daily_ic | t_stat |",
            "| --- | ---: | ---: | ---: |",
        ]
        for _, row in ic_by_regime.iterrows():
            lines.append(
                f"| {row['regime']} | {row['n_days']} | {row['mean_daily_ic']:.5f} | {row['t_stat']:.3f} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **rel_score CI** that excludes 0 → model has statistically distinguishable edge after block-bootstrap noise.",
        "- **mean_ratio of magnitude calibration** far from 1.0 → model is systematically under/overestimating moves. ",
        "  Ratio > 1 = model under-predicts magnitude; ratio < 1 = over-predicts.",
        "- **IC stability across years/regimes**: t_stat < 2 in any regime/year suggests fragility there.",
        "- **Seed stability**: std of per-seed rel_score >> point CI width means mean ensemble is essential.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
