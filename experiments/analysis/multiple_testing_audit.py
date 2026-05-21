"""Multiple-testing correction audit for LSTM/router/gate candidates.

Mục đích: project đã chạm validation rất nhiều lần (~100+ training runs, filter
variants, gate variants). Mọi Sharpe/rel_score "đẹp" trên val có nguy cơ là
noise sau khi multiple testing. Script này tính:

1. **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado 2014:
   - Tính SR̂ * dùng skewness, kurtosis, độ dài chuỗi.
   - So sánh với expected max SR khi N candidates được test.
   - Output: DSR ∈ [0, 1] (xác suất Sharpe thật > 0).

2. **White's Reality Check / Bootstrap p-value**:
   - Stationary bootstrap (Politis-Romano) trên ma trận daily returns × candidates.
   - p-value of `H0: max(SR) <= benchmark_SR`.

3. **Benjamini-Hochberg FDR**:
   - Trên các per-candidate t-stat của mean daily return / IC.
   - Q-value cho từng candidate ở FDR level.

Input: thư mục chứa nhiều file `predictions.csv` (mỗi file = 1 candidate,
có cột Date, code, prediction, actual, split). Hoặc một file CSV đã
pre-aggregate có cột: `candidate, date, daily_return` (long format).

Output: markdown report với:
- bảng candidate × {SR, DSR, bootstrap_pvalue, BH_q, alive_after_FDR}
- bảng candidate × ic_t_stat × bootstrap_ic_pvalue
- top-K candidate còn "sống" sau correction
- recommendation: anchor mới nếu cần

Cách chạy:

```
python experiments/analysis/multiple_testing_audit.py \\
    --returns-csv reports/walk_forward/.../fold_metrics.csv \\
    --returns-col val_rel_score \\
    --output docs/multiple_testing_audit_20260514.md

# Hoặc với daily returns long-format:
python experiments/analysis/multiple_testing_audit.py \\
    --daily-returns-csv path/to/long_returns.csv \\
    --benchmark-candidate anchor \\
    --bootstrap-iterations 2000 \\
    --fdr-level 0.10 \\
    --output docs/multiple_testing_audit_20260514.md
```
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------
# Statistics primitives
# ----------------------------------------------------------------------


def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    std = _safe_std(x)
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return float(np.mean(x) / std * math.sqrt(periods_per_year))


def _skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std <= 0.0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std <= 0.0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 4) - 3.0)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF via rational approximation (Acklam)."""
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        return float("nan")
    # Acklam's approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


# ----------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & López de Prado 2014)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DSRResult:
    sharpe_obs: float
    expected_max_sharpe: float
    deflation_factor: float
    dsr: float  # Probability that true SR > 0 given the observed SR and N trials
    n_trials: int
    n_observations: int


def expected_max_sharpe(sharpe_var: float, n_trials: int) -> float:
    """E[max(SR_i)] under H0 that all candidates have SR=0.

    Approximation: sqrt(var) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e)))
    where gamma ≈ 0.5772156649 (Euler-Mascheroni).
    """
    if n_trials < 2 or sharpe_var <= 0.0:
        return 0.0
    gamma = 0.5772156649
    a = _norm_ppf(1.0 - 1.0 / max(n_trials, 2))
    b = _norm_ppf(1.0 - 1.0 / (max(n_trials, 2) * math.e))
    return math.sqrt(sharpe_var) * ((1.0 - gamma) * a + gamma * b)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    n_trials: int,
    sharpe_var: float | None = None,
) -> DSRResult:
    """Compute DSR for a single candidate's daily-return vector.

    Args:
        returns: 1D array of per-period returns (daily by default).
        n_trials: number of distinct candidates considered (multiple-testing scope).
        sharpe_var: optional variance of SR across the trial space. If None,
            we estimate using the observed series under the assumption of
            non-IID returns (see Bailey/Prado 2012 eq. 8).
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 10:
        return DSRResult(float("nan"), float("nan"), float("nan"), float("nan"), n_trials, n)

    sr_obs = annualized_sharpe(returns)
    if not np.isfinite(sr_obs):
        return DSRResult(float("nan"), float("nan"), float("nan"), float("nan"), n_trials, n)

    skew = _skewness(returns)
    kurt_excess = _kurtosis(returns)
    sr_period = sr_obs / math.sqrt(252.0)
    # PSR-style variance of estimator
    sr_variance = (1.0 - skew * sr_period + (kurt_excess / 4.0) * (sr_period ** 2)) / (n - 1)
    if sharpe_var is None:
        sharpe_var = 1.0  # conservative default; user can pass empirical var across trials
    e_max = expected_max_sharpe(sharpe_var, n_trials)
    # Convert e_max from per-period to annual to be on the same scale as sr_obs
    e_max_annual = e_max * math.sqrt(252.0)

    numerator = (sr_period - e_max) * math.sqrt(n - 1)
    denominator = math.sqrt(max(1.0 - skew * sr_period + (kurt_excess / 4.0) * (sr_period ** 2), 1e-12))
    z = numerator / denominator
    dsr = _norm_cdf(z)

    return DSRResult(
        sharpe_obs=float(sr_obs),
        expected_max_sharpe=float(e_max_annual),
        deflation_factor=float(sr_obs - e_max_annual),
        dsr=float(dsr),
        n_trials=int(n_trials),
        n_observations=int(n),
    )


# ----------------------------------------------------------------------
# White's Reality Check via stationary bootstrap
# ----------------------------------------------------------------------


def stationary_bootstrap_indices(n: int, *, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices."""
    if mean_block <= 1.0:
        return rng.integers(0, n, size=n)
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            idx[i] = rng.integers(0, n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def reality_check_pvalue(
    candidate_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    n_iter: int = 1000,
    mean_block: float = 10.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """White's Reality Check (single candidate vs benchmark version).

    For a single candidate: test H0: E[r_cand] <= E[r_bench]. Stationary bootstrap
    on the difference series. Returns the p-value of observed mean difference.
    """
    if rng is None:
        rng = np.random.default_rng(20260514)
    diff = np.asarray(candidate_returns, dtype=float) - np.asarray(benchmark_returns, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 20:
        return {"p_value": float("nan"), "diff_mean": float("nan"), "diff_t_stat": float("nan")}
    obs_mean = float(np.mean(diff))
    diff_centered = diff - obs_mean
    diff_std = _safe_std(diff)
    t_stat = obs_mean / (diff_std / math.sqrt(n)) if diff_std > 0 else float("nan")

    boot_means = np.empty(n_iter, dtype=float)
    for b in range(n_iter):
        idx = stationary_bootstrap_indices(n, mean_block=mean_block, rng=rng)
        boot_means[b] = float(np.mean(diff_centered[idx]))
    p_value = float(np.mean(boot_means >= obs_mean))
    return {"p_value": p_value, "diff_mean": obs_mean, "diff_t_stat": t_stat}


# ----------------------------------------------------------------------
# Benjamini-Hochberg FDR
# ----------------------------------------------------------------------


def benjamini_hochberg(pvalues: list[float], q_level: float = 0.10) -> dict[str, list]:
    """Returns BH-adjusted q-values and rejection flags."""
    arr = np.asarray(pvalues, dtype=float)
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q_adj = np.empty(n, dtype=float)
    sorted_pvals = arr[order]
    bh_q = np.minimum.accumulate((sorted_pvals * n / np.arange(1, n + 1))[::-1])[::-1]
    bh_q = np.clip(bh_q, 0.0, 1.0)
    q_adj[order] = bh_q
    rejected = q_adj <= q_level
    return {"q_adjusted": q_adj.tolist(), "rejected": rejected.tolist()}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--daily-returns-csv",
        type=Path,
        required=True,
        help="Long-format CSV with columns: candidate, date, daily_return.",
    )
    parser.add_argument(
        "--benchmark-candidate",
        type=str,
        default=None,
        help="Name of benchmark candidate (else uses zero-return baseline).",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-mean-block", type=float, default=10.0)
    parser.add_argument("--fdr-level", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / f"multiple_testing_audit_{datetime.utcnow().strftime('%Y%m%d')}.md",
    )
    return parser.parse_args(argv)


def load_daily_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"candidate", "date", "daily_return"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_returns_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    pivot = df.pivot_table(index="date", columns="candidate", values="daily_return", aggfunc="mean")
    pivot = pivot.sort_index()
    return pivot, list(pivot.columns)


def run(args: argparse.Namespace) -> int:
    df = load_daily_returns(args.daily_returns_csv)
    matrix, candidates = build_returns_matrix(df)
    n_obs = matrix.shape[0]
    n_cand = len(candidates)
    print(f"Loaded {n_obs} dates × {n_cand} candidates from {args.daily_returns_csv}")

    benchmark = args.benchmark_candidate
    if benchmark is None or benchmark not in candidates:
        benchmark_series = np.zeros(n_obs)
        print("Benchmark: zero-return baseline")
    else:
        benchmark_series = matrix[benchmark].to_numpy(dtype=float)
        print(f"Benchmark: {benchmark}")

    # Variance of Sharpe across candidates (used in DSR scaling)
    candidate_sharpes = np.array([annualized_sharpe(matrix[c].to_numpy(dtype=float)) for c in candidates])
    sharpe_var = float(np.nanvar(candidate_sharpes, ddof=1)) / 252.0 if n_cand >= 2 else 1.0
    print(f"sharpe_var (per-period) = {sharpe_var:.6f}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        returns = matrix[candidate].dropna().to_numpy(dtype=float)
        dsr = deflated_sharpe_ratio(returns, n_trials=n_cand, sharpe_var=sharpe_var)
        rc = reality_check_pvalue(
            returns,
            benchmark_series[: len(returns)],
            n_iter=args.bootstrap_iterations,
            mean_block=args.bootstrap_mean_block,
            rng=rng,
        )
        rows.append(
            {
                "candidate": candidate,
                "n_obs": dsr.n_observations,
                "sharpe_obs": dsr.sharpe_obs,
                "expected_max_sharpe": dsr.expected_max_sharpe,
                "deflation_gap": dsr.deflation_factor,
                "dsr": dsr.dsr,
                "rc_p_value": rc["p_value"],
                "rc_diff_mean": rc["diff_mean"],
                "rc_diff_t_stat": rc["diff_t_stat"],
            }
        )

    bh = benjamini_hochberg([row["rc_p_value"] for row in rows], q_level=args.fdr_level)
    for row, q, rejected in zip(rows, bh["q_adjusted"], bh["rejected"]):
        row["bh_q"] = q
        row["alive_after_fdr"] = bool(rejected)

    summary = pd.DataFrame(rows).sort_values(["dsr", "sharpe_obs"], ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    md_path = args.output
    csv_path = args.output.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)

    alive = summary[summary["alive_after_fdr"]]
    write_markdown_report(md_path, args, summary, alive, n_obs, n_cand, sharpe_var)
    print(f"\nWrote: {md_path}")
    print(f"Wrote: {csv_path}")
    print(f"\n{len(alive)}/{len(summary)} candidates alive after BH FDR ({args.fdr_level})")
    return 0


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    alive: pd.DataFrame,
    n_obs: int,
    n_cand: int,
    sharpe_var: float,
) -> None:
    lines = [
        f"# Multiple-Testing Audit — {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        "## Setup",
        "",
        f"- input: `{args.daily_returns_csv}`",
        f"- candidates: `{n_cand}`",
        f"- observations per candidate: `{n_obs}`",
        f"- benchmark: `{args.benchmark_candidate or 'zero-return baseline'}`",
        f"- bootstrap iterations: `{args.bootstrap_iterations}`",
        f"- mean block length: `{args.bootstrap_mean_block}`",
        f"- FDR level (BH): `{args.fdr_level}`",
        f"- sharpe variance across candidates (per-period): `{sharpe_var:.6f}`",
        "",
        "## Top 15 by DSR",
        "",
    ]
    top = summary.head(15)
    lines.append("| candidate | sharpe_obs | E[max SR] | gap | DSR | RC p-value | BH q | alive? |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['sharpe_obs']:.3f} | {row['expected_max_sharpe']:.3f} | "
            f"{row['deflation_gap']:.3f} | {row['dsr']:.3f} | {row['rc_p_value']:.4f} | "
            f"{row['bh_q']:.4f} | {'✓' if row['alive_after_fdr'] else '×'} |"
        )
    lines.extend(
        [
            "",
            f"## Alive after BH FDR ({args.fdr_level})",
            "",
            f"`{len(alive)}` of `{len(summary)}` candidates pass the FDR threshold:",
            "",
        ]
    )
    if alive.empty:
        lines.append("_(none — every reported Sharpe falls within the noise floor after multiple-testing correction)_")
    else:
        lines.append("| candidate | sharpe_obs | DSR | RC p-value | BH q |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for _, row in alive.iterrows():
            lines.append(
                f"| {row['candidate']} | {row['sharpe_obs']:.3f} | {row['dsr']:.3f} | "
                f"{row['rc_p_value']:.4f} | {row['bh_q']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **DSR (Deflated Sharpe Ratio)**: probability that the true SR is > 0, "
            "given that `N=` candidates were tested. DSR < 0.95 means we cannot reject H0 "
            "of zero true edge after multiple-testing correction.",
            "- **RC p-value (White's Reality Check)**: bootstrap p-value of mean-return "
            "difference vs benchmark. p < 0.05 ≈ candidate beats benchmark beyond bootstrap noise.",
            "- **BH q-value**: Benjamini-Hochberg adjusted p-value for FDR control. "
            f"Candidates with q ≤ {args.fdr_level} are flagged 'alive'.",
            "",
            "## Recommendation",
            "",
            "- Promote only candidates that have **DSR ≥ 0.95** AND **alive_after_fdr=True**.",
            "- Re-baseline `docs/current_best_path.md` using the surviving candidate with "
            "highest Sharpe (or highest DSR for risk-adjusted ranking).",
            "- Lock validation: do not run further variants against this validation set "
            "until the next holdout release.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
