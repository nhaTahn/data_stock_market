from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_prediction_router import add_candidate_predictions  # noqa: E402
from experiments.analysis.analyze_regime_performance import (  # noqa: E402
    build_daily_quartile_returns,
    build_regime_filter_summary,
    rel_score,
)
from experiments.analysis.analyze_router_rolling_validation import DEFAULT_ROUTER_REPORT  # noqa: E402


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "downtrend_sidecar"
DEFAULT_DOWNTREND_RUN = RUN_ROOT / "downtrend_expert_phase_ic_sector19" / "reports" / "core" / "predictions.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate downtrend-only sidecars against the current anchor/router.")
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--downtrend-predictions", type=Path, default=DEFAULT_DOWNTREND_RUN)
    parser.add_argument("--stamp", default="20260427_r01")
    parser.add_argument("--output-name", default="anchor_downtrend_sidecar")
    parser.add_argument("--min-names-per-day", type=int, default=8)
    return parser.parse_args(argv)


def load_router_frame(router_report: Path) -> pd.DataFrame:
    df = pd.read_csv(router_report / "candidate_predictions.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["actual_date"] = pd.to_datetime(df["actual_date"])
    if not any(column.startswith("candidate__") for column in df.columns):
        df = add_candidate_predictions(df)
    return df[df["split"].isin({"train", "val"})].copy()


def load_sidecar_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[df["split"].isin({"train", "val"})].copy()


def add_sidecar_candidates(router_df: pd.DataFrame, sidecar_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = router_df.copy()
    coverage_rows: list[dict[str, object]] = []
    base_cols = ["code", "split", "Date", "model", "prediction"]
    sidecar_models = [
        "linear_regression",
        "lstm_seed_42",
        "lstm_ensemble",
        "lstm_best_by_val",
    ]
    for model in sidecar_models:
        model_df = sidecar_df[sidecar_df["model"] == model][base_cols].rename(
            columns={"Date": "signal_date", "prediction": f"sidecar__{model}"}
        )
        out = out.merge(model_df.drop(columns=["model"]), on=["code", "split", "signal_date"], how="left")
        sidecar_pred = out[f"sidecar__{model}"].to_numpy(dtype=float)
        has_pred = np.isfinite(sidecar_pred)
        is_downtrend = out["regime"].astype(str).eq("downtrend").to_numpy()
        use_sidecar = is_downtrend & has_pred

        clean_name = model.replace("linear_regression", "linear").replace("lstm_", "lstm_")
        out[f"candidate__downtrend_{clean_name}_else_anchor"] = np.where(
            use_sidecar,
            sidecar_pred,
            out["candidate__anchor"].to_numpy(dtype=float),
        )
        out[f"candidate__downtrend_{clean_name}_up_sector19_else_anchor"] = np.where(
            use_sidecar,
            sidecar_pred,
            out["candidate__sector19_up_anchor_else"].to_numpy(dtype=float),
        )
        for split, group in out.groupby("split", sort=True):
            downtrend = group[group["regime"] == "downtrend"]
            coverage_rows.append(
                {
                    "model": model,
                    "split": split,
                    "downtrend_rows": int(len(downtrend)),
                    "covered_downtrend_rows": int(downtrend[f"sidecar__{model}"].notna().sum()),
                    "coverage": float(downtrend[f"sidecar__{model}"].notna().mean()) if len(downtrend) else float("nan"),
                }
            )
    return out, pd.DataFrame(coverage_rows)


def candidate_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    keep_candidates = [
        "candidate__anchor",
        "candidate__challenger",
        "candidate__sector19_down_anchor_else",
        "candidate__sector19_down_up_anchor_else",
        "candidate__avg_70_challenger",
    ]
    keep_candidates.extend(column for column in df.columns if column.startswith("candidate__downtrend_"))

    frames: list[pd.DataFrame] = []
    for column in keep_candidates:
        candidate = column.removeprefix("candidate__")
        part = df[["code", "split", "signal_date", "actual_date", "actual", "regime", column]].copy()
        part = part.rename(columns={column: "prediction"})
        part["run_name"] = candidate
        part["model"] = candidate
        part["candidate"] = candidate
        part["error"] = part["actual"] - part["prediction"]
        part["year"] = part["actual_date"].dt.year
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def prediction_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, split), group in long_df.groupby(["candidate", "split"], sort=True):
        rows.append(
            {
                "candidate": candidate,
                "split": split,
                "n_obs": int(len(group)),
                "rel_score": rel_score(group["error"], group["actual"]),
                "directional_accuracy": float((np.sign(group["prediction"]) == np.sign(group["actual"])).mean()),
                "error_q2": float(group["error"].quantile(0.2)),
                "error_q8": float(group["error"].quantile(0.8)),
            }
        )
    return pd.DataFrame(rows)


def daily_ic(long_df: pd.DataFrame, min_names_per_day: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["candidate", "split", "actual_date", "regime", "year"]
    for keys, group in long_df.groupby(group_cols, sort=True):
        if len(group) < min_names_per_day:
            continue
        ic = group["prediction"].corr(group["actual"], method="spearman")
        if not np.isfinite(ic):
            continue
        rows.append({**dict(zip(group_cols, keys, strict=True)), "ic": float(ic), "name_count": int(len(group))})
    return pd.DataFrame(rows)


def summarize_ic(daily: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in daily.groupby(group_cols, sort=True):
        values = group["ic"].to_numpy(dtype=float)
        mean = float(np.mean(values)) if len(values) else float("nan")
        std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        t_stat = mean / (std / np.sqrt(len(values))) if len(values) > 1 and std > 0 else float("nan")
        rows.append(
            {
                **dict(zip(group_cols, keys, strict=True)),
                "days": int(len(values)),
                "mean_ic": mean,
                "ic_std": std,
                "t_stat": float(t_stat),
                "pct_positive": float(np.mean(values > 0.0)) if len(values) else float("nan"),
                "avg_names": float(group["name_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_plot(output_dir: Path, merged: pd.DataFrame) -> None:
    val = merged.sort_values("downtrend_mean_ic", ascending=True, kind="stable")
    fig, axis = plt.subplots(figsize=(11, 6))
    colors = ["#386641" if value > 0 else "#bc4749" for value in val["downtrend_mean_ic"]]
    axis.barh(val["candidate"], val["downtrend_mean_ic"], color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Validation downtrend mean daily IC")
    axis.set_title("Downtrend Sidecar Candidate IC")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "downtrend_sidecar_ic.png", dpi=160)
    plt.close(fig)


def write_markdown(
    output_dir: Path,
    merged: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    lines = [
        "# Downtrend Sidecar Analysis",
        "",
        "Scope: train/validation predictions only. No test/out-sample data is used.",
        "",
        "Sidecar predictions are used only when router regime is `downtrend`; otherwise candidates fall back to anchor/router rules.",
        "",
        "![Downtrend sidecar IC](downtrend_sidecar_ic.png)",
        "",
        "## Validation Ranking",
        "",
        "| Candidate | rel_score | All-regime equity | Downtrend IC | Downtrend t-stat | Downtrend positive days |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in merged.sort_values("downtrend_mean_ic", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {float(row['rel_score']):+.4f} | {float(row['final_equity']):.3f} | "
            f"{float(row['downtrend_mean_ic']):+.4f} | {float(row['downtrend_t_stat']):+.2f} | {float(row['downtrend_pct_positive']):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Sidecar Coverage",
            "",
            "| Sidecar model | Split | Downtrend rows | Covered rows | Coverage |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in coverage.iterrows():
        lines.append(
            "| "
            f"`{row['model']}` | `{row['split']}` | {int(row['downtrend_rows'])} | "
            f"{int(row['covered_downtrend_rows'])} | {float(row['coverage']):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Read",
            "",
            "- If a sidecar does not beat `sector19_down_anchor_else` on downtrend IC and all-regime equity, it is not worth keeping.",
            "- Linear sidecar is the main cheap test because it beat LSTM inside the hard-filter downtrend run.",
            "- Do not use out-sample for this selection.",
        ]
    )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    router = load_router_frame(args.router_report)
    sidecar = load_sidecar_predictions(args.downtrend_predictions)
    candidate_df, coverage = add_sidecar_candidates(router, sidecar)
    long_df = candidate_long_frame(candidate_df)
    pred_summary = prediction_summary(long_df)
    daily_quartile = build_daily_quartile_returns(long_df)
    filter_summary = build_regime_filter_summary(daily_quartile)
    ic_daily = daily_ic(long_df, args.min_names_per_day)
    ic_by_regime = summarize_ic(ic_daily, ["candidate", "split", "regime"])

    val_pred = pred_summary[pred_summary["split"] == "val"].copy()
    val_trade = filter_summary[(filter_summary["split"] == "val") & (filter_summary["filter_name"] == "all_regimes")].copy()
    val_down_ic = ic_by_regime[(ic_by_regime["split"] == "val") & (ic_by_regime["regime"] == "downtrend")].copy()
    merged = (
        val_pred.merge(val_trade[["run_name", "final_equity"]], left_on="candidate", right_on="run_name", how="left")
        .merge(
            val_down_ic[["candidate", "mean_ic", "t_stat", "pct_positive", "days"]],
            on="candidate",
            how="left",
        )
        .rename(
            columns={
                "mean_ic": "downtrend_mean_ic",
                "t_stat": "downtrend_t_stat",
                "pct_positive": "downtrend_pct_positive",
                "days": "downtrend_days",
            }
        )
    )

    pred_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    filter_summary.to_csv(output_dir / "regime_filter_summary.csv", index=False)
    ic_by_regime.to_csv(output_dir / "ic_by_regime.csv", index=False)
    coverage.to_csv(output_dir / "sidecar_coverage.csv", index=False)
    merged.to_csv(output_dir / "validation_summary.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "router_report": str(args.router_report),
                "downtrend_predictions": str(args.downtrend_predictions),
                "validation_summary": merged.to_dict(orient="records"),
                "coverage": coverage.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_plot(output_dir, merged)
    write_markdown(output_dir, merged, coverage)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
