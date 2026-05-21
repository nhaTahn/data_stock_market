from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_cycle_phase_report import (  # noqa: E402
    CyclePhaseConfig,
    assign_cycle_phases,
    build_market_proxy,
)
from src.models.training.pipeline import load_frame  # noqa: E402
from src.utils.features import ensure_paper_features  # noqa: E402


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "phase_feature_ic"
DEFAULT_ANCHOR_RUN = "broad_signmag_prune_general_sector_full_20260424_r04"
DEFAULT_MODEL = "lstm_signmag_best_by_val"
DEFAULT_SPLITS = ("train", "val")


@dataclass(frozen=True)
class IcConfig:
    min_obs: int = 300
    min_month_obs: int = 30
    min_months: int = 6
    top_k: int = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank feature usefulness by cycle phase using point-in-time IC analysis."
    )
    parser.add_argument("--anchor-run", default=DEFAULT_ANCHOR_RUN, help="Run name used for config and phase proxy.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name used only to build phase proxy from predictions.")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated splits to analyze.")
    parser.add_argument("--stamp", default="20260425_r01", help="Report stamp.")
    parser.add_argument("--output-name", default="current_best_phase_feature_ic", help="Report folder prefix.")
    parser.add_argument("--top-k", type=int, default=IcConfig.top_k, help="Top feature count per split/phase.")
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def spearman_corr(left: pd.Series, right: pd.Series) -> float:
    clean = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return float("nan")
    return float(clean.iloc[:, 0].rank(method="average").corr(clean.iloc[:, 1].rank(method="average")))


def build_analysis_frame(anchor_run: str, model_name: str, splits: set[str]) -> tuple[pd.DataFrame, list[str], dict]:
    run_dir = RUN_ROOT / anchor_run
    config = load_json(run_dir / "reports" / "core" / "config.json")
    data_path = Path(str(config["data_path"]))
    stocks = str(config.get("stocks") or "")
    feature_columns = [str(item) for item in config["feature_columns"]]
    target_column = str(config["target_column"])

    df = load_frame(data_path, stocks)
    if config.get("feature_phase") in {"paper_v1", "paper_denoise_v1"}:
        df = ensure_paper_features(df)

    train_end = pd.Timestamp(str(config["train_end_date"]))
    val_end = pd.Timestamp(str(config["val_end_date"]))
    df["split"] = np.where(df["Date"] <= train_end, "train", np.where(df["Date"] <= val_end, "val", "test"))
    df = df[df["split"].isin(splits)].copy()

    predictions = pd.read_csv(run_dir / "reports" / "core" / "predictions.csv")
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    predictions = predictions[predictions["model"] == model_name].copy()
    phase_df = assign_cycle_phases(build_market_proxy(predictions), CyclePhaseConfig())
    df = df.merge(phase_df[["Date", "phase", "phase_age", "episode_id"]], on="Date", how="left")
    df["phase"] = df["phase"].fillna("unknown")
    usable_columns = [column for column in feature_columns if column in df.columns]
    required = ["code", "Date", "split", "phase", target_column, *usable_columns]
    out = df[required].replace([np.inf, -np.inf], np.nan).copy()
    return out, usable_columns, config


def monthly_ic_stats(group: pd.DataFrame, feature: str, target_column: str, config: IcConfig) -> dict[str, float | int]:
    monthly_values: list[float] = []
    for _, month_df in group.groupby(group["Date"].dt.to_period("M"), sort=True):
        if len(month_df) < config.min_month_obs:
            continue
        value = spearman_corr(month_df[feature], month_df[target_column])
        if np.isfinite(value):
            monthly_values.append(value)

    arr = np.asarray(monthly_values, dtype=float)
    if len(arr) == 0:
        return {
            "monthly_ic_mean": float("nan"),
            "monthly_ic_std": float("nan"),
            "monthly_ic_t_stat": float("nan"),
            "monthly_ic_positive_share": float("nan"),
            "monthly_ic_same_sign_share": float("nan"),
            "monthly_ic_count": 0,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")
    t_stat = float(mean / (std / np.sqrt(len(arr)))) if len(arr) > 1 and std > 0 else float("nan")
    same_sign = float(np.mean(np.sign(arr) == np.sign(mean))) if mean != 0 else float("nan")
    return {
        "monthly_ic_mean": mean,
        "monthly_ic_std": std,
        "monthly_ic_t_stat": t_stat,
        "monthly_ic_positive_share": float(np.mean(arr > 0.0)),
        "monthly_ic_same_sign_share": same_sign,
        "monthly_ic_count": int(len(arr)),
    }


def score_feature(row: dict[str, float | int | str], config: IcConfig) -> float:
    monthly_mean = float(row.get("monthly_ic_mean", float("nan")))
    monthly_std = float(row.get("monthly_ic_std", float("nan")))
    same_sign = float(row.get("monthly_ic_same_sign_share", float("nan")))
    month_count = int(row.get("monthly_ic_count", 0))
    n_obs = int(row.get("n_obs", 0))
    if not np.isfinite(monthly_mean) or month_count < config.min_months or n_obs < config.min_obs:
        return float("nan")
    stability = abs(monthly_mean) / (monthly_std + 1e-6) if np.isfinite(monthly_std) else 0.0
    stability = float(np.clip(stability, 0.0, 2.0) / 2.0)
    consistency = max(0.0, (same_sign - 0.5) * 2.0) if np.isfinite(same_sign) else 0.0
    sample_weight = np.sqrt(month_count)
    return float(abs(monthly_mean) * sample_weight * stability * consistency)


def build_ic_summary(df: pd.DataFrame, feature_columns: list[str], target_column: str, config: IcConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, phase), group in df.groupby(["split", "phase"], sort=True):
        for feature in feature_columns:
            clean = group[["Date", feature, target_column]].dropna()
            n_obs = len(clean)
            row: dict[str, object] = {
                "split": split,
                "phase": phase,
                "feature": feature,
                "n_obs": int(n_obs),
                "n_days": int(clean["Date"].nunique()) if n_obs else 0,
                "ic": spearman_corr(clean[feature], clean[target_column]) if n_obs >= config.min_obs else float("nan"),
            }
            row.update(monthly_ic_stats(clean, feature, target_column, config) if n_obs else monthly_ic_stats(clean, feature, target_column, config))
            row["selection_score"] = score_feature(row, config)
            rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["split", "phase", "selection_score"], ascending=[True, True, False], kind="stable")


def build_top_features(summary: pd.DataFrame, top_k: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in summary.dropna(subset=["selection_score"]).groupby(["split", "phase"], sort=True):
        ranked = group.sort_values("selection_score", ascending=False, kind="stable").head(top_k).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        parts.append(ranked)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_stable_phase_features(summary: pd.DataFrame, top_k: int) -> pd.DataFrame:
    train = summary[summary["split"] == "train"].copy()
    val = summary[summary["split"] == "val"].copy()
    merged = val.merge(
        train,
        on=["phase", "feature"],
        how="inner",
        suffixes=("_val", "_train"),
    )
    if merged.empty:
        return merged

    val_sign = np.sign(merged["monthly_ic_mean_val"].to_numpy(dtype=float))
    train_sign = np.sign(merged["monthly_ic_mean_train"].to_numpy(dtype=float))
    merged["same_train_val_sign"] = val_sign == train_sign
    merged["train_t_weight"] = np.clip(np.abs(merged["monthly_ic_t_stat_train"].to_numpy(dtype=float)) / 2.0, 0.0, 1.0)
    merged["val_t_weight"] = np.clip(np.abs(merged["monthly_ic_t_stat_val"].to_numpy(dtype=float)) / 2.0, 0.0, 1.0)
    merged["stable_score"] = np.where(
        merged["same_train_val_sign"],
        merged["selection_score_val"].to_numpy(dtype=float) * merged["train_t_weight"] * merged["val_t_weight"],
        np.nan,
    )
    merged["direction"] = np.where(merged["monthly_ic_mean_val"] >= 0.0, "positive", "negative")

    parts: list[pd.DataFrame] = []
    for _, group in merged.dropna(subset=["stable_score"]).groupby("phase", sort=True):
        ranked = group.sort_values("stable_score", ascending=False, kind="stable").head(top_k).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        parts.append(ranked)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def save_heatmap(summary: pd.DataFrame, output_dir: Path, split: str) -> None:
    split_df = summary[(summary["split"] == split) & summary["selection_score"].notna()].copy()
    if split_df.empty:
        return
    top_features = (
        split_df.groupby("feature")["selection_score"]
        .max()
        .sort_values(ascending=False)
        .head(18)
        .index
        .tolist()
    )
    pivot = (
        split_df[split_df["feature"].isin(top_features)]
        .pivot_table(index="feature", columns="phase", values="monthly_ic_mean", aggfunc="first")
        .reindex(top_features)
    )
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(pivot))))
    image = ax.imshow(pivot.fillna(0.0).to_numpy(dtype=float), cmap="RdBu", vmin=-0.08, vmax=0.08, aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_title(f"Monthly IC mean by phase | {split}")
    fig.colorbar(image, ax=ax, label="monthly IC mean")
    fig.tight_layout()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"phase_feature_ic_heatmap_{split}.png", dpi=180)
    plt.close(fig)


def write_markdown(
    output_dir: Path,
    top_features: pd.DataFrame,
    stable_features: pd.DataFrame,
    summary: pd.DataFrame,
    config: IcConfig,
    anchor_run: str,
) -> None:
    lines = [
        "# Phase Feature IC Report",
        "",
        f"Anchor run: `{anchor_run}`.",
        "",
        "Scope: train/validation only. No test/out-sample data is used.",
        "",
        "IC is Spearman correlation between feature value at date `t` and `target_next_return`; phase labels are point-in-time cycle labels from the current cycle phase detector.",
        "",
        f"Selection score requires at least `{config.min_obs}` observations and `{config.min_months}` monthly IC samples.",
        "",
        "## Top Validation Features By Phase",
        "",
        "| Phase | Rank | Feature | Score | IC | Monthly IC | t-stat | Same-sign | Obs |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    val_top = top_features[top_features["split"] == "val"].copy()
    for _, row in val_top.sort_values(["phase", "rank"], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['phase']}` | {int(row['rank'])} | `{row['feature']}` | "
            f"{float(row['selection_score']):.5f} | {float(row['ic']):+.4f} | "
            f"{float(row['monthly_ic_mean']):+.4f} | {float(row['monthly_ic_t_stat']):+.2f} | "
            f"{float(row['monthly_ic_same_sign_share']):.1%} | {int(row['n_obs'])} |"
        )

    lines.extend(
        [
            "",
            "## Stable Features: Train And Validation Same IC Sign",
            "",
            "| Phase | Rank | Feature | Direction | Stable score | Val monthly IC | Train monthly IC | Val t-stat | Train t-stat |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if stable_features.empty:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    else:
        for _, row in stable_features.sort_values(["phase", "rank"], kind="stable").iterrows():
            lines.append(
                "| "
                f"`{row['phase']}` | {int(row['rank'])} | `{row['feature']}` | `{row['direction']}` | "
                f"{float(row['stable_score']):.5f} | {float(row['monthly_ic_mean_val']):+.4f} | "
                f"{float(row['monthly_ic_mean_train']):+.4f} | {float(row['monthly_ic_t_stat_val']):+.2f} | "
                f"{float(row['monthly_ic_t_stat_train']):+.2f} |"
            )

    lines.extend(
        [
            "",
            "## Phase Coverage In IC Frame",
            "",
            "| Split | Phase | Obs | Days | Features with score |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    coverage = (
        summary.groupby(["split", "phase"], as_index=False)
        .agg(
            n_obs=("n_obs", "max"),
            n_days=("n_days", "max"),
            scored_features=("selection_score", lambda values: int(pd.Series(values).notna().sum())),
        )
        .sort_values(["split", "phase"], kind="stable")
    )
    for _, row in coverage.iterrows():
        lines.append(
            f"| `{row['split']}` | `{row['phase']}` | {int(row['n_obs'])} | {int(row['n_days'])} | {int(row['scored_features'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Prefer features that rank in validation and have the same sign as train; this report does not yet enforce that automatically.",
            "- The stable table enforces same train/validation IC sign and is the safer source for the next feature-set batch.",
            "- Treat calendar-only features such as `day_of_week` as diagnostic until they survive a separate ablation; they are easy to overfit.",
            "- Do not pick phase-specific features from phases with too few months or mostly transition days.",
            "- Changing the phase detector thresholds counts as a new hypothesis test.",
        ]
    )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ic_config = IcConfig(top_k=int(args.top_k))
    splits = set(split_csv(args.splits))
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, feature_columns, run_config = build_analysis_frame(args.anchor_run, args.model, splits)
    summary = build_ic_summary(frame, feature_columns, str(run_config["target_column"]), ic_config)
    top_features = build_top_features(summary, ic_config.top_k)
    stable_features = build_stable_phase_features(summary, ic_config.top_k)

    frame.to_csv(output_dir / "feature_frame_with_phase.csv", index=False)
    summary.to_csv(output_dir / "feature_ic_by_phase.csv", index=False)
    top_features.to_csv(output_dir / "top_features_by_phase.csv", index=False)
    stable_features.to_csv(output_dir / "stable_top_features_by_phase.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "anchor_run": args.anchor_run,
                "model": args.model,
                "splits": sorted(splits),
                "config": ic_config.__dict__,
                "feature_columns": feature_columns,
                "top_features_by_phase": top_features.to_dict(orient="records"),
                "stable_top_features_by_phase": stable_features.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    for split in sorted(splits):
        save_heatmap(summary, output_dir, split)
    write_markdown(output_dir, top_features, stable_features, summary, ic_config, args.anchor_run)
    print(json.dumps({"output_dir": str(output_dir), "features": len(feature_columns)}, indent=2))


if __name__ == "__main__":
    main()
