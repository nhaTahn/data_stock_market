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
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.config import DEFAULT_DATA_PATH, DEFAULT_FEATURE_COLUMNS, TRAIN_END_DATE, VAL_END_DATE  # noqa: E402
from src.models.training.pipeline import load_frame  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "current_lstm_feature_redundancy_20260520"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "current_lstm_feature_redundancy_20260520"
DEFAULT_DOC = ROOT / "docs" / "current_lstm_feature_redundancy_20260520.md"


SEMANTIC_PRIORITY = {
    "intraday_return": 1.00,
    "gap_open": 0.98,
    "close_position": 0.96,
    "bb_width": 0.95,
    "volume_ratio_20": 0.94,
    "volatility_20": 0.93,
    "momentum_20": 0.92,
    "momentum_5": 0.90,
    "macd_hist": 0.88,
    "rsi_14": 0.86,
    "vwap_gap": 0.84,
    "obv_change": 0.82,
    "ma_200_gap": 0.80,
    "rolling_max_20_gap": 0.78,
    "market_leader_return": 0.76,
    "vnindex_return": 0.74,
    "a_d_ratio": 0.72,
    "sector_momentum_rank": 0.70,
    "is_top_2_sector": 0.68,
    "day_of_week": 0.20,
}


@dataclass(frozen=True)
class FeatureDecision:
    keep: tuple[str, ...]
    drop: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only redundancy and IC diagnostics for the current LSTM feature set.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--train-end-date", default=TRAIN_END_DATE)
    parser.add_argument("--val-end-date", default=VAL_END_DATE)
    parser.add_argument("--target-column", default="target_next_return")
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--corr-threshold", type=float, default=0.75)
    parser.add_argument("--strict-corr-threshold", type=float, default=0.85)
    parser.add_argument("--min-daily-n", type=int, default=20)
    return parser.parse_args(argv)


def parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_FEATURE_COLUMNS
    return tuple(item.strip() for item in value.split(",") if item.strip())


def robust_clean(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    work = frame.loc[:, columns].copy()
    for column in columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work.replace([np.inf, -np.inf], np.nan)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.shape[0] < 30:
        return float("nan")
    if aligned.iloc[:, 0].nunique(dropna=True) < 3 or aligned.iloc[:, 1].nunique(dropna=True) < 3:
        return float("nan")
    value = spearmanr(aligned.iloc[:, 0].to_numpy(dtype=float), aligned.iloc[:, 1].to_numpy(dtype=float)).correlation
    return float(value) if np.isfinite(value) else float("nan")


def feature_quality(train: pd.DataFrame, features: tuple[str, ...], target: str, min_daily_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        daily_ic: list[float] = []
        for _, group in train.loc[:, ["Date", feature, target]].groupby("Date", sort=True):
            clean = group[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if clean.shape[0] < min_daily_n:
                continue
            if clean[feature].nunique(dropna=True) < 3 or clean[target].nunique(dropna=True) < 3:
                continue
            value = spearmanr(clean[feature].to_numpy(dtype=float), clean[target].to_numpy(dtype=float)).correlation
            if np.isfinite(value):
                daily_ic.append(float(value))
        ic_values = np.asarray(daily_ic, dtype=float)
        rows.append(
            {
                "feature": feature,
                "missing_rate": float(train[feature].isna().mean()),
                "std": float(pd.to_numeric(train[feature], errors="coerce").std(skipna=True)),
                "overall_spearman": safe_spearman(train[feature], train[target]),
                "daily_ic_mean": float(np.nanmean(ic_values)) if ic_values.size else float("nan"),
                "daily_ic_std": float(np.nanstd(ic_values, ddof=1)) if ic_values.size > 1 else float("nan"),
                "daily_ic_tstat": (
                    float(np.nanmean(ic_values) / (np.nanstd(ic_values, ddof=1) / np.sqrt(ic_values.size)))
                    if ic_values.size > 2 and np.nanstd(ic_values, ddof=1) > 0
                    else float("nan")
                ),
                "daily_ic_positive_rate": float(np.mean(ic_values > 0.0)) if ic_values.size else float("nan"),
                "daily_ic_n": int(ic_values.size),
                "semantic_priority": float(SEMANTIC_PRIORITY.get(feature, 0.50)),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_daily_ic_mean"] = out["daily_ic_mean"].abs()
    out["abs_overall_spearman"] = out["overall_spearman"].abs()
    out["quality_score"] = (
        out["abs_daily_ic_mean"].fillna(0.0) * 2.0
        + out["abs_overall_spearman"].fillna(0.0)
        + out["semantic_priority"].fillna(0.0) * 0.02
        - out["missing_rate"].fillna(1.0) * 0.02
    )
    return out.sort_values("quality_score", ascending=False).reset_index(drop=True)


def redundant_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cols = corr.columns.tolist()
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                rows.append({"feature_a": left, "feature_b": right, "corr": float(value), "abs_corr": abs(float(value))})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def connected_components(features: tuple[str, ...], pairs: pd.DataFrame) -> list[set[str]]:
    parent = {feature: feature for feature in features}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for _, row in pairs.iterrows():
        union(str(row["feature_a"]), str(row["feature_b"]))
    groups: dict[str, set[str]] = {}
    for feature in features:
        groups.setdefault(find(feature), set()).add(feature)
    return [group for group in groups.values() if len(group) > 1]


def choose_features(features: tuple[str, ...], pairs: pd.DataFrame, quality: pd.DataFrame) -> FeatureDecision:
    quality_map = quality.set_index("feature")["quality_score"].to_dict()
    keep: set[str] = set(features)
    drop: set[str] = set()
    for component in connected_components(features, pairs):
        ordered = sorted(component, key=lambda item: float(quality_map.get(item, -999.0)), reverse=True)
        keep_one = ordered[0]
        for feature in ordered[1:]:
            drop.add(feature)
        keep.add(keep_one)
    return FeatureDecision(
        keep=tuple(feature for feature in features if feature not in drop),
        drop=tuple(feature for feature in features if feature in drop),
    )


def plot_corr_heatmap(corr: pd.DataFrame, output_path: Path, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    image = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.030, pad=0.02)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            if i != j and pd.notna(value) and abs(float(value)) >= threshold:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="black")
    ax.set_title("Current LSTM Features - Train-only Pearson Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    features: tuple[str, ...],
    quality: pd.DataFrame,
    pairs: pd.DataFrame,
    strict_pairs: pd.DataFrame,
    decision: FeatureDecision,
    args: argparse.Namespace,
) -> None:
    top_pairs = pairs.head(12)
    top_quality = quality.head(12)
    lines = [
        "# Current LSTM Feature Redundancy Readout 2026-05-20",
        "",
        "Scope: current default LSTM feature set, train-only diagnostics. Validation/test are not used for feature decisions.",
        "",
        f"- features checked: `{len(features)}`",
        f"- train_end_date: `{args.train_end_date}`",
        f"- target: `{args.target_column}`",
        f"- redundant threshold: `|corr| >= {args.corr_threshold}`",
        f"- strict threshold: `|corr| >= {args.strict_corr_threshold}`",
        "",
        "## Redundant Pairs",
        "",
        f"- pairs >= threshold: `{len(pairs)}`",
        f"- strict pairs: `{len(strict_pairs)}`",
        "",
        "| feature_a | feature_b | corr |",
        "|:--|:--|--:|",
    ]
    if top_pairs.empty:
        lines.append("| n/a | n/a | n/a |")
    else:
        for _, row in top_pairs.iterrows():
            lines.append(f"| `{row.feature_a}` | `{row.feature_b}` | {float(row['corr']):.3f} |")
    lines += [
        "",
        "## Train-Only Signal/Quality Top Features",
        "",
        "| feature | daily IC mean | daily IC t-stat | overall Spearman | missing | quality |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for _, row in top_quality.iterrows():
        lines.append(
            f"| `{row.feature}` | {float(row.daily_ic_mean):.4f} | {float(row.daily_ic_tstat):.2f} | "
            f"{float(row.overall_spearman):.4f} | {100.0 * float(row.missing_rate):.1f}% | {float(row.quality_score):.4f} |"
        )
    lines += [
        "",
        "## Suggested Correlation-Pruned Candidate",
        "",
        f"- keep `{len(decision.keep)}` features",
        f"- drop `{len(decision.drop)}` features",
        "",
        "Drop list:",
        "",
        "```text",
        ",".join(decision.drop) if decision.drop else "(none)",
        "```",
        "",
        "Feature columns for next LSTM probe:",
        "",
        "```text",
        ",".join(decision.keep),
        "```",
        "",
        "## Read",
        "",
        "- This is a diagnostic, not proof of model improvement.",
        "- The pruning rule uses train-only feature correlation and train-only feature-target IC.",
        "- A feature should only be removed permanently if the LSTM probe improves or stays neutral on validation rel_score and daily error stability.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    features = parse_csv(args.feature_columns)
    frame = load_frame(args.data, stocks=None)
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns after pipeline load: {missing}")
    train = frame[pd.to_datetime(frame["Date"]).le(pd.to_datetime(args.train_end_date))].copy()
    train_features = robust_clean(train, tuple([*features, args.target_column]))
    corr = train_features.loc[:, features].corr(method="pearson")
    pairs = redundant_pairs(corr, args.corr_threshold)
    strict_pairs = redundant_pairs(corr, args.strict_corr_threshold)
    quality = feature_quality(pd.concat([train[["Date"]], train_features], axis=1), features, args.target_column, args.min_daily_n)
    decision = choose_features(features, pairs, quality)

    corr.to_csv(args.output_dir / "feature_corr_train_only.csv")
    pairs.to_csv(args.output_dir / "redundant_pairs_train_only.csv", index=False)
    strict_pairs.to_csv(args.output_dir / "strict_redundant_pairs_train_only.csv", index=False)
    quality.to_csv(args.output_dir / "feature_quality_train_only.csv", index=False)
    (args.output_dir / "recommended_pruned_features.json").write_text(
        json.dumps(
            {
                "feature_count_original": len(features),
                "feature_count_pruned": len(decision.keep),
                "keep_features": list(decision.keep),
                "drop_features": list(decision.drop),
                "corr_threshold": args.corr_threshold,
                "strict_corr_threshold": args.strict_corr_threshold,
                "train_end_date": args.train_end_date,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_corr_heatmap(corr, args.gold_dir / "feature_corr_train_only_heatmap.png", args.corr_threshold)
    for name in (
        "feature_corr_train_only.csv",
        "redundant_pairs_train_only.csv",
        "strict_redundant_pairs_train_only.csv",
        "feature_quality_train_only.csv",
        "recommended_pruned_features.json",
    ):
        (args.gold_dir / name).write_bytes((args.output_dir / name).read_bytes())
    write_summary(
        args.output_dir / "summary.md",
        features=features,
        quality=quality,
        pairs=pairs,
        strict_pairs=strict_pairs,
        decision=decision,
        args=args,
    )
    (args.gold_dir / "summary.md").write_text((args.output_dir / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    args.doc_path.write_text((args.output_dir / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
