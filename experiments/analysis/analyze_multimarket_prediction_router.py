from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metric import directional_accuracy as metric_directional_accuracy  # noqa: E402
from src.evaluation.metric import evaluate  # noqa: E402


GLOBAL_RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_global" / "history" / "training_runs"
REPORT_ROOT = GLOBAL_RUN_ROOT / "reports" / "multimarket_router"
DEFAULT_CANDIDATES = (
    "marketplus_signmag_top2="
    "multimarket_portable_marketplus_20260504_r01:lstm_signmag_top2_by_val",
    "marketplus_signmag_ensemble="
    "multimarket_portable_marketplus_20260504_r01:lstm_signmag_ensemble",
    "compact_lstm_top2="
    "multimarket_portable_compact_signal_20260504_compactfix_r01:lstm_top2_by_val",
    "compact_signmag_top2="
    "multimarket_portable_compact_signal_20260504_compactfix_r01:lstm_signmag_top2_by_val",
    "signal_lstm_top2="
    "multimarket_portable_marketplus_signal_focus_20260504_signalfocus_r01:lstm_top2_by_val",
    "signal_signmag_top2="
    "multimarket_portable_marketplus_signal_focus_20260504_signalfocus_r01:lstm_signmag_top2_by_val",
    "signal_signmag_ensemble="
    "multimarket_portable_marketplus_signal_focus_20260504_signalfocus_r01:lstm_signmag_ensemble",
)
MARKETS = ("VN", "JP", "US")


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    run_dir: Path
    model: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate train-selected per-market routers across multi-market prediction runs."
    )
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_r%H%M"))
    parser.add_argument("--output-name", default="market_signal_router")
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Candidate as label=run_dir:model. Relative run_dir is resolved under global training_runs.",
    )
    return parser.parse_args(argv)


def parse_candidate(value: str) -> CandidateSpec:
    if "=" not in value or ":" not in value:
        raise ValueError(f"Invalid candidate '{value}'. Expected label=run_dir:model.")
    label, target = value.split("=", 1)
    run_name, model = target.rsplit(":", 1)
    run_dir = Path(run_name)
    if not run_dir.is_absolute():
        run_dir = GLOBAL_RUN_ROOT / run_dir
    return CandidateSpec(label=label, run_dir=run_dir, model=model)


def load_candidate(spec: CandidateSpec) -> pd.DataFrame:
    path = spec.run_dir / "reports" / "core" / "predictions.csv"
    df = pd.read_csv(path)
    df = df[(df["model"] == spec.model) & (df["split"].isin({"train", "val"}))].copy()
    if df.empty:
        raise ValueError(f"No predictions for {spec.label}: {spec.run_dir.name}:{spec.model}")
    df["Date"] = pd.to_datetime(df["Date"])
    df["market"] = df["code"].astype(str).str.split(":", n=1).str[0].where(
        df["code"].astype(str).str.contains(":", regex=False),
        "UNKNOWN",
    )
    df["candidate"] = spec.label
    return df[["candidate", "split", "market", "Date", "code", "actual", "prediction"]].dropna(
        subset=["actual", "prediction"]
    )


def rel_score_for_frame(df: pd.DataFrame) -> float:
    work = df.sort_values(["code", "Date"], kind="stable")
    result = evaluate(
        work["prediction"].to_numpy(dtype=float),
        work["actual"].to_numpy(dtype=float),
        group_ids=work["code"].astype(str).to_numpy(),
    )
    return float(result["rel_score"])


def directional_accuracy(df: pd.DataFrame) -> float:
    work = df.sort_values(["code", "Date"], kind="stable")
    return metric_directional_accuracy(
        work["prediction"].to_numpy(dtype=float),
        work["actual"].to_numpy(dtype=float),
        group_ids=work["code"].astype(str).to_numpy(),
    )


def daily_rank_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    daily_ic: list[float] = []
    daily_ls: list[float] = []
    for _, group in df.groupby("Date", sort=True):
        if len(group) < 10 or group["prediction"].nunique() < 3 or group["actual"].nunique() < 3:
            continue
        daily_ic.append(float(group["prediction"].corr(group["actual"], method="spearman")))
        rank_pct = group["prediction"].rank(method="first", pct=True)
        long_return = group.loc[rank_pct >= 0.75, "actual"].mean()
        short_return = group.loc[rank_pct <= 0.25, "actual"].mean()
        if pd.notna(long_return) and pd.notna(short_return):
            daily_ls.append(float(long_return - short_return))
    ic = pd.Series(daily_ic, dtype=float).dropna()
    long_short = pd.Series(daily_ls, dtype=float).dropna()
    return {
        "daily_ic": float(ic.mean()) if len(ic) else float("nan"),
        "daily_ic_t": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))
        if len(ic) > 1 and ic.std(ddof=1) > 0
        else float("nan"),
        "positive_ic_days": float((ic > 0.0).mean()) if len(ic) else float("nan"),
        "quartile_ls_mean": float(long_short.mean()) if len(long_short) else float("nan"),
        "quartile_equity": float((1.0 + long_short).prod()) if len(long_short) else float("nan"),
        "days": int(len(ic)),
    }


def summarize_frame(name: str, df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (split, market), group in df.groupby(["split", "market"], sort=True):
        metrics = daily_rank_metrics(group)
        rows.append(
            {
                "candidate": name,
                "split": split,
                "market": market,
                "rows": int(len(group)),
                "panels": int(group["code"].nunique()),
                "rel_score": rel_score_for_frame(group),
                "directional_accuracy": directional_accuracy(group),
                **metrics,
            }
        )
    for split, group in df.groupby("split", sort=True):
        rows.append(
            {
                "candidate": name,
                "split": split,
                "market": "ALL",
                "rows": int(len(group)),
                "panels": int(group["code"].nunique()),
                "rel_score": rel_score_for_frame(group),
                "directional_accuracy": directional_accuracy(group),
                **daily_rank_metrics(group),
            }
        )
    return rows


def build_router_frames(base: pd.DataFrame, train_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows: list[dict[str, object]] = []
    router_frames: list[pd.DataFrame] = []
    selectors = {
        "router_train_rel_score": "rel_score",
        "router_train_daily_ic": "daily_ic",
        "router_train_quartile_equity": "quartile_equity",
    }
    train_market = train_summary[(train_summary["split"] == "train") & (train_summary["market"].isin(MARKETS))]
    for router_name, metric in selectors.items():
        parts: list[pd.DataFrame] = []
        for market in MARKETS:
            market_rows = train_market[train_market["market"] == market].dropna(subset=[metric])
            if market_rows.empty:
                continue
            best = market_rows.sort_values(metric, ascending=False, kind="stable").iloc[0]
            selected_candidate = str(best["candidate"])
            selection_rows.append(
                {
                    "router": router_name,
                    "market": market,
                    "selector_metric": metric,
                    "selected_candidate": selected_candidate,
                    "train_metric_value": float(best[metric]),
                }
            )
            part = base[(base["candidate"] == selected_candidate) & (base["market"] == market)].copy()
            part["candidate"] = router_name
            parts.append(part)
        if parts:
            router_frames.append(pd.concat(parts, ignore_index=True))
    selection = pd.DataFrame(selection_rows)
    routers = pd.concat(router_frames, ignore_index=True) if router_frames else pd.DataFrame()
    return routers, selection


def write_markdown(output_dir: Path, summary: pd.DataFrame, selection: pd.DataFrame) -> None:
    val_all = summary[(summary["split"] == "val") & (summary["market"] == "ALL")].sort_values(
        "rel_score",
        ascending=False,
        kind="stable",
    )
    val_market = summary[(summary["split"] == "val") & (summary["market"].isin(MARKETS))]
    lines = [
        "# Multi-Market Prediction Router",
        "",
        "Selection uses train split only. Validation metrics are reported for comparison and do not drive the routers.",
        "",
        "## Validation Overall",
        "",
        "| Candidate | rel_score | Direction | Daily IC | IC t | Quartile equity | Panels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in val_all.iterrows():
        lines.append(
            f"| `{row['candidate']}` | {float(row['rel_score']):+.5f} | "
            f"{float(row['directional_accuracy']):.2%} | {float(row['daily_ic']):+.4f} | "
            f"{float(row['daily_ic_t']):+.2f} | {float(row['quartile_equity']):.3f} | {int(row['panels'])} |"
        )
    lines.extend(["", "## Train-Selected Router Choices", ""])
    for router_name, group in selection.groupby("router", sort=True):
        choices = ", ".join(
            f"{row.market}: {row.selected_candidate}" for row in group.sort_values("market").itertuples()
        )
        lines.append(f"- `{router_name}`: {choices}")
    lines.extend(["", "## Best Validation By Market", ""])
    for market in MARKETS:
        rows = val_market[val_market["market"] == market].sort_values("daily_ic", ascending=False, kind="stable").head(5)
        lines.append(f"### {market}")
        for _, row in rows.iterrows():
            lines.append(
                f"- `{row['candidate']}`: rel_score `{float(row['rel_score']):+.5f}`, "
                f"daily IC `{float(row['daily_ic']):+.4f}`, quartile equity `{float(row['quartile_equity']):.3f}`"
            )
        lines.append("")
    output_dir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    candidate_values = args.candidate or list(DEFAULT_CANDIDATES)
    specs = [parse_candidate(value) for value in candidate_values]
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.concat([load_candidate(spec) for spec in specs], ignore_index=True)
    individual_summary = pd.DataFrame(
        row for candidate, group in base.groupby("candidate", sort=True) for row in summarize_frame(str(candidate), group)
    )
    routers, selection = build_router_frames(base, individual_summary)
    all_predictions = pd.concat([base, routers], ignore_index=True) if not routers.empty else base
    summary = pd.DataFrame(
        row
        for candidate, group in all_predictions.groupby("candidate", sort=True)
        for row in summarize_frame(str(candidate), group)
    )

    base.to_csv(output_dir / "candidate_predictions.csv", index=False)
    if not routers.empty:
        routers.to_csv(output_dir / "router_predictions.csv", index=False)
    selection.to_csv(output_dir / "router_selection.csv", index=False)
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    manifest = {
        "output_dir": str(output_dir),
        "candidates": [
            {"label": spec.label, "run_dir": str(spec.run_dir), "model": spec.model}
            for spec in specs
        ],
    }
    output_dir.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(output_dir, summary, selection)
    print(json.dumps({"output_dir": str(output_dir), "candidates": len(specs), "routers": selection["router"].nunique()}, indent=2))


if __name__ == "__main__":
    main()
