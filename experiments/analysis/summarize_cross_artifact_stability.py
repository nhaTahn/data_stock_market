from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
FILTER_ROOT = RUN_ROOT / "reports" / "filter_signal"
DEFAULT_RUNS = (
    "portable_lstm_filter_signal_20260509_r06_selector_module",
    "portable_lstm_filter_signal_20260508_r05_signmag",
)
DEFAULT_COMMITTEE_OUTPUT = "committee_hypothesis_grid_val_split_t025_dd15"
DEFAULT_GATE_OUTPUT = "wyckoff_phase_gate_execution_vn_grid"
DEFAULT_OUTPUT_NAME = "cross_artifact_stability_20260509_r01"


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize committee and gate stability across VN filter-signal artifacts."
    )
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--committee-output", default=DEFAULT_COMMITTEE_OUTPUT)
    parser.add_argument("--gate-output", default=DEFAULT_GATE_OUTPUT)
    parser.add_argument("--gate-outputs", default="")
    parser.add_argument("--max-worst-drawdown", type=float, default=0.25)
    parser.add_argument("--max-avg-turnover", type=float, default=0.20)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return parser.parse_args(argv)


def load_committee_summary(run: str, output_name: str) -> pd.DataFrame:
    path = FILTER_ROOT / run / output_name / "committee_hypothesis_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "artifact": run,
            "kind": "committee",
            "policy": frame["hypothesis"].astype(str),
            "net_equity": frame["stitched_net_equity"].astype(float),
            "net_sharpe": frame["stitched_net_sharpe"].astype(float),
            "max_drawdown": frame["stitched_max_drawdown"].astype(float),
            "avg_turnover": frame["avg_turnover"].astype(float),
            "positive_fold_rate": frame["positive_fold_rate"].astype(float),
            "relaxed_folds": frame["relaxed_folds"].astype(int),
            "is_oracle": False,
        }
    )


def load_gate_summary(run: str, output_name: str) -> pd.DataFrame:
    path = FILTER_ROOT / run / output_name / "wyckoff_phase_gate_execution_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "artifact": run,
            "source_output": output_name,
            "kind": "gate",
            "policy": frame["gate"].astype(str),
            "net_equity": frame["net_equity"].astype(float),
            "net_sharpe": frame["net_sharpe"].astype(float),
            "max_drawdown": frame["max_drawdown"].astype(float),
            "avg_turnover": frame["avg_turnover"].astype(float),
            "positive_fold_rate": float("nan"),
            "relaxed_folds": float("nan"),
            "is_oracle": frame["gate"].astype(str).str.contains("oracle", case=False, regex=False),
        }
    )


def summarize_stability(
    frame: pd.DataFrame,
    expected_artifacts: int,
    *,
    max_worst_drawdown: float,
    max_avg_turnover: float,
) -> pd.DataFrame:
    grouped = frame.groupby(["kind", "policy"], sort=False, dropna=False)
    summary = grouped.agg(
        artifacts=("artifact", "nunique"),
        mean_net_equity=("net_equity", "mean"),
        min_net_equity=("net_equity", "min"),
        mean_net_sharpe=("net_sharpe", "mean"),
        min_net_sharpe=("net_sharpe", "min"),
        worst_max_drawdown=("max_drawdown", "min"),
        max_avg_turnover=("avg_turnover", "max"),
        positive_artifacts=("net_equity", lambda values: int((values > 1.0).sum())),
        any_oracle=("is_oracle", "max"),
    ).reset_index()
    summary["passes_all_artifacts"] = (
        (summary["artifacts"] == expected_artifacts)
        & (summary["positive_artifacts"] == expected_artifacts)
        & (~summary["any_oracle"].astype(bool))
    )
    summary["passes_risk_controls"] = (
        summary["passes_all_artifacts"]
        & (summary["worst_max_drawdown"] >= -abs(max_worst_drawdown))
        & (summary["max_avg_turnover"] <= max_avg_turnover)
    )
    summary["stability_score"] = (
        summary["min_net_equity"]
        + 0.25 * summary["min_net_sharpe"].clip(lower=-2.0, upper=2.0)
        + summary["worst_max_drawdown"].clip(lower=-1.0, upper=0.0)
        - 0.25 * summary["max_avg_turnover"].clip(lower=0.0, upper=1.0)
    )
    return summary.sort_values(
        ["passes_risk_controls", "passes_all_artifacts", "min_net_equity", "mean_net_equity"],
        ascending=[False, False, False, False],
        kind="stable",
    )


def write_markdown(
    output_dir: Path,
    summary: pd.DataFrame,
    details: pd.DataFrame,
    *,
    max_worst_drawdown: float,
    max_avg_turnover: float,
) -> None:
    lines = [
        "# Cross-Artifact Stability Summary",
        "",
        "This report compares committee hypotheses and execution-level Wyckoff gates across VN filter-signal artifacts. Holdout/test data is not used.",
        f"Risk-control pass requires worst-artifact max drawdown no worse than `{max_worst_drawdown:.0%}` and max average turnover no higher than `{max_avg_turnover:.2f}`.",
        "",
        "## Risk-Controlled Stable Policies",
        "",
        "| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    risk_controlled = summary.loc[summary["passes_risk_controls"]].copy()
    if risk_controlled.empty:
        lines.append("| - | - | 0 | - | - | - | - | - | No policy passed the risk-control screen. |")
    else:
        for _, row in risk_controlled.iterrows():
            lines.append(
                "| "
                f"{row['kind']} | `{row['policy']}` | {int(row['artifacts'])} | "
                f"{float(row['min_net_equity']):.3f} | {float(row['mean_net_equity']):.3f} | "
                f"{float(row['min_net_sharpe']):+.2f} | {float(row['worst_max_drawdown']):.1%} | "
                f"{float(row['max_avg_turnover']):.2f} | Passes both artifacts and the risk-control screen. |"
            )
    lines.extend(
        [
        "",
        "## Stable Non-Oracle Policies",
        "",
        "| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    stable = summary.loc[summary["passes_all_artifacts"]].copy()
    if stable.empty:
        lines.append("| - | - | 0 | - | - | - | - | - | No policy passed all artifacts. |")
    else:
        for _, row in stable.iterrows():
            lines.append(
                "| "
                f"{row['kind']} | `{row['policy']}` | {int(row['artifacts'])} | "
                f"{float(row['min_net_equity']):.3f} | {float(row['mean_net_equity']):.3f} | "
                f"{float(row['min_net_sharpe']):+.2f} | {float(row['worst_max_drawdown']):.1%} | "
                f"{float(row['max_avg_turnover']):.2f} | Passes both artifacts, but still needs stricter validation. |"
            )
    lines.extend(
        [
            "",
            "## Top Policies By Worst Artifact",
            "",
            "| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Positive artifacts | Oracle |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in summary.head(12).iterrows():
        lines.append(
            "| "
            f"{row['kind']} | `{row['policy']}` | {float(row['min_net_equity']):.3f} | "
            f"{float(row['mean_net_equity']):.3f} | {float(row['min_net_sharpe']):+.2f} | "
            f"{float(row['worst_max_drawdown']):.1%} | {float(row['max_avg_turnover']):.2f} | "
            f"{int(row['positive_artifacts'])} | {bool(row['any_oracle'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Details",
            "",
            "| Artifact | Kind | Policy | Net equity | Sharpe | Max DD | Avg turnover |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in details.sort_values(["artifact", "kind", "net_equity"], ascending=[True, True, False]).iterrows():
        lines.append(
            "| "
            f"`{row['artifact']}` | {row['kind']} | `{row['policy']}` | "
            f"{float(row['net_equity']):.3f} | {float(row['net_sharpe']):+.2f} | "
            f"{float(row['max_drawdown']):.1%} | {float(row['avg_turnover']):.2f} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs = parse_csv_list(args.runs)
    if not runs:
        raise ValueError("At least one run is required.")

    parts: list[pd.DataFrame] = []
    gate_outputs = parse_csv_list(args.gate_outputs) or (args.gate_output,)
    for run in runs:
        parts.append(load_committee_summary(run, args.committee_output))
        for gate_output in gate_outputs:
            parts.append(load_gate_summary(run, gate_output))
    details = pd.concat(parts, ignore_index=True)
    details = details.drop_duplicates(["artifact", "kind", "policy"], keep="first")
    summary = summarize_stability(
        details,
        expected_artifacts=len(runs),
        max_worst_drawdown=args.max_worst_drawdown,
        max_avg_turnover=args.max_avg_turnover,
    )

    output_dir = FILTER_ROOT / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "cross_artifact_details.csv", index=False)
    summary.to_csv(output_dir / "cross_artifact_summary.csv", index=False)
    write_markdown(
        output_dir,
        summary,
        details,
        max_worst_drawdown=args.max_worst_drawdown,
        max_avg_turnover=args.max_avg_turnover,
    )
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "runs": list(runs),
                "committee_output": args.committee_output,
                "gate_outputs": list(gate_outputs),
                "max_worst_drawdown": args.max_worst_drawdown,
                "max_avg_turnover": args.max_avg_turnover,
                "stable_non_oracle_policies": int(summary["passes_all_artifacts"].sum()),
                "risk_controlled_stable_policies": int(summary["passes_risk_controls"].sum()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "runs": list(runs),
                "stable_non_oracle_policies": int(summary["passes_all_artifacts"].sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
