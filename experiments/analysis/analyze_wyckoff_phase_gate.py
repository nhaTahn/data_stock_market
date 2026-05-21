from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_wyckoff_architecture import (  # noqa: E402
    PHASE_ORDER,
    summarize_by_phase,
    summarize_returns,
)


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
FILTER_ROOT = RUN_ROOT / "reports" / "filter_signal"
DEFAULT_RUN = "portable_lstm_filter_signal_20260509_r06_selector_module"
DEFAULT_WYCKOFF_OUTPUT = "wyckoff_architecture_eval"
DEFAULT_TRAIN_OUTPUT = "committee_hypothesis_grid_train_split_t025_dd15"
DEFAULT_EVAL_OUTPUT = "committee_hypothesis_grid_val_split_t025_dd15"
DEFAULT_FALLBACK = "legacy_filter_shortlist"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate train-selected Wyckoff phase gates across committee policies.")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--wyckoff-output", default=DEFAULT_WYCKOFF_OUTPUT)
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--fallback-hypothesis", default=DEFAULT_FALLBACK)
    parser.add_argument("--min-train-phase-days", type=int, default=20)
    parser.add_argument("--output-name", default="wyckoff_phase_gate_eval")
    return parser.parse_args(argv)


def choose_train_phase_gate(
    train_phase_summary: pd.DataFrame,
    *,
    fallback_hypothesis: str,
    min_train_phase_days: int,
) -> dict[str, str]:
    phase_gate: dict[str, str] = {}
    for phase in PHASE_ORDER:
        phase_rows = train_phase_summary.loc[train_phase_summary["phase"] == phase].copy()
        if phase_rows.empty or int(phase_rows["n_days"].max()) < min_train_phase_days:
            phase_gate[phase] = fallback_hypothesis
            continue
        phase_rows = phase_rows.sort_values(
            ["net_equity", "net_sharpe", "max_drawdown"],
            ascending=[False, False, False],
            kind="stable",
        )
        phase_gate[phase] = str(phase_rows.iloc[0]["hypothesis"])
    return phase_gate


def choose_oracle_eval_phase_gate(
    eval_phase_summary: pd.DataFrame,
    *,
    fallback_hypothesis: str,
) -> dict[str, str]:
    phase_gate: dict[str, str] = {}
    for phase in PHASE_ORDER:
        phase_rows = eval_phase_summary.loc[eval_phase_summary["phase"] == phase].copy()
        if phase_rows.empty or int(phase_rows["n_days"].max()) == 0:
            phase_gate[phase] = fallback_hypothesis
            continue
        phase_rows = phase_rows.sort_values(
            ["net_equity", "net_sharpe", "max_drawdown"],
            ascending=[False, False, False],
            kind="stable",
        )
        phase_gate[phase] = str(phase_rows.iloc[0]["hypothesis"])
    return phase_gate


def conservative_prior_gate(fallback_hypothesis: str) -> dict[str, str]:
    return {
        "accumulation": "legacy_filter_shortlist",
        "markup": fallback_hypothesis,
        "distribution": "all_committee_candidates",
        "markdown": "all_committee_candidates",
        "transition": "all_committee_candidates",
    }


def aggressive_prior_gate(fallback_hypothesis: str) -> dict[str, str]:
    return {
        "accumulation": "legacy_filter_shortlist",
        "markup": "h1_tradeability_filter",
        "distribution": "h1_tradeability_filter",
        "markdown": "all_committee_candidates",
        "transition": "h1_tradeability_filter",
    }


def apply_gate(selected_daily: pd.DataFrame, gate_name: str, phase_gate: dict[str, str]) -> pd.DataFrame:
    work = selected_daily.copy()
    work["gate_hypothesis"] = work["wyckoff_phase"].map(phase_gate).fillna(phase_gate.get("transition"))
    gated = work.loc[work["hypothesis"] == work["gate_hypothesis"]].copy()
    gated.insert(0, "gate", gate_name)
    return gated


def summarize_gate(gated_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    phase_parts: list[pd.DataFrame] = []
    for gate, group in gated_daily.groupby("gate", sort=True):
        rows.append({"gate": gate, **summarize_returns(group)})
        summary_input = group.copy()
        summary_input["hypothesis"] = gate
        phase_summary, _ = summarize_by_phase(summary_input)
        phase_summary = phase_summary.rename(columns={"hypothesis": "gate"})
        phase_parts.append(phase_summary)
    total = pd.DataFrame(rows).sort_values(["net_equity", "net_sharpe"], ascending=[False, False], kind="stable")
    phase = pd.concat(phase_parts, ignore_index=True) if phase_parts else pd.DataFrame()
    return total, phase


def write_markdown(
    output_dir: Path,
    gate_total: pd.DataFrame,
    gate_phase: pd.DataFrame,
    gate_map: dict[str, dict[str, str]],
    baseline_total: pd.DataFrame,
) -> None:
    lines = [
        "# Wyckoff Phase Gate Evaluation",
        "",
        "Gate maps are applied to validation selected-daily returns by signal-date Wyckoff phase.",
        "`oracle_eval_phase_best` is an upper-bound diagnostic and must not be promoted as a trade policy.",
        "",
        "## Gate Maps",
        "",
        "| Gate | Accumulation | Markup | Distribution | Markdown | Transition |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, mapping in gate_map.items():
        lines.append(
            "| "
            f"`{name}` | `{mapping.get('accumulation')}` | `{mapping.get('markup')}` | "
            f"`{mapping.get('distribution')}` | `{mapping.get('markdown')}` | `{mapping.get('transition')}` |"
        )
    lines.extend(
        [
            "",
            "## Gate Results",
            "",
            "| Gate | Net equity | Sharpe | Max DD | Avg turnover | Hit rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in gate_total.iterrows():
        lines.append(
            "| "
            f"`{row['gate']}` | {float(row['net_equity']):.3f} | {float(row['net_sharpe']):+.2f} | "
            f"{float(row['max_drawdown']):.1%} | {float(row['avg_turnover']):.2f} | {float(row['hit_rate']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Hypotheses",
            "",
            "| Hypothesis | Net equity | Sharpe | Max DD | Avg turnover | Hit rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in baseline_total.sort_values("net_equity", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['hypothesis']}` | {float(row['net_equity']):.3f} | {float(row['net_sharpe']):+.2f} | "
            f"{float(row['max_drawdown']):.1%} | {float(row['avg_turnover']):.2f} | {float(row['hit_rate']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Gate By Phase",
            "",
            "| Gate | Phase | Days | Net equity | Sharpe | Max DD | Avg turnover |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    phase_order = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}
    ordered_phase = gate_phase.copy()
    ordered_phase["phase_order"] = ordered_phase["phase"].map(phase_order)
    ordered_phase = ordered_phase.sort_values(["gate", "phase_order"], kind="stable")
    for _, row in ordered_phase.iterrows():
        lines.append(
            "| "
            f"`{row['gate']}` | `{row['phase']}` | {int(row['n_days'])} | "
            f"{float(row['net_equity']):.3f} | {float(row['net_sharpe']):+.2f} | "
            f"{float(row['max_drawdown']):.1%} | {float(row['avg_turnover']):.2f} |"
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = FILTER_ROOT / args.run
    wyckoff_dir = run_dir / args.wyckoff_output
    train_phase = pd.read_csv(wyckoff_dir / f"{args.train_output}_phase_summary.csv")
    eval_phase = pd.read_csv(wyckoff_dir / f"{args.eval_output}_phase_summary.csv")
    eval_daily = pd.read_csv(
        wyckoff_dir / f"{args.eval_output}_selected_daily.csv",
        parse_dates=["signal_date", "actual_date"],
    )
    baseline_total = pd.read_csv(wyckoff_dir / f"{args.eval_output}_total_summary.csv")

    gate_map = {
        "train_phase_best": choose_train_phase_gate(
            train_phase,
            fallback_hypothesis=args.fallback_hypothesis,
            min_train_phase_days=args.min_train_phase_days,
        ),
        "conservative_prior": conservative_prior_gate(args.fallback_hypothesis),
        "aggressive_prior": aggressive_prior_gate(args.fallback_hypothesis),
        "oracle_eval_phase_best": choose_oracle_eval_phase_gate(
            eval_phase,
            fallback_hypothesis=args.fallback_hypothesis,
        ),
    }
    gated_daily = pd.concat(
        [apply_gate(eval_daily, gate_name, mapping) for gate_name, mapping in gate_map.items()],
        ignore_index=True,
    )
    gate_total, gate_phase = summarize_gate(gated_daily)
    output_dir = run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    gated_daily.to_csv(output_dir / "wyckoff_phase_gate_daily.csv", index=False)
    gate_total.to_csv(output_dir / "wyckoff_phase_gate_summary.csv", index=False)
    gate_phase.to_csv(output_dir / "wyckoff_phase_gate_by_phase.csv", index=False)
    output_dir.joinpath("gate_map.json").write_text(json.dumps(gate_map, indent=2), encoding="utf-8")
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run": args.run,
                "wyckoff_output": args.wyckoff_output,
                "train_output": args.train_output,
                "eval_output": args.eval_output,
                "fallback_hypothesis": args.fallback_hypothesis,
                "min_train_phase_days": args.min_train_phase_days,
                "gate_map": gate_map,
                "summary": gate_total.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, gate_total, gate_phase, gate_map, baseline_total)
    print(json.dumps({"output_dir": str(output_dir), "gates": int(len(gate_total))}, indent=2))


if __name__ == "__main__":
    main()
