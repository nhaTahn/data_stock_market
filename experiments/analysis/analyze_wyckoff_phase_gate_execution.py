from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.analyze_committee_hypothesis_grid import (  # noqa: E402
    DEFAULT_COVERAGE_GRID,
    add_committee_candidate_columns,
)
from experiments.analysis.analyze_wyckoff_architecture import (  # noqa: E402
    PHASE_ORDER,
    load_wyckoff_daily,
    summarize_by_phase,
    summarize_returns,
)
from experiments.analysis.analyze_wyckoff_phase_gate import (  # noqa: E402
    aggressive_prior_gate,
    choose_oracle_eval_phase_gate,
    choose_train_phase_gate,
    conservative_prior_gate,
)
from src.models.selection.holding_period import desired_weights_for_day, l1_turnover  # noqa: E402


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
FILTER_ROOT = RUN_ROOT / "reports" / "filter_signal"
DEFAULT_RUN = "portable_lstm_filter_signal_20260509_r06_selector_module"
DEFAULT_WYCKOFF_OUTPUT = "wyckoff_architecture_eval"
DEFAULT_TRAIN_OUTPUT = "committee_hypothesis_grid_train_split_t025_dd15"
DEFAULT_EVAL_OUTPUT = "committee_hypothesis_grid_val_split_t025_dd15"
DEFAULT_FALLBACK = "legacy_filter_shortlist"
CASH_POLICY = "__cash__"
ALL_COMMITTEE_IF_PRESSURE_NONNEG = "all_committee_candidates_if_pressure_delta_20_gte_0"
ALL_COMMITTEE_IF_MARKET_RETURN_NONNEG = "all_committee_candidates_if_market_return_20_gte_0"
ALL_COMMITTEE_IF_MARKET_RETURN_AND_PRESSURE_NONNEG = (
    "all_committee_candidates_if_market_return_20_and_pressure_delta_20_gte_0"
)
CONDITIONAL_BASE_POLICIES = {
    ALL_COMMITTEE_IF_PRESSURE_NONNEG: "all_committee_candidates",
    ALL_COMMITTEE_IF_MARKET_RETURN_NONNEG: "all_committee_candidates",
    ALL_COMMITTEE_IF_MARKET_RETURN_AND_PRESSURE_NONNEG: "all_committee_candidates",
}


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execution-level simulation for Wyckoff phase-gated committee policies.")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--wyckoff-output", default=DEFAULT_WYCKOFF_OUTPUT)
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--fallback-hypothesis", default=DEFAULT_FALLBACK)
    parser.add_argument("--min-train-phase-days", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--min-positions", type=int, default=5)
    parser.add_argument("--output-name", default="wyckoff_phase_gate_execution")
    parser.add_argument("--include-regularized-train-gates", action="store_true")
    parser.add_argument("--regularized-min-equity-margins", default="0.02,0.05")
    parser.add_argument("--regularized-min-net-equity", type=float, default=1.0)
    parser.add_argument("--regularized-min-sharpe", type=float, default=0.0)
    parser.add_argument("--regularized-max-drawdown", type=float, default=0.20)
    parser.add_argument("--regularized-max-avg-turnover", type=float, default=0.25)
    parser.add_argument("--include-abstain-train-gates", action="store_true")
    parser.add_argument("--include-robust-abstain-gates", action="store_true")
    parser.add_argument("--robust-min-t-stat", type=float, default=0.50)
    parser.add_argument("--robust-min-positive-fold-rate", type=float, default=0.60)
    parser.add_argument("--include-transition-risk-filter-gates", action="store_true")
    return parser.parse_args(argv)


def build_gate_maps(
    train_phase: pd.DataFrame,
    eval_phase: pd.DataFrame,
    *,
    fallback_hypothesis: str,
    min_train_phase_days: int,
    include_regularized_train_gates: bool = False,
    regularized_min_equity_margins: tuple[float, ...] = (0.02, 0.05),
    regularized_min_net_equity: float = 1.0,
    regularized_min_sharpe: float = 0.0,
    regularized_max_drawdown: float = 0.20,
    regularized_max_avg_turnover: float = 0.25,
    include_abstain_train_gates: bool = False,
    train_daily: pd.DataFrame | None = None,
    include_robust_abstain_gates: bool = False,
    robust_min_t_stat: float = 0.50,
    robust_min_positive_fold_rate: float = 0.60,
    include_transition_risk_filter_gates: bool = False,
) -> dict[str, dict[str, str]]:
    gates = {
        "train_phase_best": choose_train_phase_gate(
            train_phase,
            fallback_hypothesis=fallback_hypothesis,
            min_train_phase_days=min_train_phase_days,
        ),
        "conservative_prior": conservative_prior_gate(fallback_hypothesis),
        "aggressive_prior": aggressive_prior_gate(fallback_hypothesis),
        "oracle_eval_phase_best": choose_oracle_eval_phase_gate(
            eval_phase,
            fallback_hypothesis=fallback_hypothesis,
        ),
        "baseline_h1": {phase: "h1_tradeability_filter" for phase in PHASE_ORDER},
        "baseline_all_committee": {phase: "all_committee_candidates" for phase in PHASE_ORDER},
        "baseline_legacy": {phase: "legacy_filter_shortlist" for phase in PHASE_ORDER},
    }
    if include_regularized_train_gates:
        for margin in regularized_min_equity_margins:
            token = int(round(margin * 100))
            gate_name = (
                f"train_phase_regularized_m{token:02d}_"
                f"dd{int(round(regularized_max_drawdown * 100)):02d}_"
                f"t{int(round(regularized_max_avg_turnover * 100)):02d}"
            )
            gates[gate_name] = choose_regularized_train_phase_gate(
                train_phase,
                fallback_hypothesis=fallback_hypothesis,
                reject_hypothesis=fallback_hypothesis,
                min_train_phase_days=min_train_phase_days,
                min_equity_margin=margin,
                min_net_equity=regularized_min_net_equity,
                min_sharpe=regularized_min_sharpe,
                max_drawdown=regularized_max_drawdown,
                max_avg_turnover=regularized_max_avg_turnover,
            )
            if include_abstain_train_gates:
                abstain_gate_name = (
                    f"train_phase_abstain_m{token:02d}_"
                    f"dd{int(round(regularized_max_drawdown * 100)):02d}_"
                    f"t{int(round(regularized_max_avg_turnover * 100)):02d}"
                )
                gates[abstain_gate_name] = choose_regularized_train_phase_gate(
                    train_phase,
                    fallback_hypothesis=fallback_hypothesis,
                    reject_hypothesis=CASH_POLICY,
                    min_train_phase_days=min_train_phase_days,
                    min_equity_margin=margin,
                    min_net_equity=regularized_min_net_equity,
                    min_sharpe=regularized_min_sharpe,
                    max_drawdown=regularized_max_drawdown,
                    max_avg_turnover=regularized_max_avg_turnover,
                )
            if include_robust_abstain_gates:
                if train_daily is None:
                    raise ValueError("train_daily is required when include_robust_abstain_gates is enabled.")
                robust_stats = summarize_train_phase_robustness(train_daily)
                robust_gate_name = (
                    f"train_phase_robust_abstain_m{token:02d}_"
                    f"dd{int(round(regularized_max_drawdown * 100)):02d}_"
                    f"t{int(round(regularized_max_avg_turnover * 100)):02d}_"
                    f"ts{int(round(robust_min_t_stat * 100)):03d}_"
                    f"pf{int(round(robust_min_positive_fold_rate * 100)):02d}"
                )
                gates[robust_gate_name] = choose_robust_train_phase_gate(
                    robust_stats,
                    fallback_hypothesis=fallback_hypothesis,
                    reject_hypothesis=CASH_POLICY,
                    min_train_phase_days=min_train_phase_days,
                    min_equity_margin=margin,
                    min_net_equity=regularized_min_net_equity,
                    min_t_stat=robust_min_t_stat,
                    min_positive_fold_rate=robust_min_positive_fold_rate,
                    max_drawdown=regularized_max_drawdown,
                    max_avg_turnover=regularized_max_avg_turnover,
                )
    gates.update(vn_first_gate_grid(fallback_hypothesis))
    if include_transition_risk_filter_gates:
        source_gates = list(gates.items())
        risk_filters = {
            "pressure_nonneg": ALL_COMMITTEE_IF_PRESSURE_NONNEG,
            "mr20_nonneg": ALL_COMMITTEE_IF_MARKET_RETURN_NONNEG,
            "mr20_pressure_nonneg": ALL_COMMITTEE_IF_MARKET_RETURN_AND_PRESSURE_NONNEG,
        }
        for gate_name, mapping in source_gates:
            if mapping.get("transition") != "all_committee_candidates":
                continue
            for suffix, conditional_policy in risk_filters.items():
                filtered_mapping = dict(mapping)
                filtered_mapping["transition"] = conditional_policy
                gates[f"{gate_name}_transition_{suffix}"] = filtered_mapping
    return gates


def daily_t_stat(returns: pd.Series) -> float:
    values = returns.astype(float)
    std = float(values.std(ddof=1))
    if len(values) < 2 or std <= 0.0 or math.isnan(std):
        return float("nan")
    return float(values.mean() / (std / math.sqrt(len(values))))


def summarize_train_phase_robustness(train_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (phase, hypothesis), group in train_daily.groupby(["wyckoff_phase", "hypothesis"], sort=True):
        returns = group["net_return"].astype(float)
        fold_equity = group.groupby("fold", sort=True)["net_return"].apply(lambda values: float((1.0 + values).prod()))
        equity_curve = (1.0 + returns).cumprod()
        rows.append(
            {
                "phase": str(phase),
                "hypothesis": str(hypothesis),
                "n_days": int(len(group)),
                "net_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else float("nan"),
                "net_t_stat": daily_t_stat(returns),
                "positive_fold_rate": float((fold_equity > 1.0).mean()) if not fold_equity.empty else float("nan"),
                "max_drawdown": max_drawdown_from_equity(equity_curve),
                "avg_turnover": float(group["turnover"].mean()),
                "active_day_rate": float((group["gross_exposure"] > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def max_drawdown_from_equity(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    return float((equity_curve / running_max - 1.0).min())


def choose_robust_train_phase_gate(
    train_phase_stats: pd.DataFrame,
    *,
    fallback_hypothesis: str,
    reject_hypothesis: str,
    min_train_phase_days: int,
    min_equity_margin: float,
    min_net_equity: float,
    min_t_stat: float,
    min_positive_fold_rate: float,
    max_drawdown: float,
    max_avg_turnover: float,
) -> dict[str, str]:
    phase_gate: dict[str, str] = {}
    drawdown_floor = -abs(max_drawdown)
    for phase in PHASE_ORDER:
        phase_rows = train_phase_stats.loc[train_phase_stats["phase"] == phase].copy()
        if phase_rows.empty or int(phase_rows["n_days"].max()) < min_train_phase_days:
            phase_gate[phase] = reject_hypothesis
            continue
        fallback_rows = phase_rows.loc[phase_rows["hypothesis"] == fallback_hypothesis]
        fallback_equity = float(fallback_rows["net_equity"].max()) if not fallback_rows.empty else 1.0
        phase_rows = phase_rows.sort_values(
            ["net_equity", "net_t_stat", "positive_fold_rate", "max_drawdown"],
            ascending=[False, False, False, False],
            kind="stable",
        )
        selected = reject_hypothesis
        for _, row in phase_rows.iterrows():
            passes_margin = float(row["net_equity"]) >= max(min_net_equity, fallback_equity + min_equity_margin)
            passes_t = float(row["net_t_stat"]) >= min_t_stat
            passes_fold_rate = float(row["positive_fold_rate"]) >= min_positive_fold_rate
            passes_drawdown = float(row["max_drawdown"]) >= drawdown_floor
            passes_turnover = float(row["avg_turnover"]) <= max_avg_turnover
            if passes_margin and passes_t and passes_fold_rate and passes_drawdown and passes_turnover:
                selected = str(row["hypothesis"])
                break
        phase_gate[phase] = selected
    return phase_gate


def choose_regularized_train_phase_gate(
    train_phase_summary: pd.DataFrame,
    *,
    fallback_hypothesis: str,
    reject_hypothesis: str,
    min_train_phase_days: int,
    min_equity_margin: float,
    min_net_equity: float,
    min_sharpe: float,
    max_drawdown: float,
    max_avg_turnover: float,
) -> dict[str, str]:
    phase_gate: dict[str, str] = {}
    drawdown_floor = -abs(max_drawdown)
    for phase in PHASE_ORDER:
        phase_rows = train_phase_summary.loc[train_phase_summary["phase"] == phase].copy()
        if phase_rows.empty or int(phase_rows["n_days"].max()) < min_train_phase_days:
            phase_gate[phase] = reject_hypothesis
            continue

        fallback_rows = phase_rows.loc[phase_rows["hypothesis"] == fallback_hypothesis]
        fallback_equity = float(fallback_rows["net_equity"].max()) if not fallback_rows.empty else 1.0
        phase_rows = phase_rows.sort_values(
            ["net_equity", "net_sharpe", "max_drawdown"],
            ascending=[False, False, False],
            kind="stable",
        )
        selected = reject_hypothesis
        for _, row in phase_rows.iterrows():
            hypothesis = str(row["hypothesis"])
            if hypothesis == fallback_hypothesis:
                selected = fallback_hypothesis if reject_hypothesis == fallback_hypothesis else reject_hypothesis
                break
            passes_margin = float(row["net_equity"]) >= max(min_net_equity, fallback_equity + min_equity_margin)
            passes_sharpe = float(row["net_sharpe"]) >= min_sharpe
            passes_drawdown = float(row["max_drawdown"]) >= drawdown_floor
            passes_turnover = float(row.get("avg_turnover", 0.0)) <= max_avg_turnover
            if passes_margin and passes_sharpe and passes_drawdown and passes_turnover:
                selected = hypothesis
                break
        phase_gate[phase] = selected
    return phase_gate


def vn_first_gate_grid(fallback_hypothesis: str) -> dict[str, dict[str, str]]:
    return {
        "vn_legacy_acc_all_else": {
            "accumulation": "legacy_filter_shortlist",
            "markup": "all_committee_candidates",
            "distribution": "all_committee_candidates",
            "markdown": "all_committee_candidates",
            "transition": "all_committee_candidates",
        },
        "vn_legacy_acc_markup_all_else": {
            "accumulation": "legacy_filter_shortlist",
            "markup": "legacy_filter_shortlist",
            "distribution": "all_committee_candidates",
            "markdown": "all_committee_candidates",
            "transition": "all_committee_candidates",
        },
        "vn_h1_distribution_only": {
            "accumulation": "legacy_filter_shortlist",
            "markup": fallback_hypothesis,
            "distribution": "h1_tradeability_filter",
            "markdown": "all_committee_candidates",
            "transition": "all_committee_candidates",
        },
        "vn_h1_transition_only": {
            "accumulation": "legacy_filter_shortlist",
            "markup": fallback_hypothesis,
            "distribution": "all_committee_candidates",
            "markdown": "all_committee_candidates",
            "transition": "h1_tradeability_filter",
        },
        "vn_h1_distribution_transition": {
            "accumulation": "legacy_filter_shortlist",
            "markup": fallback_hypothesis,
            "distribution": "h1_tradeability_filter",
            "markdown": "all_committee_candidates",
            "transition": "h1_tradeability_filter",
        },
        "vn_h1_markup_distribution_transition": {
            "accumulation": "legacy_filter_shortlist",
            "markup": "h1_tradeability_filter",
            "distribution": "h1_tradeability_filter",
            "markdown": "all_committee_candidates",
            "transition": "h1_tradeability_filter",
        },
    }


def selection_lookup(fold_results: pd.DataFrame) -> dict[tuple[int, str], dict[str, object]]:
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    for _, row in fold_results.iterrows():
        lookup[(int(row["fold"]), str(row["hypothesis"]))] = {
            "candidate": str(row["candidate"]),
            "rebalance_every": int(row["rebalance_every"]),
        }
    return lookup


def resolve_conditional_hypothesis(day: pd.DataFrame, hypothesis: str) -> str:
    if hypothesis == ALL_COMMITTEE_IF_PRESSURE_NONNEG:
        return "all_committee_candidates" if day_feature(day, "pressure_delta_20") >= 0.0 else CASH_POLICY
    if hypothesis == ALL_COMMITTEE_IF_MARKET_RETURN_NONNEG:
        return "all_committee_candidates" if day_feature(day, "market_return_20") >= 0.0 else CASH_POLICY
    if hypothesis == ALL_COMMITTEE_IF_MARKET_RETURN_AND_PRESSURE_NONNEG:
        if day_feature(day, "market_return_20") >= 0.0 and day_feature(day, "pressure_delta_20") >= 0.0:
            return "all_committee_candidates"
        return CASH_POLICY
    return hypothesis


def day_feature(day: pd.DataFrame, column: str) -> float:
    if column not in day.columns:
        return float("nan")
    values = day[column].dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def simulate_gate_fold(
    fold_frame: pd.DataFrame,
    *,
    fold: int,
    gate_name: str,
    phase_gate: dict[str, str],
    selected_by_fold: dict[tuple[int, str], dict[str, object]],
    cost_bps: float,
    min_positions: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_weights: dict[str, float] = {}
    current_policy: tuple[str, str, int] | None = None
    daily_groups = list(
        fold_frame.dropna(subset=["actual_date"]).sort_values("actual_date").groupby("actual_date", sort=True)
    )
    for idx, (actual_date, day) in enumerate(daily_groups):
        signal_date = pd.Timestamp(day["Date"].max())
        phase = str(day["wyckoff_phase"].dropna().iloc[0]) if day["wyckoff_phase"].notna().any() else "transition"
        hypothesis = phase_gate.get(phase, phase_gate.get("transition", "legacy_filter_shortlist"))
        hypothesis = resolve_conditional_hypothesis(day, hypothesis)
        if hypothesis == CASH_POLICY:
            candidate = CASH_POLICY
            rebalance_every = 1
        else:
            selection = selected_by_fold.get((fold, hypothesis))
            if selection is None:
                hypothesis = "legacy_filter_shortlist"
                selection = selected_by_fold[(fold, hypothesis)]
            candidate = str(selection["candidate"])
            rebalance_every = int(selection["rebalance_every"])
        if hypothesis == CASH_POLICY:
            selection = None
        elif selection is None:
            hypothesis = "legacy_filter_shortlist"
            selection = selected_by_fold[(fold, hypothesis)]
        policy = (hypothesis, candidate, rebalance_every)
        policy_changed = idx == 0 or policy != current_policy
        should_rebalance = policy_changed or idx % rebalance_every == 0
        turnover = 0.0
        if should_rebalance:
            target_weights = {} if hypothesis == CASH_POLICY else desired_weights_for_day(day, candidate, min_positions)
            turnover = l1_turnover(current_weights, target_weights)
            current_weights = target_weights
            current_policy = policy

        returns = dict(zip(day["code"].astype(str), day["actual_aligned"].astype(float)))
        gross_return = float(sum(weight * returns.get(code, 0.0) for code, weight in current_weights.items()))
        cost_return = turnover * cost_bps / 10_000.0
        rows.append(
            {
                "gate": gate_name,
                "fold": fold,
                "signal_date": signal_date,
                "actual_date": actual_date,
                "wyckoff_phase": phase,
                "hypothesis": hypothesis,
                "candidate": candidate,
                "rebalance_every": rebalance_every,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost_return": cost_return,
                "net_return": gross_return - cost_return,
                "n_positions": int(sum(abs(weight) > 0.0 for weight in current_weights.values())),
                "gross_exposure": float(sum(abs(weight) for weight in current_weights.values())),
                "is_rebalance_day": should_rebalance,
                "policy_changed": policy_changed,
            }
        )
    return pd.DataFrame(rows)


def simulate_gate(
    predictions: pd.DataFrame,
    fold_results: pd.DataFrame,
    gate_name: str,
    phase_gate: dict[str, str],
    *,
    cost_bps: float,
    min_positions: int,
) -> pd.DataFrame:
    selected_by_fold = selection_lookup(fold_results)
    parts: list[pd.DataFrame] = []
    for fold, fold_rows in fold_results.groupby("fold", sort=True):
        fold_row = fold_rows.iloc[0]
        fold_frame = predictions.loc[
            (predictions["actual_date"] >= pd.Timestamp(fold_row["test_start"]))
            & (predictions["actual_date"] <= pd.Timestamp(fold_row["test_end"]))
        ].copy()
        parts.append(
            simulate_gate_fold(
                fold_frame,
                fold=int(fold),
                gate_name=gate_name,
                phase_gate=phase_gate,
                selected_by_fold=selected_by_fold,
                cost_bps=cost_bps,
                min_positions=min_positions,
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_markdown(
    output_dir: Path,
    total: pd.DataFrame,
    phase: pd.DataFrame,
    gate_maps: dict[str, dict[str, str]],
) -> None:
    lines = [
        "# Wyckoff Phase Gate Execution Simulation",
        "",
        "This simulation recomputes positions, turnover, and cost when the gate changes policy by Wyckoff phase.",
        "`oracle_eval_phase_best` remains an upper-bound diagnostic because it uses validation phase outcomes.",
        "",
        "## Gate Maps",
        "",
        "| Gate | Accumulation | Markup | Distribution | Markdown | Transition |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for gate, mapping in gate_maps.items():
        lines.append(
            "| "
            f"`{gate}` | `{mapping.get('accumulation')}` | `{mapping.get('markup')}` | "
            f"`{mapping.get('distribution')}` | `{mapping.get('markdown')}` | `{mapping.get('transition')}` |"
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Gate | Net equity | Sharpe | Max DD | Avg turnover | Hit rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in total.iterrows():
        lines.append(
            "| "
            f"`{row['gate']}` | {float(row['net_equity']):.3f} | {float(row['net_sharpe']):+.2f} | "
            f"{float(row['max_drawdown']):.1%} | {float(row['avg_turnover']):.2f} | {float(row['hit_rate']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## By Phase",
            "",
            "| Gate | Phase | Days | Net equity | Sharpe | Max DD | Avg turnover |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    phase_order = {phase_name: idx for idx, phase_name in enumerate(PHASE_ORDER)}
    ordered_phase = phase.copy()
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
    predictions = pd.read_csv(run_dir / "filter_predictions.csv.gz", parse_dates=["Date", "actual_date"])
    add_committee_candidate_columns(predictions, DEFAULT_COVERAGE_GRID)
    wyckoff_daily = load_wyckoff_daily()
    wyckoff_columns = [
        "Date",
        "wyckoff_phase",
        "market_return_20",
        "market_return_60",
        "breadth_20",
        "location_20",
        "pressure_delta_20",
    ]
    predictions = predictions.merge(
        wyckoff_daily[wyckoff_columns],
        on="Date",
        how="left",
    )
    predictions["wyckoff_phase"] = predictions["wyckoff_phase"].fillna("transition")

    fold_results = pd.read_csv(
        run_dir / args.eval_output / "committee_hypothesis_folds.csv",
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    )
    train_phase = pd.read_csv(wyckoff_dir / f"{args.train_output}_phase_summary.csv")
    eval_phase = pd.read_csv(wyckoff_dir / f"{args.eval_output}_phase_summary.csv")
    train_daily = None
    if args.include_robust_abstain_gates:
        train_daily = pd.read_csv(
            wyckoff_dir / f"{args.train_output}_selected_daily.csv",
            parse_dates=["signal_date", "actual_date"],
        )
    gate_maps = build_gate_maps(
        train_phase,
        eval_phase,
        fallback_hypothesis=args.fallback_hypothesis,
        min_train_phase_days=args.min_train_phase_days,
        include_regularized_train_gates=args.include_regularized_train_gates,
        regularized_min_equity_margins=parse_float_list(args.regularized_min_equity_margins),
        regularized_min_net_equity=args.regularized_min_net_equity,
        regularized_min_sharpe=args.regularized_min_sharpe,
        regularized_max_drawdown=args.regularized_max_drawdown,
        regularized_max_avg_turnover=args.regularized_max_avg_turnover,
        include_abstain_train_gates=args.include_abstain_train_gates,
        train_daily=train_daily,
        include_robust_abstain_gates=args.include_robust_abstain_gates,
        robust_min_t_stat=args.robust_min_t_stat,
        robust_min_positive_fold_rate=args.robust_min_positive_fold_rate,
        include_transition_risk_filter_gates=args.include_transition_risk_filter_gates,
    )
    daily = pd.concat(
        [
            simulate_gate(
                predictions,
                fold_results,
                gate_name,
                mapping,
                cost_bps=args.cost_bps,
                min_positions=args.min_positions,
            )
            for gate_name, mapping in gate_maps.items()
        ],
        ignore_index=True,
    )
    total_rows: list[dict[str, object]] = []
    phase_parts: list[pd.DataFrame] = []
    for gate, group in daily.groupby("gate", sort=True):
        total_rows.append({"gate": gate, **summarize_returns(group)})
        phase_input = group.copy()
        phase_input["hypothesis"] = gate
        phase_summary, _ = summarize_by_phase(phase_input)
        phase_summary = phase_summary.rename(columns={"hypothesis": "gate"})
        phase_parts.append(phase_summary)
    total = pd.DataFrame(total_rows).sort_values(["net_equity", "net_sharpe"], ascending=[False, False], kind="stable")
    phase = pd.concat(phase_parts, ignore_index=True)

    output_dir = run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "wyckoff_phase_gate_execution_daily.csv", index=False)
    total.to_csv(output_dir / "wyckoff_phase_gate_execution_summary.csv", index=False)
    phase.to_csv(output_dir / "wyckoff_phase_gate_execution_by_phase.csv", index=False)
    output_dir.joinpath("gate_map.json").write_text(json.dumps(gate_maps, indent=2), encoding="utf-8")
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run": args.run,
                "wyckoff_output": args.wyckoff_output,
                "train_output": args.train_output,
                "eval_output": args.eval_output,
                "cost_bps": args.cost_bps,
                "min_positions": args.min_positions,
                "gate_maps": gate_maps,
                "summary": total.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, total, phase, gate_maps)
    print(json.dumps({"output_dir": str(output_dir), "gates": int(len(total))}, indent=2))


if __name__ == "__main__":
    main()
