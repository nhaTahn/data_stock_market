from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.summarize_current_best_signmag_feature_pruning import (  # noqa: E402
    build_summary_rows,
    write_summary_outputs,
)
from src.reporting import get_default_reporting_standard  # noqa: E402


GOLD_PACKAGE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "gold"
    / "best_vn30_broad_lstm_signmag_seed_52_20260412"
)
SOURCE_CONFIG_PATH = GOLD_PACKAGE_DIR / "core" / "source_config.json"
RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
EXPERIMENT_REPORT_ROOT = RUN_ROOT / "reports" / "feature_pruning"


PRICE_LEVEL_BLOCK = (
    "open_level_20",
    "high_level_20",
    "low_level_20",
    "close_level_20",
    "volume_level_20",
)
DELTA_BLOCK = (
    "open_delta_1",
    "high_delta_1",
    "low_delta_1",
    "close_delta_1",
    "volume_delta_1",
)
MARKET_CONTEXT_BLOCK = (
    "vnindex_return",
    "market_leader_return",
    "a_d_ratio",
    "day_of_week",
)
VINGROUP_MOMENTUM_BLOCK = ("vingroup_momentum", "market_leader_return")
SECTOR_LEADERSHIP_FEATURES = (
    "sector_momentum_rank",
    "is_top_2_sector",
)
SECTOR_MOMENTUM_RELATIVE_FEATURES = (
    "sector_momentum_rank",
    "sector_momentum_rank_pct",
    "sector_momentum_20",
    "relative_sector_momentum_20",
)
SECTOR_BREADTH_FEATURES = (
    "sector_positive_ratio",
    "sector_ad_ratio",
)
SECTOR_RETURN_FEATURES = (
    "sector_return",
    "alpha_sector",
)
MARKET_MOMENTUM_FEATURES = (
    "market_return_5",
    "market_return_20",
)
MARKET_REGIME_FEATURES = (
    "market_return_5",
    "market_return_20",
    "market_volatility_20",
    "market_ad_ratio_20",
)
FAST_OVERLAP_BLOCK = (
    "momentum_5",
    "rsi_14",
)
MOMENTUM_5_BLOCK = ("momentum_5",)
RSI_14_BLOCK = ("rsi_14",)
MACD_BLOCK = ("macd_hist",)
COMPACT_CORE_FEATURES = (
    "intraday_return",
    "gap_open",
    "close_position",
    "bb_width",
    "volume_ratio_20",
    "volatility_20",
    "momentum_20",
    "macd_hist",
    "vnindex_return",
    "market_leader_return",
    "a_d_ratio",
    "day_of_week",
)


@dataclass(frozen=True)
class FeatureCase:
    name: str
    notes: str
    feature_columns: tuple[str, ...]
    removed_groups: tuple[str, ...]
    config_overrides: dict[str, object] = field(default_factory=dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    standard = get_default_reporting_standard()
    parser = argparse.ArgumentParser(
        description="Run a narrow feature-pruning batch around the current best broad VN30 signmag setup."
    )
    parser.add_argument(
        "--stamp",
        default=datetime.now().strftime("%Y%m%d"),
        help="Batch stamp used in run names and report paths.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=ROOT / "venv" / "bin" / "python",
        help="Python executable used to launch training runs.",
    )
    parser.add_argument(
        "--case-set",
        choices=[
            "initial",
            "fast_overlap_followup",
            "sector_leadership_followup",
            "sector_generalized_followup",
            "sector_generalized_ablation",
            "sector_generalized_hparam",
            "market_regime_smoke",
            "objective_window_smoke",
            "phase_ic_compact_smoke",
            "phase_ic_sector_followup",
        ],
        default="initial",
        help="Feature-pruning case set to run.",
    )
    parser.add_argument(
        "--train-end-date",
        default=standard.window("train").end_date,
        help="Training end date. Defaults to the VN reporting standard train boundary.",
    )
    parser.add_argument(
        "--val-end-date",
        default=standard.window("val").end_date,
        help="Validation end date. Defaults to the VN reporting standard validation boundary.",
    )
    parser.add_argument(
        "--allow-nonstandard-time",
        action="store_true",
        help="Only use this if you intentionally want to override the locked in-sample/out-sample boundary.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Write the manifest and print planned commands without launching training.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_remove(base_features: tuple[str, ...], removed_features: tuple[str, ...]) -> tuple[str, ...]:
    removed = set(removed_features)
    return tuple(feature for feature in base_features if feature not in removed)


def build_feature_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    compact_core = tuple(feature for feature in base_features if feature in set(COMPACT_CORE_FEATURES))
    cases = [
        FeatureCase(
            name="full24",
            notes="Anchor broad-paper feature set. Use this as the comparison point.",
            feature_columns=base_features,
            removed_groups=(),
        ),
        FeatureCase(
            name="no_market_context",
            notes="Drop macro/breadth context to check whether cross-sectional market signals are diluting stock-specific alpha.",
            feature_columns=ordered_remove(base_features, MARKET_CONTEXT_BLOCK),
            removed_groups=("market_context",),
        ),
        FeatureCase(
            name="no_price_level_block",
            notes="Drop MA20 level-position features to test whether they duplicate the candle/volatility block.",
            feature_columns=ordered_remove(base_features, PRICE_LEVEL_BLOCK),
            removed_groups=("price_level_block",),
        ),
        FeatureCase(
            name="no_delta_block",
            notes="Drop 1-day delta features to test whether very short-horizon noise is hurting stability.",
            feature_columns=ordered_remove(base_features, DELTA_BLOCK),
            removed_groups=("delta_block",),
        ),
        FeatureCase(
            name="no_fast_overlap",
            notes="Drop the fastest overlapping momentum indicators and keep slower trend context.",
            feature_columns=ordered_remove(base_features, FAST_OVERLAP_BLOCK),
            removed_groups=("fast_overlap_block",),
        ),
        FeatureCase(
            name="compact_core12",
            notes="Keep only the compact core of candle state, volatility, medium-horizon trend, and market breadth/context.",
            feature_columns=compact_core,
            removed_groups=("price_level_block", "delta_block", "fast_overlap_block"),
        ),
    ]
    return cases


def build_fast_overlap_followup_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    no_fast_overlap = ordered_remove(base_features, FAST_OVERLAP_BLOCK)
    return [
        FeatureCase(
            name="no_momentum_5",
            notes="Drop only momentum_5 to isolate whether short momentum is the noisy feature.",
            feature_columns=ordered_remove(base_features, MOMENTUM_5_BLOCK),
            removed_groups=("momentum_5",),
        ),
        FeatureCase(
            name="no_rsi_14",
            notes="Drop only rsi_14 to isolate whether the oscillator is the noisy feature.",
            feature_columns=ordered_remove(base_features, RSI_14_BLOCK),
            removed_groups=("rsi_14",),
        ),
        FeatureCase(
            name="no_fast_overlap",
            notes="Repeat the current best feature-pruned setup as the batch anchor.",
            feature_columns=no_fast_overlap,
            removed_groups=("fast_overlap_block",),
        ),
        FeatureCase(
            name="no_fast_overlap_no_macd",
            notes="Drop momentum_5, rsi_14, and macd_hist to test whether all fast technical oscillators are redundant.",
            feature_columns=ordered_remove(base_features, (*FAST_OVERLAP_BLOCK, *MACD_BLOCK)),
            removed_groups=("fast_overlap_block", "macd_hist"),
        ),
        FeatureCase(
            name="no_fast_overlap_no_market_context",
            notes="Combine the winning fast-overlap removal with no market-context features.",
            feature_columns=ordered_remove(base_features, (*FAST_OVERLAP_BLOCK, *MARKET_CONTEXT_BLOCK)),
            removed_groups=("fast_overlap_block", "market_context"),
        ),
        FeatureCase(
            name="no_fast_overlap_no_price_level",
            notes="Combine the winning fast-overlap removal with no MA20 level-position block.",
            feature_columns=ordered_remove(base_features, (*FAST_OVERLAP_BLOCK, *PRICE_LEVEL_BLOCK)),
            removed_groups=("fast_overlap_block", "price_level_block"),
        ),
    ]


def ordered_replace(
    base_features: tuple[str, ...],
    removed_features: tuple[str, ...],
    added_features: tuple[str, ...],
    *,
    insert_after: str,
) -> tuple[str, ...]:
    removed = set(removed_features)
    added_set = set(added_features)
    out: list[str] = []
    inserted = False
    for feature in base_features:
        if feature in removed or feature in added_set:
            continue
        out.append(feature)
        if feature == insert_after:
            out.extend(feature for feature in added_features if feature not in out)
            inserted = True
    if not inserted:
        out.extend(feature for feature in added_features if feature not in out)
    return tuple(out)


def ordered_insert_after(
    base_features: tuple[str, ...],
    added_features: tuple[str, ...],
    *,
    insert_after: str,
) -> tuple[str, ...]:
    return ordered_replace(base_features, (), added_features, insert_after=insert_after)


def build_sector_leadership_followup_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    no_fast_overlap = ordered_remove(base_features, FAST_OVERLAP_BLOCK)
    no_fast_no_vingroup = ordered_remove(no_fast_overlap, VINGROUP_MOMENTUM_BLOCK)
    leadership_both = ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        SECTOR_LEADERSHIP_FEATURES,
        insert_after="vnindex_return",
    )
    leadership_rank = ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        ("sector_momentum_rank",),
        insert_after="vnindex_return",
    )
    leadership_top_flag = ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        ("is_top_2_sector",),
        insert_after="vnindex_return",
    )
    return [
        FeatureCase(
            name="no_fast_overlap",
            notes="Current best feature-pruned setup; still contains vingroup_momentum.",
            feature_columns=no_fast_overlap,
            removed_groups=("fast_overlap_block",),
        ),
        FeatureCase(
            name="no_fast_overlap_no_vingroup",
            notes="Remove vingroup_momentum without replacement to measure dependency on the non-generalizable VIN basket.",
            feature_columns=no_fast_no_vingroup,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
        FeatureCase(
            name="no_fast_overlap_sector_rank",
            notes="Replace vingroup_momentum with causal sector_momentum_rank.",
            feature_columns=leadership_rank,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
        FeatureCase(
            name="no_fast_overlap_top_sector_flag",
            notes="Replace vingroup_momentum with causal is_top_2_sector flag.",
            feature_columns=leadership_top_flag,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
        FeatureCase(
            name="no_fast_overlap_sector_leadership",
            notes="Replace vingroup_momentum with both sector_momentum_rank and is_top_2_sector.",
            feature_columns=leadership_both,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
    ]


def build_sector_generalized_followup_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    no_fast_overlap = ordered_remove(base_features, FAST_OVERLAP_BLOCK)
    sector_rank = ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        ("sector_momentum_rank",),
        insert_after="vnindex_return",
    )
    sector_rank_alpha = ordered_insert_after(
        sector_rank,
        ("alpha_sector",),
        insert_after="sector_momentum_rank",
    )
    sector_rank_return_alpha = ordered_insert_after(
        sector_rank,
        SECTOR_RETURN_FEATURES,
        insert_after="sector_momentum_rank",
    )
    sector_momentum_relative = ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        SECTOR_MOMENTUM_RELATIVE_FEATURES,
        insert_after="vnindex_return",
    )
    sector_breadth = ordered_insert_after(
        sector_rank,
        SECTOR_BREADTH_FEATURES,
        insert_after="sector_momentum_rank",
    )
    sector_general_full = ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        (
            *SECTOR_MOMENTUM_RELATIVE_FEATURES,
            *SECTOR_RETURN_FEATURES,
            *SECTOR_BREADTH_FEATURES,
        ),
        insert_after="vnindex_return",
    )
    return [
        FeatureCase(
            name="general_sector_rank",
            notes="Portable anchor: replace vingroup_momentum with causal sector_momentum_rank.",
            feature_columns=sector_rank,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
        FeatureCase(
            name="general_sector_rank_alpha",
            notes="Portable rank plus stock-vs-sector one-day alpha.",
            feature_columns=sector_rank_alpha,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
        FeatureCase(
            name="general_sector_rank_return_alpha",
            notes="Portable rank plus leave-one-out sector return and stock-vs-sector alpha.",
            feature_columns=sector_rank_return_alpha,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
        FeatureCase(
            name="general_sector_momentum_relative",
            notes="Portable causal sector momentum strength, normalized rank, and stock-vs-sector momentum gap.",
            feature_columns=sector_momentum_relative,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
        FeatureCase(
            name="general_sector_breadth",
            notes="Portable sector rank plus leave-one-out sector breadth.",
            feature_columns=sector_breadth,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
        FeatureCase(
            name="general_sector_full",
            notes="Portable sector rank, momentum, return, alpha, and breadth context.",
            feature_columns=sector_general_full,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
    ]


def build_sector_generalized_ablation_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    no_fast_overlap = ordered_remove(base_features, FAST_OVERLAP_BLOCK)
    sector_general_full = ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        (
            *SECTOR_MOMENTUM_RELATIVE_FEATURES,
            *SECTOR_RETURN_FEATURES,
            *SECTOR_BREADTH_FEATURES,
        ),
        insert_after="vnindex_return",
    )
    return [
        FeatureCase(
            name="general_sector_full",
            notes="Portable sector rank, momentum, return, alpha, and breadth context.",
            feature_columns=sector_general_full,
            removed_groups=("fast_overlap_block", "vingroup_momentum"),
        ),
        FeatureCase(
            name="general_sector_full_no_breadth",
            notes="Ablate sector breadth from the portable full sector context.",
            feature_columns=ordered_remove(sector_general_full, SECTOR_BREADTH_FEATURES),
            removed_groups=("fast_overlap_block", "vingroup_momentum", "sector_breadth"),
        ),
        FeatureCase(
            name="general_sector_full_no_return_alpha",
            notes="Ablate sector one-day return and stock-vs-sector alpha from the portable full sector context.",
            feature_columns=ordered_remove(sector_general_full, SECTOR_RETURN_FEATURES),
            removed_groups=("fast_overlap_block", "vingroup_momentum", "sector_return_alpha"),
        ),
        FeatureCase(
            name="general_sector_full_no_mom_gap",
            notes="Ablate raw sector momentum strength and stock-vs-sector momentum gap, keeping rank signals.",
            feature_columns=ordered_remove(
                sector_general_full,
                ("sector_momentum_20", "relative_sector_momentum_20"),
            ),
            removed_groups=("fast_overlap_block", "vingroup_momentum", "sector_momentum_gap"),
        ),
        FeatureCase(
            name="general_sector_full_no_rank_pct",
            notes="Ablate normalized sector momentum rank, keeping ordinal rank.",
            feature_columns=ordered_remove(sector_general_full, ("sector_momentum_rank_pct",)),
            removed_groups=("fast_overlap_block", "vingroup_momentum", "sector_rank_pct"),
        ),
        FeatureCase(
            name="general_sector_full_no_sector_rank",
            notes="Ablate ordinal sector rank, keeping normalized rank and sector context.",
            feature_columns=ordered_remove(sector_general_full, ("sector_momentum_rank",)),
            removed_groups=("fast_overlap_block", "vingroup_momentum", "sector_rank"),
        ),
    ]


def build_sector_generalized_hparam_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    no_fast_overlap = ordered_remove(base_features, FAST_OVERLAP_BLOCK)
    sector_general_full = ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        (
            *SECTOR_MOMENTUM_RELATIVE_FEATURES,
            *SECTOR_RETURN_FEATURES,
            *SECTOR_BREADTH_FEATURES,
        ),
        insert_after="vnindex_return",
    )
    base_removed = ("fast_overlap_block", "vingroup_momentum")
    return [
        FeatureCase(
            name="general_sector_full",
            notes="Portable sector-full anchor with source hyperparameters.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
        ),
        FeatureCase(
            name="general_sector_full_dropout08",
            notes="Portable sector-full with slightly stronger dropout.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides={"dropout": 0.08},
        ),
        FeatureCase(
            name="general_sector_full_dropout10",
            notes="Portable sector-full with stronger dropout.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides={"dropout": 0.10},
        ),
        FeatureCase(
            name="general_sector_full_units48_24",
            notes="Portable sector-full with smaller LSTM capacity.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides={"lstm_units": [48, 24]},
        ),
        FeatureCase(
            name="general_sector_full_sample_w2",
            notes="Portable sector-full with stronger magnitude sample weighting.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides={"sample_weight_strength": 2.0},
        ),
        FeatureCase(
            name="general_sector_full_signheavy",
            notes="Portable sector-full with stronger signed and sign auxiliary weights.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides={
                "signmag_signed_loss_weight": 2.0,
                "signmag_sign_loss_weight": 0.25,
            },
        ),
    ]


def build_general_sector_full_features(base_features: tuple[str, ...]) -> tuple[str, ...]:
    no_fast_overlap = ordered_remove(base_features, FAST_OVERLAP_BLOCK)
    return ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        (
            *SECTOR_MOMENTUM_RELATIVE_FEATURES,
            *SECTOR_RETURN_FEATURES,
            *SECTOR_BREADTH_FEATURES,
        ),
        insert_after="vnindex_return",
    )


def build_market_regime_smoke_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    sector_general_full = build_general_sector_full_features(base_features)
    base_removed = ("fast_overlap_block", "vingroup_momentum")
    smoke_overrides = {"lstm_seeds": [42, 52]}
    return [
        FeatureCase(
            name="general_sector_full_smoke",
            notes="Two-seed smoke anchor for portable sector-full candidate.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides=smoke_overrides,
        ),
        FeatureCase(
            name="general_sector_full_market_mom_smoke",
            notes="Smoke test adding general market rolling momentum context.",
            feature_columns=ordered_insert_after(
                sector_general_full,
                MARKET_MOMENTUM_FEATURES,
                insert_after="vnindex_return",
            ),
            removed_groups=base_removed,
            config_overrides=smoke_overrides,
        ),
        FeatureCase(
            name="general_sector_full_market_vol_smoke",
            notes="Smoke test adding general market rolling volatility context.",
            feature_columns=ordered_insert_after(
                sector_general_full,
                ("market_volatility_20",),
                insert_after="vnindex_return",
            ),
            removed_groups=base_removed,
            config_overrides=smoke_overrides,
        ),
        FeatureCase(
            name="general_sector_full_market_regime_smoke",
            notes="Smoke test adding general market rolling momentum, volatility, and breadth context.",
            feature_columns=ordered_insert_after(
                sector_general_full,
                MARKET_REGIME_FEATURES,
                insert_after="vnindex_return",
            ),
            removed_groups=base_removed,
            config_overrides=smoke_overrides,
        ),
    ]


def build_objective_window_smoke_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    sector_general_full = build_general_sector_full_features(base_features)
    base_removed = ("fast_overlap_block", "vingroup_momentum")
    smoke_overrides = {"lstm_seeds": [42, 52], "epochs": 18, "patience": 5}
    return [
        FeatureCase(
            name="general_sector_full_obj_anchor_smoke",
            notes="Two-seed smoke anchor for objective/window comparison.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides=smoke_overrides,
        ),
        FeatureCase(
            name="general_sector_full_w10_smoke",
            notes="Shorter sequence to test whether stale context hurts next-day return ranking.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides={**smoke_overrides, "window_size": 10},
        ),
        FeatureCase(
            name="general_sector_full_w20_smoke",
            notes="Longer sequence to test whether cycle context needs more history than 15 days.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides={**smoke_overrides, "window_size": 20},
        ),
        FeatureCase(
            name="general_sector_full_sharp_smoke",
            notes="Use rel_score_sharp to penalize wrong-sign large moves more directly.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides={
                **smoke_overrides,
                "loss": "rel_score_sharp",
                "rel_score_large_move_quantile": 0.8,
                "rel_score_directional_penalty": 0.6,
                "rel_score_confidence_penalty": 0.25,
                "rel_score_confidence_ratio": 0.2,
            },
        ),
        FeatureCase(
            name="general_sector_full_weighted_smoke",
            notes="Use rel_score_weighted to focus the q2/q8 objective on larger next-day moves without changing features.",
            feature_columns=sector_general_full,
            removed_groups=base_removed,
            config_overrides={
                **smoke_overrides,
                "loss": "rel_score_weighted",
                "rel_score_weighted_high_quantile": 0.8,
                "rel_score_weighted_high_weight": 2.0,
                "rel_score_weighted_base_weight": 1.0,
            },
        ),
    ]


def build_phase_ic_compact_smoke_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    smoke_overrides = {"lstm_seeds": [42, 52]}
    price_action_core = (
        "open_delta_1",
        "high_delta_1",
        "low_delta_1",
        "open_level_20",
        "low_level_20",
        "close_level_20",
        "intraday_return",
        "gap_open",
        "close_position",
        "bb_width",
        "momentum_20",
        "macd_hist",
    )
    phase_ic_core = (
        "open_delta_1",
        "low_delta_1",
        "volume_delta_1",
        "low_level_20",
        "close_level_20",
        "intraday_return",
        "gap_open",
        "close_position",
        "bb_width",
        "volume_ratio_20",
        "momentum_20",
        "macd_hist",
        "sector_positive_ratio",
        "sector_ad_ratio",
        "sector_momentum_rank",
    )
    phase_ic_sector_context = (
        *phase_ic_core,
        "sector_momentum_20",
        "relative_sector_momentum_20",
        "sector_return",
        "alpha_sector",
    )
    return [
        FeatureCase(
            name="phase_ic_price_action12_smoke",
            notes="Two-seed compact price-action set from stable phase IC; excludes calendar and broad macro context.",
            feature_columns=price_action_core,
            removed_groups=("phase_ic_compact", "calendar", "macro_context"),
            config_overrides=smoke_overrides,
        ),
        FeatureCase(
            name="phase_ic_core15_smoke",
            notes="Two-seed compact phase-IC core with price action, volume, momentum, and sector breadth/rank.",
            feature_columns=phase_ic_core,
            removed_groups=("phase_ic_compact", "calendar", "macro_context"),
            config_overrides=smoke_overrides,
        ),
        FeatureCase(
            name="phase_ic_sector19_smoke",
            notes="Two-seed phase-IC compact core plus portable sector momentum/return/alpha context.",
            feature_columns=phase_ic_sector_context,
            removed_groups=("phase_ic_compact", "calendar", "macro_context"),
            config_overrides=smoke_overrides,
        ),
    ]


def build_phase_ic_sector_followup_cases(base_features: tuple[str, ...]) -> list[FeatureCase]:
    phase_ic_sector19 = (
        "open_delta_1",
        "low_delta_1",
        "volume_delta_1",
        "low_level_20",
        "close_level_20",
        "intraday_return",
        "gap_open",
        "close_position",
        "bb_width",
        "volume_ratio_20",
        "momentum_20",
        "macd_hist",
        "sector_positive_ratio",
        "sector_ad_ratio",
        "sector_momentum_rank",
        "sector_momentum_20",
        "relative_sector_momentum_20",
        "sector_return",
        "alpha_sector",
    )
    phase_ic_sector17_no_level = ordered_remove(phase_ic_sector19, ("low_level_20", "close_level_20"))
    phase_ic_sector20_vol = ordered_insert_after(
        phase_ic_sector19,
        ("volatility_20",),
        insert_after="volume_ratio_20",
    )
    return [
        FeatureCase(
            name="phase_ic_sector19",
            notes="Full 3-seed follow-up of the best compact phase-IC smoke candidate.",
            feature_columns=phase_ic_sector19,
            removed_groups=("phase_ic_compact", "calendar", "macro_context"),
        ),
        FeatureCase(
            name="phase_ic_sector17_no_level",
            notes="Remove low/close level features to test whether price-level overlap hurts rel_score.",
            feature_columns=phase_ic_sector17_no_level,
            removed_groups=("phase_ic_compact", "calendar", "macro_context", "price_level_overlap"),
        ),
        FeatureCase(
            name="phase_ic_sector20_vol",
            notes="Add volatility_20 to the compact sector candidate to test recovery/downtrend sensitivity.",
            feature_columns=phase_ic_sector20_vol,
            removed_groups=("phase_ic_compact", "calendar", "macro_context"),
        ),
    ]


def resolve_feature_cases(case_set: str, base_features: tuple[str, ...]) -> list[FeatureCase]:
    if case_set == "initial":
        return build_feature_cases(base_features)
    if case_set == "fast_overlap_followup":
        return build_fast_overlap_followup_cases(base_features)
    if case_set == "sector_leadership_followup":
        return build_sector_leadership_followup_cases(base_features)
    if case_set == "sector_generalized_followup":
        return build_sector_generalized_followup_cases(base_features)
    if case_set == "sector_generalized_ablation":
        return build_sector_generalized_ablation_cases(base_features)
    if case_set == "sector_generalized_hparam":
        return build_sector_generalized_hparam_cases(base_features)
    if case_set == "market_regime_smoke":
        return build_market_regime_smoke_cases(base_features)
    if case_set == "objective_window_smoke":
        return build_objective_window_smoke_cases(base_features)
    if case_set == "phase_ic_compact_smoke":
        return build_phase_ic_compact_smoke_cases(base_features)
    if case_set == "phase_ic_sector_followup":
        return build_phase_ic_sector_followup_cases(base_features)
    raise ValueError(f"Unsupported case_set: {case_set}")


def build_lstm_units_arg(value: int | list[int]) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def build_train_command(
    python_bin: Path,
    run_name: str,
    feature_columns: tuple[str, ...],
    source_config: dict,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(python_bin),
        str(ROOT / "main.py"),
        "train",
        "--run-name",
        run_name,
        "--target-mode",
        str(source_config["target_mode"]),
        "--train-end-date",
        str(args.train_end_date),
        "--val-end-date",
        str(args.val_end_date),
        "--stocks",
        str(source_config["stocks"]),
        "--feature-columns",
        ",".join(feature_columns),
        "--window-size",
        str(source_config["window_size"]),
        "--lstm-units",
        build_lstm_units_arg(source_config["lstm_units"]),
        "--dropout",
        str(source_config["dropout"]),
        "--lr",
        str(source_config["lr"]),
        "--loss",
        str(source_config["loss"]),
        "--batch-size",
        str(source_config["batch_size"]),
        "--epochs",
        str(source_config["epochs"]),
        "--patience",
        str(source_config["patience"]),
        "--target-normalizer",
        str(source_config["target_normalizer"]),
        "--sequence-normalization",
        str(source_config.get("sequence_normalization", "none")),
        "--lstm-seeds",
        ",".join(str(seed) for seed in source_config["lstm_seeds"]),
        "--sample-weight-mode",
        str(source_config["sample_weight_mode"]),
        "--sample-weight-strength",
        str(source_config["sample_weight_strength"]),
        "--sample-weight-quantile",
        str(source_config["sample_weight_quantile"]),
        "--sample-weight-clip",
        str(source_config["sample_weight_clip"]),
        "--signmag-signed-loss-weight",
        str(source_config["signmag_signed_loss_weight"]),
        "--signmag-sign-loss-weight",
        str(source_config["signmag_sign_loss_weight"]),
        "--signmag-magnitude-loss-weight",
        str(source_config["signmag_magnitude_loss_weight"]),
    ]
    if args.allow_nonstandard_time:
        command.append("--allow-nonstandard-time")
    if source_config.get("data_path"):
        command.extend(["--data-path", str(source_config["data_path"])])
    if source_config.get("feature_phase") not in {None, "", "none"}:
        command.extend(["--feature-phase", str(source_config["feature_phase"])])
    optional_float_args = {
        "huber_delta": "--huber-delta",
        "rel_score_large_move_quantile": "--rel-score-large-move-quantile",
        "rel_score_directional_penalty": "--rel-score-directional-penalty",
        "rel_score_confidence_penalty": "--rel-score-confidence-penalty",
        "rel_score_confidence_ratio": "--rel-score-confidence-ratio",
        "rel_score_weighted_high_quantile": "--rel-score-weighted-high-quantile",
        "rel_score_weighted_high_weight": "--rel-score-weighted-high-weight",
        "rel_score_weighted_base_weight": "--rel-score-weighted-base-weight",
    }
    for config_key, cli_arg in optional_float_args.items():
        if source_config.get(config_key) is not None:
            command.extend([cli_arg, str(source_config[config_key])])
    if not bool(source_config.get("signmag_log_magnitude", True)):
        command.append("--no-signmag-log-magnitude")
    return command


def build_manifest(
    cases: list[FeatureCase],
    source_config: dict,
    batch_dir: Path,
    stamp: str,
    args: argparse.Namespace,
) -> dict:
    manifest_cases = []
    for case in cases:
        run_name = f"broad_signmag_prune_{case.name}_{stamp}"
        effective_config = {**source_config, **case.config_overrides}
        manifest_cases.append(
            {
                "case_name": case.name,
                "run_name": run_name,
                "notes": case.notes,
                "feature_count": len(case.feature_columns),
                "feature_columns": list(case.feature_columns),
                "removed_groups": list(case.removed_groups),
                "config_overrides": case.config_overrides,
                "effective_config": {
                    "window_size": effective_config["window_size"],
                    "lstm_units": effective_config["lstm_units"],
                    "dropout": effective_config["dropout"],
                    "lr": effective_config["lr"],
                    "loss": effective_config["loss"],
                    "sample_weight_mode": effective_config["sample_weight_mode"],
                    "sample_weight_strength": effective_config["sample_weight_strength"],
                    "sample_weight_quantile": effective_config["sample_weight_quantile"],
                    "sample_weight_clip": effective_config["sample_weight_clip"],
                    "signmag_signed_loss_weight": effective_config["signmag_signed_loss_weight"],
                    "signmag_sign_loss_weight": effective_config["signmag_sign_loss_weight"],
                    "signmag_magnitude_loss_weight": effective_config["signmag_magnitude_loss_weight"],
                    "rel_score_large_move_quantile": effective_config.get("rel_score_large_move_quantile"),
                    "rel_score_directional_penalty": effective_config.get("rel_score_directional_penalty"),
                    "rel_score_confidence_penalty": effective_config.get("rel_score_confidence_penalty"),
                    "rel_score_confidence_ratio": effective_config.get("rel_score_confidence_ratio"),
                    "rel_score_weighted_high_quantile": effective_config.get("rel_score_weighted_high_quantile"),
                    "rel_score_weighted_high_weight": effective_config.get("rel_score_weighted_high_weight"),
                    "rel_score_weighted_base_weight": effective_config.get("rel_score_weighted_base_weight"),
                },
            }
        )
    return {
        "batch_name": f"broad_signmag_prune_{stamp}",
        "stamp": stamp,
        "source_package_dir": str(GOLD_PACKAGE_DIR),
        "source_config_path": str(SOURCE_CONFIG_PATH),
        "batch_dir": str(batch_dir),
        "cases": manifest_cases,
        "source_config_subset": {
            "case_set": args.case_set,
            "stocks": source_config["stocks"],
            "source_train_end_date": source_config["train_end_date"],
            "source_val_end_date": source_config["val_end_date"],
            "experiment_train_end_date": args.train_end_date,
            "experiment_val_end_date": args.val_end_date,
            "allow_nonstandard_time": bool(args.allow_nonstandard_time),
            "window_size": source_config["window_size"],
            "lstm_units": source_config["lstm_units"],
            "dropout": source_config["dropout"],
            "lr": source_config["lr"],
            "loss": source_config["loss"],
            "target_normalizer": source_config["target_normalizer"],
            "sequence_normalization": source_config.get("sequence_normalization", "none"),
            "lstm_seeds": source_config["lstm_seeds"],
            "sample_weight_mode": source_config["sample_weight_mode"],
        },
    }


def write_manifest(manifest: dict, batch_dir: Path) -> Path:
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    csv_path = batch_dir / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_name", "run_name", "feature_count", "removed_groups", "feature_columns", "notes"],
        )
        writer.writeheader()
        for item in manifest["cases"]:
            writer.writerow(
                {
                    "case_name": item["case_name"],
                    "run_name": item["run_name"],
                    "feature_count": item["feature_count"],
                    "removed_groups": ",".join(item["removed_groups"]),
                    "feature_columns": ",".join(item["feature_columns"]),
                    "notes": item["notes"],
                }
            )
    return manifest_path


def run_batch(args: argparse.Namespace) -> None:
    source_config = load_json(SOURCE_CONFIG_PATH)
    base_features = tuple(str(item) for item in source_config["feature_columns"])
    cases = resolve_feature_cases(args.case_set, base_features)
    batch_dir = EXPERIMENT_REPORT_ROOT / f"broad_signmag_prune_{args.stamp}"
    log_dir = batch_dir / "logs"
    manifest = build_manifest(cases, source_config, batch_dir, args.stamp, args)
    manifest_path = write_manifest(manifest, batch_dir)

    commands = []
    for case in cases:
        run_name = f"broad_signmag_prune_{case.name}_{args.stamp}"
        effective_config = {**source_config, **case.config_overrides}
        command = build_train_command(args.python_bin, run_name, case.feature_columns, effective_config, args)
        commands.append((case, run_name, command))

    if args.print_only:
        for case, run_name, command in commands:
            print(
                json.dumps(
                    {
                        "case_name": case.name,
                        "run_name": run_name,
                        "feature_count": len(case.feature_columns),
                        "command": command,
                    },
                    ensure_ascii=True,
                )
            )
        print(json.dumps({"manifest_path": str(manifest_path), "mode": "print_only"}, indent=2))
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    for case, run_name, command in commands:
        log_path = log_dir / f"{run_name}.log"
        print(f"[RUN] {run_name} ({len(case.feature_columns)} features)")
        with log_path.open("w", encoding="utf-8") as handle:
            subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)

    summary_rows = build_summary_rows(manifest)
    write_summary_outputs(summary_rows, batch_dir)
    print(json.dumps({"manifest_path": str(manifest_path), "summary_rows": summary_rows}, indent=2))


def main(argv: list[str] | None = None) -> None:
    run_batch(parse_args(argv))


if __name__ == "__main__":
    main()
