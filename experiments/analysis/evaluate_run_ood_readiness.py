from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline.market_config import CleanConfig, get_market_config  # noqa: E402
from src.data_pipeline.quality_dataset_core import _get_event_mask, load_market_data, summarize_tickers  # noqa: E402
from src.evaluation.metric import evaluate  # noqa: E402
from src.models.architectures.plain import build_model  # noqa: E402
from src.models.architectures.signmag import build_sign_magnitude_model  # noqa: E402
from src.models.training.pipeline import augment_sequence_with_stock_identity, load_frame, split_sequence_dataset  # noqa: E402
from src.models.training.prediction import predict  # noqa: E402
from src.models.training.scalers import (  # noqa: E402
    FeatureScaler,
    LocalTargetNormalizer,
    TargetScaler,
    apply_feature_scaler,
    fit_local_target_normalizer,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)
from src.models.training.datasets import build_sequence_dataset  # noqa: E402
from src.utils.features import ensure_columns, ensure_paper_features  # noqa: E402


REPORT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "ood_readiness"
)

NEUTRAL_FEATURE_DEFAULTS: dict[str, float] = {
    "vnindex_return": 0.0,
    "market_return_5": 0.0,
    "market_return_20": 0.0,
    "market_return_60": 0.0,
    "market_volatility_20": 0.0,
    "volatility_expanding_median": 0.0,
    "market_leader_return": 0.0,
    "vingroup_momentum": 0.0,
    "a_d_ratio": 0.5,
    "market_ad_ratio_20": 0.5,
    "breadth_20": 0.5,
    "sector_return": 0.0,
    "alpha_sector": 0.0,
    "sector_positive_ratio": 0.5,
    "sector_ad_ratio": 1.0,
    "sector_momentum_20": 0.0,
    "sector_momentum_rank": 1.0,
    "sector_momentum_rank_pct": 0.0,
    "relative_sector_momentum_20": 0.0,
    "is_top_2_sector": 0.0,
    "day_of_week": 2.0,
}


@dataclass(frozen=True)
class ModelSpec:
    family: str
    model_name: str
    checkpoint_path: Path
    prediction_key: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved VN run on an out-of-distribution market universe."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--market", choices=["JP", "KR", "US"], required=True)
    parser.add_argument("--codes-path", type=Path, required=True)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional processed dataset CSV to use instead of rebuilding from market raw data.",
    )
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--model-family", choices=["best_signmag", "best_plain"], default="best_signmag")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_code_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [token.strip().upper() for token in re.split(r"[\s,]+", text) if token.strip()]


def get_ood_market_config(market: str) -> CleanConfig:
    if market != "KR":
        return get_market_config(market)
    return CleanConfig(
        market="KR",
        data_dir=ROOT / "data/raw/KR",
        output_dir=ROOT / "data/processed/assets/data_info_kr/history",
        output_prefix="kr",
        train_start_date="2020-01-01",
        min_coverage=0.97,
        recent_active_tolerance_days=15,
        drop_imputed_value_match=False,
        drop_neighbors_around_events=False,
        max_close_return_abs=0.30,
        max_adjust_return_abs=0.30,
    )


def load_feature_scaler(path: Path) -> FeatureScaler:
    payload = np.load(path, allow_pickle=True)
    feature_columns = tuple(str(item) for item in payload["feature_columns"].tolist())
    return FeatureScaler(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        feature_columns=feature_columns,
    )


def load_target_scaler(path: Path | None) -> TargetScaler | None:
    if path is None or not path.exists():
        return None
    payload = np.load(path)
    return TargetScaler(
        mean=float(np.asarray(payload["mean"]).reshape(-1)[0]),
        std=float(np.asarray(payload["std"]).reshape(-1)[0]),
    )


def resolve_model_spec(run_dir: Path, family_choice: str) -> ModelSpec:
    summary = load_json(run_dir / "reports" / "core" / "family_selection_summary.json")
    if family_choice == "best_plain":
        model_name = str((summary.get("lstm") or {}).get("best_by_val", ""))
        if not model_name:
            raise ValueError("Run has no plain-LSTM best_by_val model.")
        seed = model_name.replace("lstm_seed_", "")
        return ModelSpec(
            family="plain",
            model_name=model_name,
            checkpoint_path=run_dir / f"model_seed_{seed}.keras",
            prediction_key=None,
        )

    model_name = str((summary.get("lstm_signmag") or {}).get("best_by_val", ""))
    if not model_name:
        raise ValueError("Run has no signmag best_by_val model.")
    seed = model_name.replace("lstm_signmag_seed_", "")
    return ModelSpec(
        family="signmag",
        model_name=model_name,
        checkpoint_path=run_dir / f"model_signmag_seed_{seed}.keras",
        prediction_key="signed_prediction",
    )


def add_market_context_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    price_column = "adjust" if "adjust" in work.columns else "close"
    close_column = "close" if "close" in work.columns else price_column

    work["temp_return"] = work.groupby("code")[price_column].pct_change()
    work["return_daily"] = work.groupby("code")[close_column].pct_change()

    market_daily = (
        work.groupby("Date", sort=False)["temp_return"]
        .mean()
        .rename("vnindex_return")
        .reset_index()
    )
    market_daily["market_return_5"] = market_daily["vnindex_return"].rolling(5, min_periods=3).mean()
    market_daily["market_return_20"] = market_daily["vnindex_return"].rolling(20, min_periods=5).mean()
    market_daily["market_return_60"] = market_daily["vnindex_return"].rolling(60, min_periods=30).mean()
    market_daily["market_volatility_20"] = market_daily["vnindex_return"].rolling(20, min_periods=5).std()
    market_daily["volatility_expanding_median"] = (
        market_daily["market_volatility_20"].expanding(min_periods=60).median()
    )

    value_column = "value_match" if "value_match" in work.columns else None
    volume_column = "volume_match" if "volume_match" in work.columns else "volume" if "volume" in work.columns else None
    if value_column is not None:
        leader_traded_value = work[value_column].astype(float)
    elif volume_column is not None:
        leader_traded_value = work[close_column].astype(float).abs() * work[volume_column].astype(float)
    else:
        leader_traded_value = pd.Series(np.nan, index=work.index, dtype=float)
    leader_frame = work.loc[:, ["Date", "code", "temp_return"]].copy()
    leader_frame["leader_traded_value"] = leader_traded_value
    leader_frame["leader_liquidity_score"] = leader_frame.groupby("code", sort=False)["leader_traded_value"].transform(
        lambda series: series.shift(1).rolling(60, min_periods=20).mean()
    )
    leader_frame["leader_rank"] = leader_frame.groupby("Date", sort=False)["leader_liquidity_score"].rank(
        ascending=False,
        method="first",
    )
    market_leaders = leader_frame[
        (leader_frame["leader_rank"] <= 10) & leader_frame["leader_liquidity_score"].notna()
    ].copy()
    if market_leaders.empty:
        market_leader_return = pd.DataFrame(
            {
                "Date": market_daily["Date"],
                "market_leader_return": np.zeros(len(market_daily), dtype=np.float32),
            }
        )
    else:
        market_leaders["weighted_return"] = (
            market_leaders["temp_return"].fillna(0.0) * market_leaders["leader_liquidity_score"].fillna(0.0)
        )
        market_leader_return = (
            market_leaders.groupby("Date", sort=False)
            .agg(weighted_return=("weighted_return", "sum"), weight=("leader_liquidity_score", "sum"))
            .reset_index()
        )
        market_leader_return["market_leader_return"] = (
            market_leader_return["weighted_return"] / market_leader_return["weight"].replace(0.0, np.nan)
        )
        market_leader_return = market_leader_return[["Date", "market_leader_return"]]
    market_leader_return["vingroup_momentum"] = market_leader_return["market_leader_return"]

    breadth = (
        work.assign(
            advancing=(work["return_daily"] > 0).astype(np.int32),
            declining=(work["return_daily"] < 0).astype(np.int32),
        )
        .groupby("Date", sort=False)[["advancing", "declining"]]
        .sum()
        .reset_index()
    )
    breadth["a_d_ratio"] = breadth["advancing"] / (breadth["declining"] + 1.0)
    breadth["market_ad_ratio_20"] = breadth["a_d_ratio"].rolling(20, min_periods=5).mean()
    breadth["breadth_20"] = (
        breadth["advancing"] / (breadth["advancing"] + breadth["declining"] + 1.0)
    ).rolling(20, min_periods=10).mean()

    work = work.merge(market_daily, on="Date", how="left")
    work = work.merge(market_leader_return, on="Date", how="left")
    work = work.merge(
        breadth[["Date", "a_d_ratio", "market_ad_ratio_20", "breadth_20"]],
        on="Date",
        how="left",
    )
    market_columns = [
        "vnindex_return",
        "market_return_5",
        "market_return_20",
        "market_return_60",
        "market_volatility_20",
        "volatility_expanding_median",
        "market_leader_return",
    ]
    work[["vingroup_momentum", *market_columns]] = work[["vingroup_momentum", *market_columns]].fillna(0.0)
    work[["a_d_ratio", "market_ad_ratio_20", "breadth_20"]] = work[
        ["a_d_ratio", "market_ad_ratio_20", "breadth_20"]
    ].fillna(0.5)
    work = work.drop(columns=["temp_return", "return_daily"])
    return work


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    by_code = work.groupby("code", group_keys=False)
    next_adjust = by_code["adjust"].shift(-1)
    next_adjust_3d = by_code["adjust"].shift(-3)
    next_adjust_5d = by_code["adjust"].shift(-5)
    work["target_next_price"] = next_adjust
    work["target_next_growth_pct"] = (next_adjust / work["adjust"] - 1.0) * 100.0
    work["target_next_return"] = next_adjust / work["adjust"] - 1.0
    work["target_next_3d_return"] = next_adjust_3d / work["adjust"] - 1.0
    work["target_next_5d_return"] = next_adjust_5d / work["adjust"] - 1.0
    return work


def prepare_market_frame(raw_df: pd.DataFrame, feature_phase: str) -> pd.DataFrame:
    df = raw_df.copy()
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "exchange" not in df.columns:
        df["exchange"] = "Unknown"

    for column in ["open", "high", "low", "close", "adjust", "volume_match", "value_match"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "value_match_imputed" not in df.columns:
        df["value_match_imputed"] = 0
    df["value_match_imputed"] = pd.to_numeric(df["value_match_imputed"], errors="coerce").fillna(0).astype(int)

    required_cols = ["Date", "code", "open", "high", "low", "close", "adjust", "volume_match"]
    df["missing_required"] = df[required_cols].isna().any(axis=1)
    df["duplicate_code_date"] = df.duplicated(subset=["code", "Date"], keep=False)
    df["negative_price"] = (df[["open", "high", "low", "close", "adjust"]] <= 0).any(axis=1)
    df["negative_volume"] = df["volume_match"] < 0
    df["ohlc_invalid"] = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    df["has_hard_issue"] = (
        df["missing_required"]
        | df["duplicate_code_date"]
        | df["negative_price"]
        | df["negative_volume"]
        | df["ohlc_invalid"]
    )

    df = ensure_columns(df)
    if feature_phase in {"paper_v1", "paper_denoise_v1"}:
        df = ensure_paper_features(df)
    df = add_market_context_features(df)
    df = add_target_columns(df)
    return df.replace([np.inf, -np.inf], np.nan)


def fill_missing_feature_columns(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    filled: list[str] = []
    for column in feature_columns:
        default = NEUTRAL_FEATURE_DEFAULTS.get(column, 0.0)
        if column not in work.columns:
            work[column] = default
            filled.append(column)
            continue
        if work[column].notna().sum() == 0:
            work[column] = default
            filled.append(column)
    return work, filled


def filter_clean_universe(
    prepared: pd.DataFrame,
    config: CleanConfig,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_summary = summarize_tickers(prepared, config)
    valid_tickers = ticker_summary[
        (ticker_summary["coverage_pct"] >= config.min_coverage)
        & (ticker_summary["days_since_latest"] <= config.recent_active_tolerance_days)
    ]["code"]

    clean = prepared[prepared["code"].isin(valid_tickers)].copy()
    clean["event_row"] = _get_event_mask(clean, config)
    if config.drop_neighbors_around_events:
        by_code = clean.groupby("code")["event_row"]
        clean["drop_event_buffer"] = clean["event_row"] | by_code.shift(1).eq(True)
    else:
        clean["drop_event_buffer"] = clean["event_row"]

    keep_mask = ~clean["has_hard_issue"] & ~clean["drop_event_buffer"]
    if config.drop_imputed_value_match:
        keep_mask &= clean["value_match_imputed"].eq(0)

    clean = clean[keep_mask].copy()
    clean = clean.dropna(subset=[target_column])
    clean = clean.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)
    return clean, ticker_summary


def load_source_local_target_normalizer(run_config: dict, feature_columns: tuple[str, ...]) -> LocalTargetNormalizer | None:
    target_normalizer = run_config.get("target_normalizer")
    if not target_normalizer:
        return None

    data_path = Path(str(run_config["data_path"]))
    train_df = load_frame(data_path, str(run_config.get("stocks") or ""))
    if run_config.get("feature_phase") in {"paper_v1", "paper_denoise_v1"}:
        train_df = ensure_paper_features(train_df)

    alias = f"__target_normalizer__{target_normalizer}"
    if target_normalizer not in train_df.columns:
        raise ValueError(f"Target normalizer column missing from source dataset: {target_normalizer}")
    train_df[alias] = train_df[target_normalizer].astype(float)

    _, _, meta_all = build_sequence_dataset(
        train_df,
        feature_columns,
        str(run_config["target_column"]),
        int(run_config["window_size"]),
        extra_meta_columns=(alias,),
        sequence_normalization=str(run_config.get("sequence_normalization", "none")),
    )
    if meta_all.empty:
        return None
    train_mask = pd.to_datetime(meta_all["Date"]) <= pd.Timestamp(str(run_config["train_end_date"]))
    scale_values = meta_all.loc[train_mask, alias].to_numpy(dtype=np.float32)
    return fit_local_target_normalizer(scale_values, target_normalizer)


def build_model_for_inference(
    run_config: dict,
    spec: ModelSpec,
    num_features: int,
    target_scaler: TargetScaler | None,
    local_target_normalizer: LocalTargetNormalizer | None,
):
    common_kwargs = dict(
        window_size=int(run_config["window_size"]),
        num_features=num_features,
        lstm_units=run_config["lstm_units"],
        lr=float(run_config["lr"]),
        dropout=float(run_config["dropout"]),
        loss=str(run_config["loss"]),
        huber_delta=float(run_config.get("huber_delta", 0.01)),
        rel_score_large_move_quantile=float(run_config.get("rel_score_large_move_quantile", 0.8)),
        rel_score_directional_penalty=float(run_config.get("rel_score_directional_penalty", 0.6)),
        rel_score_confidence_penalty=float(run_config.get("rel_score_confidence_penalty", 0.35)),
        rel_score_confidence_ratio=float(run_config.get("rel_score_confidence_ratio", 0.25)),
        rel_score_weighted_high_quantile=float(run_config.get("rel_score_weighted_high_quantile", 0.8)),
        rel_score_weighted_high_weight=float(run_config.get("rel_score_weighted_high_weight", 3.0)),
        rel_score_weighted_base_weight=float(run_config.get("rel_score_weighted_base_weight", 1.0)),
    )
    if spec.family == "plain":
        return build_model(
            target_scaler=target_scaler,
            local_target_normalizer=local_target_normalizer,
            **common_kwargs,
        )
    return build_sign_magnitude_model(
        sign_loss_weight=float(run_config.get("signmag_sign_loss_weight", 0.15)),
        magnitude_loss_weight=float(run_config.get("signmag_magnitude_loss_weight", 0.35)),
        signed_loss_weight=float(run_config.get("signmag_signed_loss_weight", 1.5)),
        rank_loss_weight=float(run_config.get("signmag_rank_loss_weight", 0.0)),
        rank_temperature=float(run_config.get("signmag_rank_temperature", 1.0)),
        rank_min_group_size=int(run_config.get("signmag_rank_min_group_size", 5)),
        use_log_magnitude=bool(run_config.get("signmag_log_magnitude", True)),
        local_target_normalizer=local_target_normalizer,
        **common_kwargs,
    )


def compute_daily_rank_metrics(group: pd.DataFrame) -> dict[str, float]:
    actual = group["actual"].to_numpy(dtype=float)
    prediction = group["prediction"].to_numpy(dtype=float)
    if len(actual) < 5 or np.unique(actual).size < 2 or np.unique(prediction).size < 2:
        return {
            "spearman_ic": float("nan"),
            "top_bottom_return": float("nan"),
            "quartile_return": float("nan"),
        }

    ic = float(spearmanr(prediction, actual).correlation)
    ranks = pd.Series(prediction).rank(method="first", pct=True).to_numpy(dtype=float)
    top = actual[ranks >= 0.75]
    bottom = actual[ranks <= 0.25]
    top_bottom = float(top.mean() - bottom.mean()) if len(top) and len(bottom) else float("nan")
    return {
        "spearman_ic": ic,
        "top_bottom_return": top_bottom,
        "quartile_return": top_bottom,
    }


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return float(drawdown.min())


def summarize_daily_metrics(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    for date, group in pred_df.groupby("Date", sort=True):
        metrics = compute_daily_rank_metrics(group)
        rows.append({"Date": date, **metrics})
    daily = pd.DataFrame(rows).sort_values("Date", kind="stable")

    ic = daily["spearman_ic"].dropna().to_numpy(dtype=float)
    returns = daily["quartile_return"].dropna().to_numpy(dtype=float)
    equity = np.cumprod(1.0 + returns) if len(returns) else np.asarray([], dtype=float)
    summary = {
        "days": int(pred_df["Date"].nunique()),
        "mean_spearman_ic": float(ic.mean()) if len(ic) else float("nan"),
        "ic_t_stat": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 1 and ic.std(ddof=1) > 0 else float("nan"),
        "positive_ic_days": float((ic > 0.0).mean()) if len(ic) else float("nan"),
        "mean_top_bottom_return": float(returns.mean()) if len(returns) else float("nan"),
        "quartile_equity": float(equity[-1]) if len(equity) else float("nan"),
        "quartile_hit_rate": float((returns > 0.0).mean()) if len(returns) else float("nan"),
        "quartile_max_drawdown": max_drawdown(equity),
    }
    return daily, summary


def write_markdown(output_dir: Path, summary: dict[str, object]) -> None:
    lines = [
        "# OOD Readiness Report",
        "",
        f"- Run: `{summary['run_name']}`",
        f"- Model: `{summary['model_name']}`",
        f"- Market: `{summary['market']}`",
        f"- Universe list: `{summary['codes_path']}`",
        f"- Requested codes: `{summary['requested_codes']}`",
        f"- Clean accepted codes: `{summary['accepted_codes']}`",
        f"- Unique sectors in clean OOD frame: `{summary['sector_count']}`",
        f"- Unknown-sector share: `{float(summary['unknown_sector_share']):.1%}`",
        f"- Test rows: `{summary['test_rows']}`",
        f"- Test codes: `{summary['test_codes']}`",
        f"- Stock identity enabled: `{summary['stock_identity_enabled']}`",
        f"- Stock-identity code share known to VN model: `{float(summary['known_code_share']):.1%}`",
        f"- Stock-identity row share known to VN model: `{float(summary['known_row_share']):.1%}`",
        "",
        "## Prediction Metrics",
        f"- rel_score: `{float(summary['rel_score']):+.5f}`",
        f"- directional_accuracy: `{float(summary['directional_accuracy']):.1%}`",
        f"- error q2/q8 (E = prediction - actual): `{float(summary['error_q2']):+.5f}` / `{float(summary['error_q8']):+.5f}`",
        f"- error mean/std: `{float(summary['error_mean']):+.5f}` / `{float(summary['error_std']):.5f}`",
        "",
        "## Ranking Metrics",
        f"- mean daily Spearman IC: `{float(summary['mean_spearman_ic']):+.5f}`",
        f"- IC t-stat: `{float(summary['ic_t_stat']):+.2f}`",
        f"- positive IC days: `{float(summary['positive_ic_days']):.1%}`",
        f"- quartile equity: `{float(summary['quartile_equity']):.3f}`",
        f"- quartile hit rate: `{float(summary['quartile_hit_rate']):.1%}`",
        f"- quartile max drawdown: `{float(summary['quartile_max_drawdown']):.1%}`",
    ]
    if summary["filled_feature_columns"]:
        lines.extend(
            [
                "",
                "## Neutral Fills",
                f"- Columns filled with neutral defaults: `{', '.join(summary['filled_feature_columns'])}`",
            ]
        )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    run_config = load_json(run_dir / "reports" / "core" / "config.json")
    feature_columns = tuple(str(column) for column in run_config["feature_columns"])
    market_config = get_ood_market_config(args.market)
    requested_codes = parse_code_list(args.codes_path)

    if args.dataset_path is not None:
        clean_df = pd.read_csv(args.dataset_path)
        clean_df = clean_df[clean_df["code"].astype(str).isin(set(requested_codes))].copy()
        clean_df["Date"] = pd.to_datetime(clean_df["Date"])
        if "sector" not in clean_df.columns:
            clean_df["sector"] = "Unknown"
        if "exchange" not in clean_df.columns:
            clean_df["exchange"] = "Unknown"
        if "has_hard_issue" not in clean_df.columns:
            clean_df["has_hard_issue"] = False
        if "value_match_imputed" not in clean_df.columns:
            clean_df["value_match_imputed"] = 0
        clean_df = ensure_columns(clean_df)
        if str(run_config.get("feature_phase", "none")) in {"paper_v1", "paper_denoise_v1"}:
            clean_df = ensure_paper_features(clean_df)
        clean_df = add_market_context_features(clean_df)
        clean_df, filled_columns = fill_missing_feature_columns(clean_df, feature_columns)
        clean_df = clean_df.dropna(subset=[str(run_config["target_column"])])
        clean_df = clean_df.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)
        ticker_summary = summarize_tickers(clean_df, market_config)
    else:
        raw_df = load_market_data(market_config.data_dir)
        raw_df = raw_df[raw_df["code"].astype(str).isin(set(requested_codes))].copy()
        raw_df = raw_df.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)

        prepared = prepare_market_frame(raw_df, str(run_config.get("feature_phase", "none")))
        prepared, filled_columns = fill_missing_feature_columns(prepared, feature_columns)
        clean_df, ticker_summary = filter_clean_universe(prepared, market_config, str(run_config["target_column"]))

    scaler = load_feature_scaler(run_dir / "feature_scaler.npz")
    target_scaler = load_target_scaler(run_dir / "target_scaler.npz")
    local_target_normalizer = load_source_local_target_normalizer(run_config, scaler.feature_columns)

    alias = None
    extra_meta_columns: tuple[str, ...] = ()
    if local_target_normalizer is not None:
        alias = f"__target_normalizer__{local_target_normalizer.column}"
        clean_df[alias] = clean_df[local_target_normalizer.column].astype(float)
        extra_meta_columns = (alias,)

    scaled_df = apply_feature_scaler(clean_df, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled_df,
        scaler.feature_columns,
        str(run_config["target_column"]),
        int(run_config["window_size"]),
        extra_meta_columns=extra_meta_columns,
        sequence_normalization=str(run_config.get("sequence_normalization", "none")),
    )
    splits = split_sequence_dataset(
        x_all,
        y_all,
        meta_all,
        str(run_config["train_end_date"]),
        str(run_config["val_end_date"]),
    )
    x_test, y_test, meta_test = splits["test"]
    if len(x_test) == 0:
        raise ValueError("No OOD test sequences were available after cleaning and split.")

    use_stock_identity = bool(run_config.get("lstm_use_stock_identity", run_config.get("use_stock_identity", True)))
    stock_to_idx = {
        str(code): idx
        for idx, code in enumerate(run_config.get("lstm_stock_identity_codes", []) or [])
    }
    known_code_mask = meta_test["code"].astype(str).isin(set(stock_to_idx))
    if use_stock_identity and stock_to_idx:
        x_test = augment_sequence_with_stock_identity(x_test, meta_test, stock_to_idx)

    spec = resolve_model_spec(run_dir, args.model_family)
    model = build_model_for_inference(
        run_config=run_config,
        spec=spec,
        num_features=int(x_test.shape[2]),
        target_scaler=target_scaler if spec.family == "plain" else None,
        local_target_normalizer=local_target_normalizer if spec.family == "signmag" else None,
    )
    model.load_weights(spec.checkpoint_path)

    raw_pred = predict(model, x_test, prediction_key=spec.prediction_key)
    pred = np.asarray(raw_pred, dtype=np.float32).reshape(-1)
    if spec.family == "plain":
        pred = inverse_target_scaler_values(pred, target_scaler)
    if local_target_normalizer is not None:
        scale_values = meta_test[alias].to_numpy(dtype=np.float32) if alias is not None else None
        pred = inverse_local_target_normalizer(pred, scale_values, local_target_normalizer)

    pred_df = meta_test.copy()
    pred_df["prediction"] = pred
    pred_df["actual"] = np.asarray(y_test, dtype=np.float32).reshape(-1)
    pred_df["Date"] = pd.to_datetime(pred_df["Date"])
    pred_df = pred_df.sort_values(["Date", "code"], kind="stable").reset_index(drop=True)

    eval_df = pred_df.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)
    eval_result = evaluate(
        eval_df["prediction"].to_numpy(dtype=np.float32),
        eval_df["actual"].to_numpy(dtype=np.float32),
        group_ids=eval_df["code"].astype(str).to_numpy(),
    )
    e_values = -np.asarray(eval_result["error"], dtype=np.float32)
    daily_df, rank_summary = summarize_daily_metrics(pred_df)

    output_name = args.output_name or (
        f"{run_dir.name}__{spec.model_name}__{args.market.lower()}__{args.codes_path.stem}"
    )
    output_dir = REPORT_ROOT / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    daily_df.to_csv(output_dir / "daily_rank_metrics.csv", index=False)
    ticker_summary.to_csv(output_dir / "ticker_quality_summary.csv", index=False)

    summary = {
        "run_name": run_dir.name,
        "model_name": spec.model_name,
        "market": args.market,
        "codes_path": str(args.codes_path),
        "requested_codes": int(len(requested_codes)),
        "accepted_codes": int(clean_df["code"].astype(str).nunique()),
        "sector_count": int(clean_df["sector"].astype(str).nunique()) if "sector" in clean_df.columns else 0,
        "unknown_sector_share": float(clean_df["sector"].astype(str).eq("Unknown").mean()) if "sector" in clean_df.columns else 1.0,
        "test_rows": int(len(pred_df)),
        "test_codes": int(pred_df["code"].astype(str).nunique()),
        "stock_identity_enabled": use_stock_identity,
        "known_code_share": float(len(set(pred_df["code"].astype(str)) & set(stock_to_idx)) / max(pred_df["code"].astype(str).nunique(), 1)) if use_stock_identity else 0.0,
        "known_row_share": float(known_code_mask.mean()) if use_stock_identity else 0.0,
        "rel_score": float(eval_result["rel_score"]),
        "directional_accuracy": float(eval_result["directional_accuracy"]),
        "error_q2": float(np.quantile(e_values, 0.20)),
        "error_q8": float(np.quantile(e_values, 0.80)),
        "error_mean": float(np.mean(e_values)),
        "error_std": float(np.std(e_values)),
        "filled_feature_columns": sorted(set(filled_columns)),
        **rank_summary,
    }
    output_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    write_markdown(output_dir, summary)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    output_dir = run_analysis(parse_args(argv))
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
