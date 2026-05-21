from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.architectures.plain import build_model
from src.models.architectures.signmag import build_sign_magnitude_model
from src.models.training.datasets import build_sequence_dataset
from src.models.training.feature_normalization import add_multimarket_feature_normalization
from src.models.training.pipeline import augment_sequence_with_stock_identity
from src.models.training.prediction import predict
from src.models.training.scalers import (
    FeatureScaler,
    LocalTargetNormalizer,
    TargetScaler,
    apply_feature_scaler,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)
from src.utils.features import ensure_columns, ensure_paper_features


MAIN_NOTEBOOK_COLUMNS = ["date", "opn", "cls", "low", "high", "nsh", "vol", "adj"]
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
MARKET_LEADER_COUNT = 10
MARKET_LEADER_LIQUIDITY_WINDOW = 60
MARKET_LEADER_LIQUIDITY_MIN_PERIODS = 20


@dataclass(frozen=True)
class BundleModelSpec:
    family: str
    checkpoint_name: str
    prediction_key: str | int | tuple[int, ...] | None
    resolved_model_name: str
    source_alias: str | None


@dataclass(frozen=True)
class BundleMetadata:
    config: dict[str, object]
    feature_scaler: FeatureScaler
    target_scaler: TargetScaler | None
    local_target_normalizer: LocalTargetNormalizer | None
    model_spec: BundleModelSpec


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feature_scaler(path: Path) -> FeatureScaler:
    payload = np.load(path, allow_pickle=True)
    feature_columns = tuple(str(item) for item in payload["feature_columns"].tolist())
    return FeatureScaler(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        feature_columns=feature_columns,
    )


def _load_target_scaler(path: Path | None) -> TargetScaler | None:
    if path is None or not path.exists():
        return None
    payload = np.load(path)
    return TargetScaler(
        mean=float(np.asarray(payload["mean"]).reshape(-1)[0]),
        std=float(np.asarray(payload["std"]).reshape(-1)[0]),
    )


def _load_local_target_normalizer(payload: object) -> LocalTargetNormalizer | None:
    if not isinstance(payload, dict):
        return None
    column = payload.get("column")
    floor = payload.get("floor")
    if not column or floor is None:
        return None
    return LocalTargetNormalizer(column=str(column), floor=float(floor))


def load_bundle_metadata(package_dir: str | Path) -> BundleMetadata:
    root = Path(package_dir).resolve()
    manifest = _load_json(root / "core" / "bundle_manifest.json")
    config = _load_json(root / "core" / "source_config.json")
    model_spec = BundleModelSpec(
        family=str(manifest["model_family"]),
        checkpoint_name=str(manifest["checkpoint_name"]),
        prediction_key=manifest.get("prediction_key"),
        resolved_model_name=str(manifest["resolved_model_name"]),
        source_alias=str(manifest["source_alias"]) if manifest.get("source_alias") else None,
    )
    feature_scaler = _load_feature_scaler(root / "model" / "feature_scaler.npz")
    target_scaler = _load_target_scaler(root / "model" / "target_scaler.npz")
    local_target_normalizer = _load_local_target_normalizer(manifest.get("local_target_normalizer"))
    return BundleMetadata(
        config=config,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        local_target_normalizer=local_target_normalizer,
        model_spec=model_spec,
    )


def _coerce_main_npy_frame(array: np.ndarray, code: str, market: str, sector: str, exchange: str) -> pd.DataFrame:
    df = pd.DataFrame(array, columns=MAIN_NOTEBOOK_COLUMNS[: array.shape[1]])
    rename_map = {
        "opn": "open",
        "cls": "close",
        "vol": "volume_match",
        "adj": "adjust",
    }
    df = df.rename(columns=rename_map)
    if "date" in df.columns:
        date_text = df["date"].astype(str).str.replace(r"\.0$", "", regex=True)
        df["Date"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
        df = df.drop(columns=["date"])
    else:
        raise ValueError("Input .npy must include a 'date' column in Main.ipynb format.")

    numeric_columns = ["open", "high", "low", "close", "adjust", "volume_match"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "adjust" not in df.columns and "close" in df.columns:
        df["adjust"] = df["close"]
    if "volume_match" not in df.columns and "nsh" in df.columns:
        df["volume_match"] = pd.to_numeric(df["nsh"], errors="coerce")
    if "value_match" not in df.columns:
        df["value_match"] = df["volume_match"]
    if "value_match_imputed" not in df.columns:
        df["value_match_imputed"] = 0
    df["code"] = str(code)
    df["market"] = str(market)
    df["sector"] = str(sector)
    df["exchange"] = str(exchange)
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def build_panel_from_main_npy(
    npy_path: str | Path,
    code: str = "COLAB_STOCK",
    market: str = "VN",
    sector: str = "Unknown",
    exchange: str = "Unknown",
) -> pd.DataFrame:
    array = np.load(Path(npy_path), allow_pickle=True)
    if array.ndim != 2:
        raise ValueError("Input .npy must be a 2D array shaped like Main.ipynb sample data.")
    return _coerce_main_npy_frame(array, code=code, market=market, sector=sector, exchange=exchange)


def build_panel_from_npy_directory(
    data_dir: str | Path,
    market: str = "VN",
    sector: str = "Unknown",
    exchange: str = "Unknown",
) -> pd.DataFrame:
    root = Path(data_dir).resolve()
    npy_paths = sorted(root.glob("*.npy"))
    if not npy_paths:
        raise FileNotFoundError(f"No .npy files found in {root}")
    frames = [
        build_panel_from_main_npy(path, code=path.stem, market=market, sector=sector, exchange=exchange)
        for path in npy_paths
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["code", "Date"], kind="stable").reset_index(drop=True)


def _add_market_context_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    price_column = "adjust" if "adjust" in work.columns else "close"
    close_column = "close" if "close" in work.columns else price_column

    market_group_columns = ["market"] if "market" in work.columns else []
    market_date_group_columns = [*market_group_columns, "Date"]
    native_code_column = "native_code" if "native_code" in work.columns else "code"

    def grouped_rolling_mean(frame: pd.DataFrame, column: str, window: int, min_periods: int) -> pd.Series:
        if not market_group_columns:
            return frame[column].rolling(window, min_periods=min_periods).mean()
        return frame.groupby(market_group_columns, sort=False)[column].transform(
            lambda series: series.rolling(window, min_periods=min_periods).mean()
        )

    def grouped_rolling_std(frame: pd.DataFrame, column: str, window: int, min_periods: int) -> pd.Series:
        if not market_group_columns:
            return frame[column].rolling(window, min_periods=min_periods).std()
        return frame.groupby(market_group_columns, sort=False)[column].transform(
            lambda series: series.rolling(window, min_periods=min_periods).std()
        )

    def grouped_expanding_median(frame: pd.DataFrame, column: str, min_periods: int) -> pd.Series:
        if not market_group_columns:
            return frame[column].expanding(min_periods=min_periods).median()
        return frame.groupby(market_group_columns, sort=False)[column].transform(
            lambda series: series.expanding(min_periods=min_periods).median()
        )

    work["temp_return"] = work.groupby("code")[price_column].pct_change()
    work["return_daily"] = work.groupby("code")[close_column].pct_change()

    market_daily = (
        work.groupby(market_date_group_columns, sort=False)["temp_return"]
        .mean()
        .rename("vnindex_return")
        .reset_index()
    )
    market_daily["market_return_5"] = grouped_rolling_mean(market_daily, "vnindex_return", 5, 3)
    market_daily["market_return_20"] = grouped_rolling_mean(market_daily, "vnindex_return", 20, 5)
    market_daily["market_return_60"] = grouped_rolling_mean(market_daily, "vnindex_return", 60, 30)
    market_daily["market_volatility_20"] = grouped_rolling_std(market_daily, "vnindex_return", 20, 5)
    market_daily["volatility_expanding_median"] = grouped_expanding_median(market_daily, "market_volatility_20", 60)

    value_column = "value_match" if "value_match" in work.columns else None
    volume_column = "volume_match" if "volume_match" in work.columns else "volume" if "volume" in work.columns else None
    leader_key_columns = [*market_group_columns, native_code_column]
    if value_column is not None:
        leader_traded_value = work[value_column].astype(float)
    elif volume_column is not None:
        leader_traded_value = work[close_column].astype(float).abs() * work[volume_column].astype(float)
    else:
        leader_traded_value = pd.Series(np.nan, index=work.index, dtype=float)
    leader_frame = work.loc[:, [*market_group_columns, "Date", native_code_column, "temp_return"]].copy()
    leader_frame["leader_traded_value"] = leader_traded_value
    leader_frame["leader_liquidity_score"] = leader_frame.groupby(leader_key_columns, sort=False)[
        "leader_traded_value"
    ].transform(
        lambda series: series.shift(1).rolling(
            MARKET_LEADER_LIQUIDITY_WINDOW,
            min_periods=MARKET_LEADER_LIQUIDITY_MIN_PERIODS,
        ).mean()
    )
    leader_frame["leader_rank"] = leader_frame.groupby(market_date_group_columns, sort=False)[
        "leader_liquidity_score"
    ].rank(ascending=False, method="first")
    market_leaders = leader_frame[
        (leader_frame["leader_rank"] <= MARKET_LEADER_COUNT)
        & leader_frame["leader_liquidity_score"].notna()
    ].copy()
    if market_leaders.empty:
        market_leader_return = pd.DataFrame(
            {
                **({column: market_daily[column] for column in market_group_columns} if market_group_columns else {}),
                "Date": market_daily["Date"],
                "market_leader_return": np.zeros(len(market_daily), dtype=np.float32),
            }
        )
    else:
        market_leaders["weighted_return"] = (
            market_leaders["temp_return"].fillna(0.0) * market_leaders["leader_liquidity_score"].fillna(0.0)
        )
        market_leader_return = (
            market_leaders.groupby(market_date_group_columns, sort=False)
            .agg(
                weighted_return=("weighted_return", "sum"),
                weight=("leader_liquidity_score", "sum"),
            )
            .reset_index()
        )
        market_leader_return["market_leader_return"] = (
            market_leader_return["weighted_return"] / market_leader_return["weight"].replace(0.0, np.nan)
        )
        market_leader_return = market_leader_return[[*market_date_group_columns, "market_leader_return"]]
    market_leader_return["vingroup_momentum"] = market_leader_return["market_leader_return"]

    breadth = (
        work.assign(
            advancing=(work["return_daily"] > 0).astype(np.int32),
            declining=(work["return_daily"] < 0).astype(np.int32),
        )
        .groupby(market_date_group_columns, sort=False)[["advancing", "declining"]]
        .sum()
        .reset_index()
    )
    breadth["a_d_ratio"] = breadth["advancing"] / (breadth["declining"] + 1.0)
    breadth["market_ad_ratio_20"] = grouped_rolling_mean(breadth, "a_d_ratio", 20, 5)
    breadth["breadth_20"] = grouped_rolling_mean(
        breadth.assign(breadth_daily=breadth["advancing"] / (breadth["advancing"] + breadth["declining"] + 1.0)),
        "breadth_daily",
        20,
        10,
    )

    merge_keys = market_date_group_columns
    work = work.merge(market_daily, on=merge_keys, how="left")
    work = work.merge(market_leader_return, on=merge_keys, how="left")
    work = work.merge(breadth[[*merge_keys, "a_d_ratio", "market_ad_ratio_20", "breadth_20"]], on=merge_keys, how="left")

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
    return work.drop(columns=["temp_return", "return_daily"])


def _add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def _fill_missing_feature_columns(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    work = df.copy()
    for column in feature_columns:
        default = NEUTRAL_FEATURE_DEFAULTS.get(column, 0.0)
        if column not in work.columns:
            work[column] = default
            continue
        if work[column].notna().sum() == 0:
            work[column] = default
    return work


def _feature_normalization_base_columns(config: dict[str, object]) -> tuple[str, ...]:
    columns = config.get("feature_normalization_base_columns")
    if isinstance(columns, list) and columns:
        return tuple(str(column) for column in columns)
    metadata = config.get("feature_normalization_metadata")
    if isinstance(metadata, dict):
        metadata_columns = metadata.get("base_feature_columns")
        if isinstance(metadata_columns, list) and metadata_columns:
            return tuple(str(column) for column in metadata_columns)
    return ()


def _apply_configured_feature_normalization(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    mode = str(config.get("feature_normalization_mode", "none"))
    if mode == "none":
        return df
    if mode != "multimarket_v1":
        raise ValueError(f"Unsupported feature_normalization_mode in bundle: {mode}")

    base_columns = _feature_normalization_base_columns(config)
    if not base_columns:
        raise ValueError("Bundle config is missing feature_normalization_base_columns.")

    prepared = _fill_missing_feature_columns(df, base_columns)
    result = add_multimarket_feature_normalization(
        prepared,
        base_columns,
        rolling_window=int(config.get("feature_normalization_window", 60)),
        min_periods=int(config.get("feature_normalization_min_periods", 20)),
        include_cross_sectional_z=bool(config.get("feature_normalization_include_cs_z", True)),
        include_cross_sectional_rank=bool(config.get("feature_normalization_include_cs_rank", True)),
        strict_past=bool(config.get("feature_normalization_strict_past", True)),
    )
    return result.frame


def _build_stock_identity_map(config: dict[str, object]) -> dict[str, int]:
    codes = [str(code) for code in config.get("lstm_stock_identity_codes", [])]
    return {code: idx for idx, code in enumerate(codes)}


def _build_model_for_inference(
    metadata: BundleMetadata,
    num_features: int,
):
    config = metadata.config
    spec = metadata.model_spec
    common_kwargs = dict(
        window_size=int(config["window_size"]),
        num_features=num_features,
        lstm_units=config["lstm_units"],
        lr=float(config["lr"]),
        dropout=float(config["dropout"]),
        loss=str(config["loss"]),
        huber_delta=float(config.get("huber_delta", 0.01)),
        rel_score_large_move_quantile=float(config.get("rel_score_large_move_quantile", 0.8)),
        rel_score_directional_penalty=float(config.get("rel_score_directional_penalty", 0.6)),
        rel_score_confidence_penalty=float(config.get("rel_score_confidence_penalty", 0.35)),
        rel_score_confidence_ratio=float(config.get("rel_score_confidence_ratio", 0.25)),
        rel_score_weighted_high_quantile=float(config.get("rel_score_weighted_high_quantile", 0.8)),
        rel_score_weighted_high_weight=float(config.get("rel_score_weighted_high_weight", 3.0)),
        rel_score_weighted_base_weight=float(config.get("rel_score_weighted_base_weight", 1.0)),
    )
    if spec.family == "plain":
        return build_model(
            target_scaler=metadata.target_scaler,
            local_target_normalizer=metadata.local_target_normalizer,
            **common_kwargs,
        )
    if spec.family == "signmag":
        return build_sign_magnitude_model(
            sign_loss_weight=float(config.get("signmag_sign_loss_weight", 0.15)),
            magnitude_loss_weight=float(config.get("signmag_magnitude_loss_weight", 0.35)),
            signed_loss_weight=float(config.get("signmag_signed_loss_weight", 1.5)),
            rank_loss_weight=float(config.get("signmag_rank_loss_weight", 0.0)),
            rank_temperature=float(config.get("signmag_rank_temperature", 1.0)),
            rank_min_group_size=int(config.get("signmag_rank_min_group_size", 5)),
            use_log_magnitude=bool(config.get("signmag_log_magnitude", True)),
            local_target_normalizer=metadata.local_target_normalizer,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported bundle family: {spec.family}")


def _postprocess_prediction(
    prediction: np.ndarray,
    meta: pd.DataFrame,
    target_scaler: TargetScaler | None,
    local_target_normalizer: LocalTargetNormalizer | None,
) -> np.ndarray:
    work = inverse_target_scaler_values(np.asarray(prediction, dtype=np.float32).reshape(-1), target_scaler)
    if local_target_normalizer is None:
        return work
    alias = f"__target_normalizer__{local_target_normalizer.column}"
    scale_values = meta[alias].to_numpy(dtype=np.float32) if alias in meta.columns else None
    return inverse_local_target_normalizer(work, scale_values, local_target_normalizer).reshape(-1)


def prepare_panel_for_bundle(panel_df: pd.DataFrame, metadata: BundleMetadata) -> pd.DataFrame:
    config = metadata.config
    work = panel_df.copy()
    for required_column in ["Date", "code", "open", "high", "low", "close", "adjust", "volume_match"]:
        if required_column not in work.columns:
            raise ValueError(f"Input panel is missing required column: {required_column}")
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)
    work["value_match_imputed"] = pd.to_numeric(work.get("value_match_imputed", 0), errors="coerce").fillna(0).astype(int)
    if "sector" not in work.columns:
        work["sector"] = "Unknown"
    if "exchange" not in work.columns:
        work["exchange"] = "Unknown"
    if "market" not in work.columns:
        work["market"] = str(config.get("market", "VN"))

    work = ensure_columns(work)
    if str(config.get("feature_phase", "none")) in {"paper_v1", "paper_denoise_v1"}:
        work = ensure_paper_features(work)
    work = _add_market_context_features(work)
    work = _add_target_columns(work)
    work = _apply_configured_feature_normalization(work, config)
    work = _fill_missing_feature_columns(work, metadata.feature_scaler.feature_columns)
    return work.replace([np.inf, -np.inf], np.nan)


def predict_bundle_on_panel(package_dir: str | Path, panel_df: pd.DataFrame) -> pd.DataFrame:
    metadata = load_bundle_metadata(package_dir)
    prepared = prepare_panel_for_bundle(panel_df, metadata)

    alias = None
    extra_meta_columns: tuple[str, ...] = ()
    if metadata.local_target_normalizer is not None:
        alias = f"__target_normalizer__{metadata.local_target_normalizer.column}"
        if metadata.local_target_normalizer.column not in prepared.columns:
            raise ValueError(f"Missing target normalizer column: {metadata.local_target_normalizer.column}")
        prepared[alias] = prepared[metadata.local_target_normalizer.column].astype(float)
        extra_meta_columns = (alias,)

    scaled = apply_feature_scaler(prepared, metadata.feature_scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled,
        metadata.feature_scaler.feature_columns,
        str(metadata.config["target_column"]),
        int(metadata.config["window_size"]),
        extra_meta_columns=extra_meta_columns,
        sequence_normalization=str(metadata.config.get("sequence_normalization", "none")),
    )
    if len(x_all) == 0:
        raise ValueError("Not enough rows to build sequences for this bundle.")

    stock_to_idx = _build_stock_identity_map(metadata.config)
    if stock_to_idx:
        x_all = augment_sequence_with_stock_identity(x_all, meta_all, stock_to_idx)

    model = _build_model_for_inference(metadata, num_features=int(x_all.shape[2]))
    model.load_weights(Path(package_dir).resolve() / "model" / metadata.model_spec.checkpoint_name)
    raw_pred = predict(model, x_all, prediction_key=metadata.model_spec.prediction_key)
    pred = _postprocess_prediction(raw_pred, meta_all, metadata.target_scaler, metadata.local_target_normalizer)

    out = meta_all.copy()
    out["prediction"] = pred.astype(np.float32)
    out["actual"] = y_all.astype(np.float32)
    out["model_name"] = metadata.model_spec.resolved_model_name
    out["model_family"] = metadata.model_spec.family
    if "close" in prepared.columns:
        close_map = prepared.loc[:, ["code", "Date", "close"]].rename(columns={"close": "current_close"})
        out = out.merge(close_map, on=["code", "Date"], how="left")
        out["predicted_next_close"] = out["current_close"] * (1.0 + out["prediction"])
        out["actual_next_close"] = out["current_close"] * (1.0 + out["actual"])
    return out.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)


def predict_bundle_on_main_npy(
    package_dir: str | Path,
    npy_path: str | Path,
    code: str = "COLAB_STOCK",
    market: str = "VN",
    sector: str = "Unknown",
    exchange: str = "Unknown",
) -> pd.DataFrame:
    panel_df = build_panel_from_main_npy(npy_path, code=code, market=market, sector=sector, exchange=exchange)
    return predict_bundle_on_panel(package_dir, panel_df)
