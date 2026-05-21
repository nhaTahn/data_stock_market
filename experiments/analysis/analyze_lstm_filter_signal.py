from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metric import evaluate  # noqa: E402
from src.models.training import (  # noqa: E402
    apply_feature_scaler,
    fit_local_target_normalizer,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset  # noqa: E402
from src.models.training.pipeline import load_frame  # noqa: E402
from src.models.training.prediction import predict  # noqa: E402
from src.models.training.scalers import FeatureScaler, TargetScaler  # noqa: E402
from src.models.selection import filter_signal as filter_selection  # noqa: E402
from src.utils.features import ensure_paper_features  # noqa: E402


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "filter_signal"
DEFAULT_RUN = "broad_signmag_portable_no_identity_20260428_allvn_r01"
DEFAULT_MODEL = "lstm_seed_52"
DEFAULT_FILTER_FEATURES = (
    "base_prediction",
    "base_abs_prediction",
    "base_prediction_rank_pct",
    "base_prediction_zscore",
    "market_proxy_return_1",
    "market_proxy_return_5",
    "market_proxy_return_20",
    "market_proxy_return_60",
    "market_proxy_volatility_20",
    "market_proxy_volatility_ratio_20",
    "market_breadth_20",
    "market_ad_ratio_20",
    "market_proxy_drawdown_60",
    "market_liquidity_zscore_20",
    "market_leader_return",
    "ichi_8_22_44_tenkan_kijun_gap",
)
MARKET_RETURN_SOURCE_CANDIDATES = (
    "market_proxy_return_1",
    "market_index_return",
    "index_return",
    "benchmark_return",
    "vnindex_return",
)


@dataclass(frozen=True)
class FilterConfig:
    filter_window: int = 10
    min_names_per_day: int = 20
    tradeable_abs_quantile: float = 0.60
    epochs: int = 8
    patience: int = 2
    batch_size: int = 256
    seed: int = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a portable LSTM confidence filter on top of an existing base LSTM prediction stream."
    )
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stamp", default="20260508_r01")
    parser.add_argument("--output-name", default="portable_lstm_filter_signal")
    parser.add_argument("--filter-window", type=int, default=FilterConfig.filter_window)
    parser.add_argument("--tradeable-abs-quantile", type=float, default=FilterConfig.tradeable_abs_quantile)
    parser.add_argument("--epochs", type=int, default=FilterConfig.epochs)
    parser.add_argument("--patience", type=int, default=FilterConfig.patience)
    parser.add_argument("--batch-size", type=int, default=FilterConfig.batch_size)
    parser.add_argument("--seed", type=int, default=FilterConfig.seed)
    parser.add_argument("--market-leader-count", type=int, default=10)
    parser.add_argument("--market-leader-liquidity-window", type=int, default=60)
    parser.add_argument("--market-leader-liquidity-min-periods", type=int, default=20)
    parser.add_argument(
        "--exclude-filter-feature",
        action="append",
        default=[],
        help="Filter feature to exclude from this artifact. Can be passed multiple times.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_scaler(path: Path) -> FeatureScaler:
    payload = np.load(path, allow_pickle=True)
    return FeatureScaler(
        mean=payload["mean"].astype(float),
        std=payload["std"].astype(float),
        feature_columns=tuple(str(item) for item in payload["feature_columns"].tolist()),
    )


def load_target_scaler(path: Path) -> TargetScaler:
    payload = np.load(path, allow_pickle=True)
    return TargetScaler(mean=float(payload["mean"].reshape(-1)[0]), std=float(payload["std"].reshape(-1)[0]))


def model_path_for(run_dir: Path, model_name: str) -> Path:
    if model_name.startswith("lstm_seed_"):
        seed = model_name.removeprefix("lstm_seed_")
        return run_dir / f"model_seed_{seed}.keras"
    if model_name.startswith("lstm_signmag_seed_"):
        seed = model_name.removeprefix("lstm_signmag_seed_")
        return run_dir / f"model_signmag_seed_{seed}.keras"
    if model_name == "lstm_best_by_val":
        family = load_json(run_dir / "reports" / "core" / "family_selection_summary.json")
        return model_path_for(run_dir, str(family["lstm"]["best_by_val"]))
    if model_name == "lstm_signmag_best_by_val":
        family = load_json(run_dir / "reports" / "core" / "family_selection_summary.json")
        return model_path_for(run_dir, str(family["lstm_signmag"]["best_by_val"]))
    raise ValueError(f"Unsupported model name: {model_name}")


def load_model_compat(path: Path) -> keras.Model:
    original_dense_from_config = keras.layers.Dense.from_config

    @classmethod
    def dense_from_config(cls: type[keras.layers.Dense], config: dict) -> keras.layers.Dense:
        clean_config = dict(config)
        clean_config.pop("quantization_config", None)
        return original_dense_from_config(clean_config)

    keras.layers.Dense.from_config = dense_from_config
    try:
        return keras.models.load_model(
            path,
            compile=False,
            custom_objects={"expm1": tf.math.expm1},
            safe_mode=False,
        )
    finally:
        keras.layers.Dense.from_config = original_dense_from_config


def market_proxy_return_source(df: pd.DataFrame) -> str | None:
    for column in MARKET_RETURN_SOURCE_CANDIDATES:
        if column in df.columns:
            return column
    return None


def add_portable_market_context(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    source_column = market_proxy_return_source(out)
    if source_column is not None:
        out["market_proxy_return_1"] = out[source_column].astype(float)
        market_proxy_source = source_column
    else:
        out["market_proxy_return_1"] = out.groupby("Date")["adjust_return"].transform("mean")
        market_proxy_source = "cross_section_equal_weight_adjust_return"
    derived_market_columns = [
        "market_proxy_return_5",
        "market_proxy_return_20",
        "market_proxy_return_60",
        "market_proxy_volatility_20",
        "market_proxy_volatility_ratio_20",
        "market_breadth_20",
        "market_ad_ratio_20",
        "market_proxy_drawdown_60",
        "market_liquidity_zscore_20",
    ]
    out = out.drop(columns=[column for column in derived_market_columns if column in out.columns])

    market_daily = (
        out.groupby("Date", sort=True)
        .agg(
            market_proxy_return_1=("market_proxy_return_1", "first"),
            market_liquidity=("volume_match", "sum") if "volume_match" in out.columns else ("code", "count"),
        )
        .reset_index()
        .sort_values("Date", kind="stable")
    )
    market_daily["market_proxy_return_5"] = market_daily["market_proxy_return_1"].rolling(5, min_periods=3).mean()
    market_daily["market_proxy_return_20"] = market_daily["market_proxy_return_1"].rolling(20, min_periods=5).mean()
    market_daily["market_proxy_return_60"] = market_daily["market_proxy_return_1"].rolling(60, min_periods=30).mean()
    market_daily["market_proxy_volatility_20"] = market_daily["market_proxy_return_1"].rolling(20, min_periods=5).std()
    vol_median = market_daily["market_proxy_volatility_20"].expanding(min_periods=60).median()
    market_daily["market_proxy_volatility_ratio_20"] = market_daily["market_proxy_volatility_20"] / vol_median.replace(0.0, np.nan)
    market_index = (1.0 + market_daily["market_proxy_return_1"].fillna(0.0)).cumprod()
    market_daily["market_proxy_drawdown_60"] = market_index / market_index.rolling(60, min_periods=20).max() - 1.0
    liquidity_mean = market_daily["market_liquidity"].rolling(20, min_periods=5).mean()
    liquidity_std = market_daily["market_liquidity"].rolling(20, min_periods=5).std()
    market_daily["market_liquidity_zscore_20"] = (
        (market_daily["market_liquidity"] - liquidity_mean) / liquidity_std.replace(0.0, np.nan)
    )

    if "a_d_ratio" in out.columns:
        breadth = out.groupby("Date", sort=True)["a_d_ratio"].first().rename("market_ad_ratio_raw").reset_index()
    else:
        daily_ret = out.groupby("code")["adjust"].pct_change() if "adjust" in out.columns else out.groupby("code")["close"].pct_change()
        tmp = out.assign(_ret=daily_ret, _adv=(daily_ret > 0).astype(float), _dec=(daily_ret < 0).astype(float))
        breadth = (
            tmp.groupby("Date", sort=True)
            .agg(market_advancers=("_adv", "sum"), market_decliners=("_dec", "sum"))
            .reset_index()
        )
        breadth["market_ad_ratio_raw"] = breadth["market_advancers"] / (breadth["market_decliners"] + 1.0)

    if "breadth_20" in out.columns:
        breadth["market_breadth_20"] = out.groupby("Date", sort=True)["breadth_20"].first().to_numpy(dtype=float)
    else:
        if "market_advancers" not in breadth.columns:
            daily_ret = out.groupby("code")["adjust"].pct_change() if "adjust" in out.columns else out.groupby("code")["close"].pct_change()
            tmp = out.assign(_ret=daily_ret, _adv=(daily_ret > 0).astype(float), _dec=(daily_ret < 0).astype(float))
            adv_dec = tmp.groupby("Date", sort=True).agg(market_advancers=("_adv", "sum"), market_decliners=("_dec", "sum")).reset_index()
            breadth = breadth.merge(adv_dec, on="Date", how="left")
        breadth_daily = breadth["market_advancers"] / (breadth["market_advancers"] + breadth["market_decliners"] + 1.0)
        breadth["market_breadth_20"] = breadth_daily.rolling(20, min_periods=10).mean()
    breadth["market_ad_ratio_20"] = breadth["market_ad_ratio_raw"].rolling(20, min_periods=5).mean()

    out = out.merge(
        market_daily[
            [
                "Date",
                "market_proxy_return_5",
                "market_proxy_return_20",
                "market_proxy_return_60",
                "market_proxy_volatility_20",
                "market_proxy_volatility_ratio_20",
                "market_proxy_drawdown_60",
                "market_liquidity_zscore_20",
            ]
        ],
        on="Date",
        how="left",
    )
    out = out.merge(breadth[["Date", "market_breadth_20", "market_ad_ratio_20"]], on="Date", how="left")
    fill_zero = [
        "market_proxy_return_1",
        "market_proxy_return_5",
        "market_proxy_return_20",
        "market_proxy_return_60",
        "market_proxy_volatility_20",
        "market_proxy_drawdown_60",
        "market_liquidity_zscore_20",
    ]
    out[fill_zero] = out[fill_zero].fillna(0.0)
    out["market_proxy_volatility_ratio_20"] = out["market_proxy_volatility_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out[["market_breadth_20", "market_ad_ratio_20"]] = out[["market_breadth_20", "market_ad_ratio_20"]].fillna(0.5)
    return out, market_proxy_source


def prepare_frame(
    run_config: dict,
    *,
    market_leader_count: int,
    market_leader_liquidity_window: int,
    market_leader_liquidity_min_periods: int,
) -> tuple[pd.DataFrame, str]:
    df = load_frame(
        Path(str(run_config["data_path"])),
        run_config.get("stocks"),
        market_leader_count=market_leader_count,
        market_leader_liquidity_window=market_leader_liquidity_window,
        market_leader_liquidity_min_periods=market_leader_liquidity_min_periods,
    )
    if run_config.get("feature_phase") in {"paper_v1", "paper_denoise_v1"}:
        df = ensure_paper_features(df)
    df, market_proxy_source = add_portable_market_context(df)
    return df.replace([np.inf, -np.inf], np.nan), market_proxy_source


def build_base_prediction_frame(
    run_dir: Path,
    run_config: dict,
    model_name: str,
    *,
    market_leader_count: int,
    market_leader_liquidity_window: int,
    market_leader_liquidity_min_periods: int,
) -> tuple[pd.DataFrame, str]:
    feature_scaler = load_feature_scaler(run_dir / "feature_scaler.npz")
    target_scaler = load_target_scaler(run_dir / "target_scaler.npz")
    raw_df, market_proxy_source = prepare_frame(
        run_config,
        market_leader_count=market_leader_count,
        market_leader_liquidity_window=market_leader_liquidity_window,
        market_leader_liquidity_min_periods=market_leader_liquidity_min_periods,
    )
    scaled_df = apply_feature_scaler(raw_df, feature_scaler)

    target_normalizer_alias = None
    extra_meta_columns = tuple(
        column
        for column in (
            "market_proxy_return_1",
            "market_proxy_return_5",
            "market_proxy_return_20",
            "market_proxy_return_60",
            "market_proxy_volatility_20",
            "market_proxy_volatility_ratio_20",
            "market_breadth_20",
            "market_ad_ratio_20",
            "market_proxy_drawdown_60",
            "market_liquidity_zscore_20",
            "market_leader_return",
            "ichi_8_22_44_tenkan_kijun_gap",
        )
        if column in scaled_df.columns
    )
    if run_config.get("target_normalizer"):
        target_normalizer_alias = f"__target_normalizer__{run_config['target_normalizer']}"
        scaled_df[target_normalizer_alias] = raw_df[str(run_config["target_normalizer"])].astype(float)
        extra_meta_columns = (target_normalizer_alias, *extra_meta_columns)

    x_all, y_all, meta_all = build_sequence_dataset(
        scaled_df,
        tuple(str(item) for item in run_config["feature_columns"]),
        str(run_config["target_column"]),
        int(run_config["window_size"]),
        extra_meta_columns=extra_meta_columns,
        sequence_normalization=str(run_config.get("sequence_normalization", "none")),
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, str(run_config["train_end_date"]), str(run_config["val_end_date"]))
    model = load_model_compat(model_path_for(run_dir, model_name))
    prediction_key = "signed_prediction" if model_name.startswith("lstm_signmag") else None
    frames: list[pd.DataFrame] = []

    local_target_normalizer = None
    if target_normalizer_alias:
        train_scale_values = splits["train"][2][target_normalizer_alias].to_numpy(dtype=np.float32)
        local_target_normalizer = fit_local_target_normalizer(train_scale_values, str(run_config["target_normalizer"]))

    for split_name in ("train", "val"):
        x_split, y_split, meta_split = splits[split_name]
        pred = predict(model, x_split, prediction_key=prediction_key)
        if prediction_key is None:
            pred = inverse_target_scaler_values(pred, target_scaler)
        if local_target_normalizer is not None and target_normalizer_alias is not None:
            pred = inverse_local_target_normalizer(
                pred,
                meta_split[target_normalizer_alias].to_numpy(dtype=np.float32),
                local_target_normalizer,
            )
        frame = meta_split.copy()
        frame["split"] = split_name
        frame["base_prediction"] = pred
        frame["actual"] = y_split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), market_proxy_source


def align_signal_actual(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (code, split), group in frame.sort_values(["code", "split", "Date"], kind="stable").groupby(["code", "split"], sort=False):
        if len(group) < 3:
            continue
        signal_rows = group.iloc[1:-1].reset_index(drop=True)
        actual_rows = group.iloc[2:].reset_index(drop=True)
        part = signal_rows.copy()
        part["actual_date"] = actual_rows["Date"]
        part["actual_aligned"] = actual_rows["actual"].to_numpy(dtype=float)
        parts.append(part)
    aligned = pd.concat(parts, ignore_index=True)
    aligned["base_abs_prediction"] = aligned["base_prediction"].abs()
    grouped_date = aligned.groupby(["split", "Date"])["base_prediction"]
    aligned["base_prediction_rank_pct"] = grouped_date.rank(method="average", pct=True)
    aligned["base_prediction_zscore"] = grouped_date.transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 1e-8 else 1.0)
    )
    return aligned.replace([np.inf, -np.inf], np.nan)


def build_filter_sequences(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    label_column: str,
    window: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    x_map: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    y_map: dict[str, list[float]] = {"train": [], "val": []}
    meta_map: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    required = [*feature_columns, label_column, "code", "Date", "actual_date", "actual_aligned", "base_prediction", "split"]
    for (code, split), group in frame.dropna(subset=required).sort_values(["code", "Date"], kind="stable").groupby(["code", "split"], sort=False):
        if split not in x_map or len(group) < window:
            continue
        values = group.loc[:, feature_columns].to_numpy(dtype=np.float32)
        labels = group[label_column].to_numpy(dtype=np.float32)
        rows = group.reset_index(drop=True)
        for end_idx in range(window - 1, len(rows)):
            x_map[str(split)].append(values[end_idx - window + 1 : end_idx + 1])
            y_map[str(split)].append(float(labels[end_idx]))
            meta_map[str(split)].append(rows.iloc[end_idx].to_dict())

    out_x = {split: np.asarray(items, dtype=np.float32) for split, items in x_map.items()}
    out_y = {split: np.asarray(items, dtype=np.float32) for split, items in y_map.items()}
    out_meta = {split: pd.DataFrame(items) for split, items in meta_map.items()}
    return out_x, out_y, out_meta


def fit_filter_scaler(train_frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    mean = train_frame.loc[:, feature_columns].astype(float).mean(axis=0)
    std = train_frame.loc[:, feature_columns].astype(float).std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return mean, std


def apply_filter_scaler(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    mean: pd.Series,
    std: pd.Series,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    out = frame.copy()
    scaled_columns = tuple(f"{column}__scaled" for column in feature_columns)
    for source, target in zip(feature_columns, scaled_columns):
        out[target] = (out[source].astype(float) - float(mean[source])) / float(std[source])
    return out, scaled_columns


def build_filter_model(window: int, n_features: int, seed: int) -> keras.Model:
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window, n_features)),
            keras.layers.LSTM(16, dropout=0.05),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def metric_bundle(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    clean = frame.dropna(subset=[prediction_column, "actual_aligned", "code"]).sort_values(["code", "Date"], kind="stable")
    if len(clean) < 10:
        return {
            "n_obs": int(len(clean)),
            "rel_score": float("nan"),
            "directional_accuracy": float("nan"),
            "mean_daily_ic": float("nan"),
            "daily_ic_t": float("nan"),
            "quartile_equity": float("nan"),
            "quartile_hit_rate": float("nan"),
        }
    scores = evaluate(
        clean[prediction_column].to_numpy(dtype=float),
        clean["actual_aligned"].to_numpy(dtype=float),
        group_ids=clean["code"].astype(str).to_numpy(),
    )
    daily_ics: list[float] = []
    long_short: list[float] = []
    for _, day in clean.groupby("actual_date", sort=True):
        day = day.dropna(subset=[prediction_column, "actual_aligned"])
        if len(day) < 20 or day[prediction_column].nunique() < 3:
            continue
        daily_ics.append(float(day[prediction_column].rank().corr(day["actual_aligned"].rank())))
        rank_pct = day[prediction_column].rank(method="first", pct=True)
        top = day[rank_pct >= 0.8]
        bottom = day[rank_pct <= 0.2]
        if len(top) and len(bottom):
            long_short.append(float(top["actual_aligned"].mean() - bottom["actual_aligned"].mean()))
    ic_arr = np.asarray([value for value in daily_ics if np.isfinite(value)], dtype=float)
    ls_arr = np.asarray([value for value in long_short if np.isfinite(value)], dtype=float)
    ic_std = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else float("nan")
    return {
        "n_obs": int(len(clean)),
        "rel_score": float(scores["rel_score"]),
        "directional_accuracy": float(scores["directional_accuracy"]),
        "mean_daily_ic": float(np.mean(ic_arr)) if len(ic_arr) else float("nan"),
        "daily_ic_t": float(np.mean(ic_arr) / (ic_std / np.sqrt(len(ic_arr)))) if len(ic_arr) > 1 and ic_std > 0 else float("nan"),
        "quartile_equity": float(np.prod(1.0 + ls_arr)) if len(ls_arr) else float("nan"),
        "quartile_hit_rate": float(np.mean(ls_arr > 0.0)) if len(ls_arr) else float("nan"),
    }


def risk_scale(frame: pd.DataFrame) -> np.ndarray:
    scale = np.ones(len(frame), dtype=float)
    scale *= np.where(frame["market_proxy_drawdown_60"].to_numpy(dtype=float) < -0.05, 0.70, 1.0)
    scale *= np.where(frame["market_proxy_volatility_ratio_20"].to_numpy(dtype=float) > 1.25, 0.80, 1.0)
    scale *= np.where(frame["market_breadth_20"].to_numpy(dtype=float) < 0.45, 0.80, 1.0)
    return np.clip(scale, 0.25, 1.0)


def gate_coverage_bundle(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    coverage: dict[str, dict[str, float]] = {}
    for split, group in frame.groupby("split", sort=True):
        active = group["prediction_gate"].to_numpy(dtype=float) != 0.0
        base_hit = (
            group["base_prediction"].to_numpy(dtype=float) * group["actual_aligned"].to_numpy(dtype=float)
        ) > 0.0
        active_group = group.loc[active]
        active_hit = (
            active_group["prediction_gate"].to_numpy(dtype=float)
            * active_group["actual_aligned"].to_numpy(dtype=float)
        ) > 0.0
        coverage[str(split)] = {
            "gate_coverage": float(np.mean(active)) if len(group) else float("nan"),
            "base_hit_rate": float(np.mean(base_hit)) if len(group) else float("nan"),
            "gate_active_hit_rate": float(np.mean(active_hit)) if len(active_group) else float("nan"),
            "filter_probability_mean": float(group["filter_probability"].mean()),
            "filter_probability_q90": float(group["filter_probability"].quantile(0.90)),
        }
    return coverage


def write_markdown(
    output_dir: Path,
    summary: pd.DataFrame,
    train_threshold: float,
    train_daily_coverage: float,
    train_move_daily_coverage: float,
    train_move_daily_ic_coverage: float,
    label_rate: dict[str, float],
    market_proxy_source: str,
    coverage_stats: dict[str, dict[str, float]],
    candidate_coverage: pd.DataFrame,
) -> None:
    lines = [
        "# Portable LSTM Filter Signal Probe",
        "",
        "Scope: train/validation only. Test/out-sample data is not used.",
        "",
        "Architecture: base LSTM prediction -> small LSTM confidence filter -> generic market risk gate.",
        "",
        "Market/risk context uses portable names such as `market_proxy_return_20`, `market_breadth_20`, and `market_proxy_drawdown_60`; market-index source columns are treated as aliases behind `market_proxy_return_1`.",
        "",
        f"Market proxy source used in this run: `{market_proxy_source}`.",
        f"Train-selected gate threshold: `{train_threshold:.2f}`.",
        f"Train-selected daily top coverage by filter probability: `{train_daily_coverage:.1%}`.",
        f"Train-selected daily top coverage by expected move / rel_score: `{train_move_daily_coverage:.1%}`.",
        f"Train-selected daily top coverage by expected move / mean IC: `{train_move_daily_ic_coverage:.1%}`.",
        f"Tradeable label rate: train `{label_rate['train']:.1%}`, val `{label_rate['val']:.1%}`.",
        "",
        "## Metrics",
        "",
        "| Split | Candidate | rel_score | Direction | Mean IC | IC t | Quartile equity | Hit rate | Obs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values(["split", "rel_score"], ascending=[True, False], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['split']}` | `{row['candidate']}` | {float(row['rel_score']):+.4f} | "
            f"{float(row['directional_accuracy']):.1%} | {float(row['mean_daily_ic']):+.4f} | "
            f"{float(row['daily_ic_t']):+.2f} | {float(row['quartile_equity']):.3f} | "
            f"{float(row['quartile_hit_rate']):.1%} | {int(row['n_obs'])} |"
        )
    lines.extend(
        [
            "",
            "## Gate Coverage",
            "",
            "| Split | Gate coverage | Base hit rate | Active gate hit rate | Mean probability | P90 probability |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, values in coverage_stats.items():
        lines.append(
            "| "
            f"`{split}` | {values['gate_coverage']:.1%} | {values['base_hit_rate']:.1%} | "
            f"{values['gate_active_hit_rate']:.1%} | {values['filter_probability_mean']:.3f} | "
            f"{values['filter_probability_q90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Coverage",
            "",
            "| Split | Candidate | Coverage | Active hit rate |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for _, row in candidate_coverage.sort_values(["split", "coverage"], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['split']}` | `{row['candidate']}` | {float(row['coverage']):.1%} | "
            f"{float(row['active_hit_rate']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- If validation improves only after hard gating, keep the filter as a post-model selection layer rather than retraining the base model.",
            "- If expected-move daily selection beats hard gating with materially higher coverage, prefer it as the next router candidate.",
            "- If probability scaling hurts `rel_score`, the filter is learning tradeability but not calibration; use it for position sizing or no-trade gates only.",
            "- If naive risk scaling hurts, keep market regime context as input to the filter before using it as an output multiplier.",
            "- This is a first probe, not final model selection. Threshold is selected on train only.",
        ]
    )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = FilterConfig(
        filter_window=args.filter_window,
        tradeable_abs_quantile=args.tradeable_abs_quantile,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    run_dir = RUN_ROOT / args.run
    run_config = load_json(run_dir / "reports" / "core" / "config.json")
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_frame, market_proxy_source = build_base_prediction_frame(
        run_dir,
        run_config,
        args.model,
        market_leader_count=args.market_leader_count,
        market_leader_liquidity_window=args.market_leader_liquidity_window,
        market_leader_liquidity_min_periods=args.market_leader_liquidity_min_periods,
    )
    base_frame = align_signal_actual(prediction_frame)
    train_abs_threshold = float(
        base_frame.loc[base_frame["split"] == "train", "actual_aligned"].abs().quantile(config.tradeable_abs_quantile)
    )
    base_frame["tradeable_label"] = (
        (np.sign(base_frame["base_prediction"]) == np.sign(base_frame["actual_aligned"]))
        & (base_frame["actual_aligned"].abs() >= train_abs_threshold)
    ).astype(float)

    excluded_filter_features = {str(column) for column in args.exclude_filter_feature}
    filter_features = tuple(
        column
        for column in DEFAULT_FILTER_FEATURES
        if column in base_frame.columns and column not in excluded_filter_features
    )
    train_mean, train_std = fit_filter_scaler(base_frame[base_frame["split"] == "train"], filter_features)
    scaled_frame, scaled_filter_features = apply_filter_scaler(base_frame, filter_features, train_mean, train_std)
    x_map, y_map, meta_map = build_filter_sequences(
        scaled_frame,
        scaled_filter_features,
        "tradeable_label",
        config.filter_window,
    )
    if len(x_map["train"]) == 0 or len(x_map["val"]) == 0:
        raise ValueError("Not enough filter sequences for train/val.")

    model = build_filter_model(config.filter_window, len(filter_features), config.seed)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=config.patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_map["train"],
        y_map["train"],
        validation_data=(x_map["val"], y_map["val"]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    scored_parts: list[pd.DataFrame] = []
    for split in ("train", "val"):
        part = meta_map[split].copy()
        part["filter_probability"] = model.predict(x_map[split], verbose=0).reshape(-1)
        scored_parts.append(part)
    scored = pd.concat(scored_parts, ignore_index=True)
    selection_params = filter_selection.fit_filter_signal_selection(scored, metric_bundle)
    scored["market_risk_scale"] = risk_scale(scored)
    scored, candidate_columns = filter_selection.apply_filter_signal_selection(scored, selection_params)
    summary_rows: list[dict[str, object]] = []
    for split, group in scored.groupby("split", sort=True):
        for candidate, column in candidate_columns.items():
            row = {"split": split, "candidate": candidate}
            row.update(metric_bundle(group, column))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    label_rate = {
        "train": float(scored.loc[scored["split"] == "train", "tradeable_label"].mean()),
        "val": float(scored.loc[scored["split"] == "val", "tradeable_label"].mean()),
    }
    coverage_stats = gate_coverage_bundle(scored)
    candidate_coverage = pd.DataFrame(filter_selection.candidate_coverage_bundle(scored, candidate_columns))

    summary.to_csv(output_dir / "filter_candidate_summary.csv", index=False)
    candidate_coverage.to_csv(output_dir / "filter_candidate_coverage.csv", index=False)
    scored[
        [
            "code",
            "split",
            "Date",
            "actual_date",
            "actual_aligned",
            "base_prediction",
            "filter_probability",
            "filter_expected_move",
            "market_risk_scale",
            "tradeable_label",
            *candidate_columns.values(),
        ]
    ].to_csv(output_dir / "filter_predictions.csv", index=False)
    pd.DataFrame(history.history).to_csv(output_dir / "filter_history.csv", index=False)
    model.save(output_dir / "filter_model.keras")
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run": args.run,
                "model": args.model,
                "model_path": str(model_path_for(run_dir, args.model)),
                "filter_config": config.__dict__,
                "filter_features": list(filter_features),
                "excluded_filter_features": sorted(excluded_filter_features),
                "market_proxy_source": market_proxy_source,
                "market_leader_count": args.market_leader_count,
                "market_leader_liquidity_window": args.market_leader_liquidity_window,
                "market_leader_liquidity_min_periods": args.market_leader_liquidity_min_periods,
                "train_abs_threshold": train_abs_threshold,
                "selection_params": selection_params.to_dict(),
                "train_gate_threshold": selection_params.gate_threshold,
                "train_daily_coverage": selection_params.daily_coverage,
                "train_move_daily_coverage": selection_params.move_daily_coverage,
                "train_move_daily_ic_coverage": selection_params.move_daily_ic_coverage,
                "label_rate": label_rate,
                "coverage_stats": coverage_stats,
                "candidate_coverage": candidate_coverage.to_dict(orient="records"),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(
        output_dir,
        summary,
        selection_params.gate_threshold,
        selection_params.daily_coverage,
        selection_params.move_daily_coverage,
        selection_params.move_daily_ic_coverage,
        label_rate,
        market_proxy_source,
        coverage_stats,
        candidate_coverage,
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": int(len(scored)), "features": len(filter_features)}, indent=2))


if __name__ == "__main__":
    main()
