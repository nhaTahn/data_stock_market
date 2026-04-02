from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fk_lstm_classifier.config import ExperimentConfig, config_to_dict
from fk_lstm_classifier.data import (
    ANCHOR_DATE_COLUMN,
    PreparedPanel,
    build_classification_panel,
    build_datasets_for_date_splits,
    load_market_data,
)
from fk_lstm_classifier.evaluation import (
    build_long_short_holdings,
    build_prediction_frame,
    compute_classification_metrics,
    compute_long_short_returns_from_holdings,
)
from fk_lstm_classifier.market_rules import MarketRuntimeSettings, resolve_market_runtime_settings
from fk_lstm_classifier.model import build_lstm_classifier, set_seed
from fk_lstm_classifier.reporting import render_dashboard, summarize_strategy
from fk_lstm_classifier.training import TrainResult, train_classifier


@dataclass
class ExperimentResult:
    run_dir: Path
    metrics: dict[str, float]
    strategy_summary: dict[str, float]
    dashboard_path: Path
    prediction_count: int
    fold_count: int


def _resolve_run_name(config: ExperimentConfig) -> str:
    if config.run_name:
        return config.run_name
    return f"{'_'.join(config.markets).lower()}_{config.model_type}"


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _with_schema(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not frame.empty:
        return frame
    return pd.DataFrame(columns=columns)


def _aggregate_histories(histories: list[dict[str, list[float]]]) -> pd.DataFrame:
    if not histories:
        return pd.DataFrame()

    frames = []
    for fold_id, history in enumerate(histories, start=1):
        frame = pd.DataFrame(history)
        frame["epoch"] = frame.index + 1
        frame["fold_id"] = fold_id
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    averaged = combined.groupby("epoch").mean(numeric_only=True).reset_index(drop=True)
    return averaged.drop(columns=["fold_id"], errors="ignore")


def _train_and_score(
    panel: pd.DataFrame,
    config: ExperimentConfig,
    market_settings: MarketRuntimeSettings,
    train_dates: tuple[pd.Timestamp, ...],
    validation_dates: tuple[pd.Timestamp, ...],
    test_dates: tuple[pd.Timestamp, ...] | None,
    fold_id: int,
) -> tuple[PreparedPanel, TrainResult, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float]]:
    prepared, train_dataset, validation_dataset, test_dataset = build_datasets_for_date_splits(
        panel=panel,
        lookback=config.lookback,
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
    )
    score_dataset = validation_dataset if test_dataset is None else test_dataset

    set_seed(config.seed + fold_id - 1)
    model = build_lstm_classifier(
        lookback=config.lookback,
        feature_dim=1,
        lstm_units=config.lstm_units,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        use_attention=(config.model_type == "attention"),
    )
    train_result = train_classifier(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
    )

    score_probabilities = model.predict(score_dataset.features, verbose=0)
    predictions = build_prediction_frame(score_dataset, score_probabilities, panel_frame=prepared.frame)
    predictions["fold_id"] = fold_id

    holdings = build_long_short_holdings(
        predictions,
        top_k=config.top_k,
        allow_short=market_settings.allow_short,
        min_daily_value_traded=market_settings.min_daily_value_traded,
        min_adv20_value_traded=market_settings.min_adv20_value_traded,
        max_position_adv_fraction=market_settings.max_position_adv_fraction,
        portfolio_notional=market_settings.portfolio_notional,
        block_limit_up_entry=market_settings.block_limit_up_entry,
        exclude_hard_issues=market_settings.exclude_hard_issues,
    )
    if not holdings.empty:
        holdings["fold_id"] = fold_id
    strategy_returns = compute_long_short_returns_from_holdings(
        holdings=holdings,
        transaction_cost_bps=config.transaction_cost_bps,
        buy_cost_bps=market_settings.buy_cost_bps,
        sell_cost_bps=market_settings.sell_cost_bps,
        sell_tax_bps=market_settings.sell_tax_bps,
    )
    if not strategy_returns.empty:
        strategy_returns["fold_id"] = fold_id

    metrics = compute_classification_metrics(
        labels=score_dataset.labels,
        prob_class_1=score_probabilities[:, 1],
    )
    strategy_summary = summarize_strategy(strategy_returns)
    return prepared, train_result, predictions, holdings, strategy_returns, metrics, strategy_summary


def _save_artifacts(
    run_dir: Path,
    config: ExperimentConfig,
    market_settings: MarketRuntimeSettings,
    histories: list[dict[str, list[float]]],
    predictions: pd.DataFrame,
    holdings: pd.DataFrame,
    strategy_returns: pd.DataFrame,
    fold_summary: pd.DataFrame | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    history_df = _aggregate_histories(histories)
    if not history_df.empty:
        history_df.to_csv(run_dir / "fit_history.csv", index=False)
    _with_schema(
        predictions,
        [
            "anchor_date",
            "realized_date",
            "code",
            "target_class",
            "next_return",
            "prob_class_0",
            "prob_class_1",
            "fold_id",
        ],
    ).to_csv(run_dir / "validation_predictions.csv", index=False)
    _with_schema(
        strategy_returns,
        [
            "realized_date",
            "long_count",
            "short_count",
            "long_return",
            "short_return",
            "spread_return",
            "strategy_return_gross",
            "turnover",
            "buy_turnover",
            "sell_turnover",
            "buy_cost",
            "sell_cost",
            "sell_tax",
            "transaction_cost",
            "strategy_return",
            "equity_curve_gross",
            "equity_curve",
            "fold_id",
        ],
    ).to_csv(run_dir / "validation_long_short_returns.csv", index=False)
    _with_schema(
        holdings,
        [
            "realized_date",
            "anchor_date",
            "code",
            "side",
            "weight",
            "prob_class_1",
            "target_class",
            "next_return",
            "return_contribution",
            "fold_id",
        ],
    ).to_csv(run_dir / "portfolio_holdings.csv", index=False)
    if fold_summary is not None and not fold_summary.empty:
        fold_summary.to_csv(run_dir / "fold_summary.csv", index=False)
    _save_json(run_dir / "config.json", config_to_dict(config))
    _save_json(run_dir / "market_rules.json", market_settings.to_dict())


def _build_walk_forward_splits(
    unique_dates: tuple[pd.Timestamp, ...],
    config: ExperimentConfig,
) -> list[dict[str, tuple[pd.Timestamp, ...]]]:
    splits: list[dict[str, tuple[pd.Timestamp, ...]]] = []
    total_dates = len(unique_dates)
    train_end = config.min_train_days

    while True:
        validation_end = train_end + config.validation_days
        test_end = validation_end + config.test_days
        if test_end > total_dates:
            break

        if config.window_scheme == "rolling":
            train_start = max(0, train_end - config.min_train_days)
        else:
            train_start = 0

        splits.append(
            {
                "train_dates": unique_dates[train_start:train_end],
                "validation_dates": unique_dates[train_end:validation_end],
                "test_dates": unique_dates[validation_end:test_end],
            }
        )
        train_end += config.step_days

    if not splits:
        raise ValueError(
            "No walk-forward folds could be created. Reduce min_train_days/validation_days/test_days."
        )
    return splits


def _fold_row(
    fold_id: int,
    prepared: PreparedPanel,
    metrics: dict[str, float],
    strategy_summary: dict[str, float],
) -> dict[str, object]:
    return {
        "fold_id": fold_id,
        "train_start": prepared.train_dates[0],
        "train_end": prepared.train_dates[-1],
        "validation_start": prepared.validation_dates[0],
        "validation_end": prepared.validation_dates[-1],
        "test_start": prepared.test_dates[0] if prepared.test_dates else prepared.validation_dates[0],
        "test_end": prepared.test_dates[-1] if prepared.test_dates else prepared.validation_dates[-1],
        "scaler_mean": prepared.scaler_mean,
        "scaler_std": prepared.scaler_std,
        **metrics,
        **strategy_summary,
    }


def run_holdout_experiment(config: ExperimentConfig) -> ExperimentResult:
    market_settings = resolve_market_runtime_settings(config)
    price_frame = load_market_data(config.data_dir, config.markets, vn_data_profile=market_settings.vn_data_profile)
    panel = build_classification_panel(
        price_frame=price_frame,
        min_cross_sectional_count=config.min_cross_sectional_count,
        forward_horizon_days=market_settings.forward_horizon_days,
    )
    unique_dates = tuple(sorted(panel[ANCHOR_DATE_COLUMN].drop_duplicates().tolist()))
    split_idx = int(len(unique_dates) * config.train_ratio)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)

    train_dates = unique_dates[:split_idx]
    validation_dates = unique_dates[split_idx:]
    prepared, train_result, predictions, holdings, strategy_returns, metrics, strategy_summary = _train_and_score(
        panel=panel,
        config=config,
        market_settings=market_settings,
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=None,
        fold_id=1,
    )

    run_dir = config.output_dir / _resolve_run_name(config)
    fold_summary = pd.DataFrame([_fold_row(1, prepared, metrics, strategy_summary)])
    _save_artifacts(
        run_dir=run_dir,
        config=config,
        market_settings=market_settings,
        histories=[train_result.history],
        predictions=predictions,
        holdings=holdings,
        strategy_returns=strategy_returns,
        fold_summary=fold_summary,
    )
    dashboard_path = render_dashboard(run_dir)
    return ExperimentResult(
        run_dir=run_dir,
        metrics=metrics,
        strategy_summary=strategy_summary,
        dashboard_path=dashboard_path,
        prediction_count=int(len(predictions)),
        fold_count=1,
    )


def run_walk_forward_experiment(config: ExperimentConfig) -> ExperimentResult:
    market_settings = resolve_market_runtime_settings(config)
    price_frame = load_market_data(config.data_dir, config.markets, vn_data_profile=market_settings.vn_data_profile)
    panel = build_classification_panel(
        price_frame=price_frame,
        min_cross_sectional_count=config.min_cross_sectional_count,
        forward_horizon_days=market_settings.forward_horizon_days,
    )
    unique_dates = tuple(sorted(panel[ANCHOR_DATE_COLUMN].drop_duplicates().tolist()))
    splits = _build_walk_forward_splits(unique_dates, config)

    histories: list[dict[str, list[float]]] = []
    all_predictions: list[pd.DataFrame] = []
    all_holdings: list[pd.DataFrame] = []
    all_strategy_returns: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for fold_id, split in enumerate(splits, start=1):
        prepared, train_result, predictions, holdings, strategy_returns, metrics, strategy_summary = _train_and_score(
            panel=panel,
            config=config,
            market_settings=market_settings,
            train_dates=split["train_dates"],
            validation_dates=split["validation_dates"],
            test_dates=split["test_dates"],
            fold_id=fold_id,
        )
        histories.append(train_result.history)
        all_predictions.append(predictions)
        all_holdings.append(holdings)
        all_strategy_returns.append(strategy_returns)
        fold_rows.append(_fold_row(fold_id, prepared, metrics, strategy_summary))

    predictions_df = pd.concat(all_predictions, ignore_index=True).sort_values(
        ["realized_date", "prob_class_1"],
        ascending=[True, False],
    )
    holdings_df = pd.concat(all_holdings, ignore_index=True) if all_holdings else pd.DataFrame()
    strategy_df = pd.concat(all_strategy_returns, ignore_index=True).sort_values("realized_date")
    strategy_df = strategy_df.drop_duplicates(subset=["realized_date"], keep="last").reset_index(drop=True)
    fold_summary = pd.DataFrame(fold_rows)

    metrics = compute_classification_metrics(
        labels=predictions_df["target_class"].to_numpy(),
        prob_class_1=predictions_df["prob_class_1"].to_numpy(),
    )
    strategy_summary = summarize_strategy(strategy_df)

    run_dir = config.output_dir / _resolve_run_name(config)
    _save_artifacts(
        run_dir=run_dir,
        config=config,
        market_settings=market_settings,
        histories=histories,
        predictions=predictions_df,
        holdings=holdings_df,
        strategy_returns=strategy_df,
        fold_summary=fold_summary,
    )
    dashboard_path = render_dashboard(run_dir)
    return ExperimentResult(
        run_dir=run_dir,
        metrics=metrics,
        strategy_summary=strategy_summary,
        dashboard_path=dashboard_path,
        prediction_count=int(len(predictions_df)),
        fold_count=int(len(fold_summary)),
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    if config.evaluation_mode == "walk_forward":
        return run_walk_forward_experiment(config)
    return run_holdout_experiment(config)
