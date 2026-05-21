from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DEFAULT_REBALANCE_DAYS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)


@dataclass(frozen=True)
class HoldingPeriodSelection:
    selector: str
    candidate: str
    rebalance_every: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def annualized_sharpe(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std <= 0.0:
        return float("nan")
    return float(clean.mean() / std * np.sqrt(252.0))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    return float((equity / peak.replace(0.0, np.nan) - 1.0).min())


def desired_weights_for_day(day: pd.DataFrame, prediction_column: str, min_positions: int) -> dict[str, float]:
    active = day.loc[day[prediction_column].astype(float) != 0.0, ["code", prediction_column]]
    if len(active) < min_positions:
        return {}
    longs = active[prediction_column] > 0.0
    shorts = active[prediction_column] < 0.0
    n_long = int(longs.sum())
    n_short = int(shorts.sum())
    if n_long + n_short < min_positions:
        return {}

    weights: dict[str, float] = {}
    if n_long > 0 and n_short > 0:
        for code in active.loc[longs, "code"].astype(str):
            weights[code] = 0.5 / n_long
        for code in active.loc[shorts, "code"].astype(str):
            weights[code] = -0.5 / n_short
    elif n_long > 0:
        for code in active.loc[longs, "code"].astype(str):
            weights[code] = 1.0 / n_long
    elif n_short > 0:
        for code in active.loc[shorts, "code"].astype(str):
            weights[code] = -1.0 / n_short
    return weights


def l1_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    keys = set(previous).union(current)
    return float(sum(abs(current.get(key, 0.0) - previous.get(key, 0.0)) for key in keys))


def simulate_rebalance(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    rebalance_every: int,
    cost_bps: float,
    min_positions: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_weights: dict[str, float] = {}
    daily_groups = list(frame.dropna(subset=["actual_date"]).sort_values("actual_date").groupby("actual_date", sort=True))
    for idx, (actual_date, day) in enumerate(daily_groups):
        returns = dict(zip(day["code"].astype(str), day["actual_aligned"].astype(float)))
        turnover = 0.0
        if idx % rebalance_every == 0:
            target_weights = desired_weights_for_day(day, prediction_column, min_positions)
            turnover = l1_turnover(current_weights, target_weights)
            current_weights = target_weights

        gross_return = float(sum(weight * returns.get(code, 0.0) for code, weight in current_weights.items()))
        cost_return = turnover * cost_bps / 10_000.0
        rows.append(
            {
                "actual_date": actual_date,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost_return": cost_return,
                "net_return": gross_return - cost_return,
                "n_positions": int(sum(abs(weight) > 0.0 for weight in current_weights.values())),
                "gross_exposure": float(sum(abs(weight) for weight in current_weights.values())),
                "is_rebalance_day": bool(idx % rebalance_every == 0),
            }
        )
    return pd.DataFrame(rows)


def summarize_rebalance(
    daily: pd.DataFrame,
    split: str,
    candidate: str,
    rebalance_every: int,
    cost_bps: float,
) -> dict[str, object]:
    gross_equity = (1.0 + daily["gross_return"]).cumprod()
    net_equity = (1.0 + daily["net_return"]).cumprod()
    return {
        "split": split,
        "candidate": candidate,
        "rebalance_every": rebalance_every,
        "cost_bps": cost_bps,
        "n_days": int(len(daily)),
        "avg_positions": float(daily["n_positions"].mean()),
        "avg_turnover": float(daily["turnover"].mean()),
        "gross_equity": float(gross_equity.iloc[-1]) if len(gross_equity) else float("nan"),
        "net_equity": float(net_equity.iloc[-1]) if len(net_equity) else float("nan"),
        "gross_sharpe": annualized_sharpe(daily["gross_return"]),
        "net_sharpe": annualized_sharpe(daily["net_return"]),
        "net_max_drawdown": max_drawdown(net_equity),
        "net_hit_rate": float((daily["net_return"] > 0.0).mean()),
    }


def window_year_score(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "worst_year_equity": float("nan"),
            "median_year_equity": float("nan"),
            "full_equity": float("nan"),
            "net_sharpe": float("nan"),
        }
    tmp = daily.copy()
    tmp["year"] = tmp["actual_date"].dt.year
    yearly = tmp.groupby("year", sort=True)["net_return"].apply(lambda s: float((1.0 + s).prod()))
    return {
        "worst_year_equity": float(yearly.min()),
        "median_year_equity": float(yearly.median()),
        "full_equity": float((1.0 + tmp["net_return"]).prod()),
        "net_sharpe": annualized_sharpe(tmp["net_return"]),
    }


def evaluate_rebalance_grid(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    cost_bps: float,
    min_positions: int,
    *,
    rebalance_days: tuple[int, ...] = DEFAULT_REBALANCE_DAYS,
) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    daily_map: dict[tuple[str, int], pd.DataFrame] = {}
    for candidate in candidates:
        for rebalance_every in rebalance_days:
            daily = simulate_rebalance(
                frame,
                candidate,
                rebalance_every=rebalance_every,
                cost_bps=cost_bps,
                min_positions=min_positions,
            )
            daily_map[(candidate, rebalance_every)] = daily
            summary = summarize_rebalance(daily, "window", candidate, rebalance_every, cost_bps)
            rows.append(
                {
                    "candidate": candidate,
                    "rebalance_every": rebalance_every,
                    **window_year_score(daily),
                    **{
                        f"summary_{key}": value
                        for key, value in summary.items()
                        if key not in {"split", "candidate", "rebalance_every"}
                    },
                }
            )
    return pd.DataFrame(rows), daily_map


def select_by_worst_year(train_grid: pd.DataFrame) -> pd.Series:
    clean = train_grid.dropna(subset=["worst_year_equity", "median_year_equity", "full_equity"]).copy()
    if clean.empty:
        raise ValueError("No train rows available for robust selection.")
    clean = clean.sort_values(
        ["worst_year_equity", "median_year_equity", "full_equity", "net_sharpe"],
        ascending=[False, False, False, False],
        kind="stable",
    )
    return clean.iloc[0]


def select_by_full_sharpe(train_grid: pd.DataFrame) -> pd.Series:
    clean = train_grid.dropna(subset=["net_sharpe", "full_equity"]).copy()
    if clean.empty:
        raise ValueError("No train rows available for Sharpe selection.")
    clean = clean.sort_values(["net_sharpe", "full_equity"], ascending=[False, False], kind="stable")
    return clean.iloc[0]


def constrain_rebalance_grid(
    train_grid: pd.DataFrame,
    *,
    max_avg_turnover: float | None = None,
    max_drawdown: float | None = None,
    min_worst_year_equity: float | None = None,
) -> tuple[pd.DataFrame, bool]:
    clean = train_grid.copy()
    if max_avg_turnover is not None and "summary_avg_turnover" in clean.columns:
        clean = clean.loc[clean["summary_avg_turnover"] <= max_avg_turnover].copy()
    if max_drawdown is not None and "summary_net_max_drawdown" in clean.columns:
        clean = clean.loc[clean["summary_net_max_drawdown"] >= -abs(max_drawdown)].copy()
    if min_worst_year_equity is not None:
        clean = clean.loc[clean["worst_year_equity"] >= min_worst_year_equity].copy()
    if clean.empty:
        return train_grid.copy(), True
    return clean, False


def select_by_constrained_worst_year(
    train_grid: pd.DataFrame,
    *,
    max_avg_turnover: float | None = None,
    max_drawdown: float | None = None,
    min_worst_year_equity: float | None = None,
) -> pd.Series:
    constrained, constraints_relaxed = constrain_rebalance_grid(
        train_grid,
        max_avg_turnover=max_avg_turnover,
        max_drawdown=max_drawdown,
        min_worst_year_equity=min_worst_year_equity,
    )
    selected = select_by_worst_year(constrained).copy()
    selected["constraints_relaxed"] = constraints_relaxed
    return selected


def select_holding_period(train_grid: pd.DataFrame, selector: str) -> HoldingPeriodSelection:
    if selector == "worst_year_net_equity":
        selected = select_by_worst_year(train_grid)
    elif selector == "full_train_net_sharpe":
        selected = select_by_full_sharpe(train_grid)
    else:
        raise ValueError(f"Unsupported holding-period selector: {selector}")
    return HoldingPeriodSelection(
        selector=selector,
        candidate=str(selected["candidate"]),
        rebalance_every=int(selected["rebalance_every"]),
    )


def build_walk_forward_folds(
    dates: list[pd.Timestamp],
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    start = 0
    fold_id = 1
    while start + train_days + test_days <= len(dates):
        train_start = start
        train_end = start + train_days
        test_start = train_end
        test_end = test_start + test_days
        folds.append(
            {
                "fold": fold_id,
                "train_start": dates[train_start],
                "train_end": dates[train_end - 1],
                "test_start": dates[test_start],
                "test_end": dates[test_end - 1],
                "train_dates": set(dates[train_start:train_end]),
                "test_dates": set(dates[test_start:test_end]),
            }
        )
        fold_id += 1
        start += step_days
    return folds


def aggregate_selected_daily(selected_daily: pd.DataFrame) -> dict[str, float]:
    if selected_daily.empty:
        return {
            "stitched_net_equity": float("nan"),
            "stitched_gross_equity": float("nan"),
            "stitched_net_sharpe": float("nan"),
            "stitched_max_drawdown": float("nan"),
            "stitched_hit_rate": float("nan"),
            "avg_turnover": float("nan"),
            "avg_positions": float("nan"),
        }
    net_equity = (1.0 + selected_daily["net_return"]).cumprod()
    gross_equity = (1.0 + selected_daily["gross_return"]).cumprod()
    return {
        "stitched_net_equity": float(net_equity.iloc[-1]),
        "stitched_gross_equity": float(gross_equity.iloc[-1]),
        "stitched_net_sharpe": annualized_sharpe(selected_daily["net_return"]),
        "stitched_max_drawdown": max_drawdown(net_equity),
        "stitched_hit_rate": float((selected_daily["net_return"] > 0.0).mean()),
        "avg_turnover": float(selected_daily["turnover"].mean()),
        "avg_positions": float(selected_daily["n_positions"].mean()),
    }
