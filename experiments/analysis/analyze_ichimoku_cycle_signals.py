from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "signal_search" / "ichimoku_cycle"
DEFAULT_ANCHOR_RUN = "broad_signmag_portable_no_identity_20260428_allvn_r01"


@dataclass(frozen=True)
class IchimokuSpec:
    name: str
    conversion: int
    base: int
    span_b: int
    displacement: int
    chikou: int


@dataclass(frozen=True)
class EvalConfig:
    min_names_per_day: int = 20
    top_quantile: float = 0.2
    cost_bps: float = 10.0


DEFAULT_SPECS = (
    IchimokuSpec("standard_9_26_52_26_26", 9, 26, 52, 26, 26),
    IchimokuSpec("vn_8_22_44_22_22", 8, 22, 44, 22, 22),
    IchimokuSpec("vn_7_21_42_21_21", 7, 21, 42, 21, 21),
    IchimokuSpec("vn_8_21_42_21_21", 8, 21, 42, 21, 21),
    IchimokuSpec("vn_9_22_44_22_22", 9, 22, 44, 22, 22),
    IchimokuSpec("vn_10_22_44_22_22", 10, 22, 44, 22, 22),
    IchimokuSpec("fast_6_18_36_18_18", 6, 18, 36, 18, 18),
)

BASELINE_COLUMNS = (
    "momentum_20",
    "macd_hist",
    "rolling_max_20_gap",
    "close_position",
    "volume_ratio_20",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate causal Ichimoku cycle variants as VN train/validation ranking signals."
    )
    parser.add_argument("--anchor-run", default=DEFAULT_ANCHOR_RUN)
    parser.add_argument("--stamp", default="20260506_r01")
    parser.add_argument("--output-name", default="vn_ichimoku_8_22_44_signal_search")
    parser.add_argument("--min-names-per-day", type=int, default=EvalConfig.min_names_per_day)
    parser.add_argument("--top-quantile", type=float, default=EvalConfig.top_quantile)
    parser.add_argument("--cost-bps", type=float, default=EvalConfig.cost_bps)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_config(run_dir: Path) -> dict:
    for relative in ("reports/core/config.json", "config.json"):
        path = run_dir / relative
        if path.exists():
            return load_json(path)
    raise FileNotFoundError(f"No config.json found under {run_dir}")


def load_market_frame(path: Path, stocks: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    sort_columns = ["code", "Date"]
    if "market" in df.columns:
        sort_columns = ["market", *sort_columns]
    df = df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if stocks:
        selected = {item.strip() for item in stocks.split(",") if item.strip()}
        df = df[df["code"].isin(selected)].copy()
    return df


def split_frame(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    train_end = pd.Timestamp(str(config["train_end_date"]))
    val_end = pd.Timestamp(str(config["val_end_date"]))
    out = df.copy()
    out["split"] = np.where(out["Date"] <= train_end, "train", np.where(out["Date"] <= val_end, "val", "test"))
    return out[out["split"].isin({"train", "val"})].copy()


def adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].replace(0, np.nan)
    factor = out["adjust"] / close if "adjust" in out.columns else pd.Series(1.0, index=out.index)
    out["_ichi_close"] = out["adjust"] if "adjust" in out.columns else out["close"]
    out["_ichi_high"] = out["high"] * factor
    out["_ichi_low"] = out["low"] * factor
    return out


def midpoint(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    high_roll = high.rolling(window, min_periods=window).max()
    low_roll = low.rolling(window, min_periods=window).min()
    return (high_roll + low_roll) / 2.0


def add_ichimoku_signals(df: pd.DataFrame, specs: tuple[IchimokuSpec, ...]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in adjusted_ohlc(df).groupby("code", sort=False):
        part = group.sort_values("Date", kind="stable").copy()
        close = part["_ichi_close"].astype(float)
        high = part["_ichi_high"].astype(float)
        low = part["_ichi_low"].astype(float)
        close_safe = close.replace(0.0, np.nan)

        for spec in specs:
            prefix = f"ichi_{spec.name}"
            tenkan = midpoint(high, low, spec.conversion)
            kijun = midpoint(high, low, spec.base)
            span_a_now = (tenkan + kijun) / 2.0
            span_b_now = midpoint(high, low, spec.span_b)
            cloud_top = pd.concat([span_a_now, span_b_now], axis=1).max(axis=1)
            cloud_bottom = pd.concat([span_a_now, span_b_now], axis=1).min(axis=1)
            cloud_mid = (span_a_now + span_b_now) / 2.0
            cloud_width = (cloud_top - cloud_bottom).replace(0.0, np.nan)

            part[f"{prefix}__tenkan_kijun_gap"] = (tenkan - kijun) / close_safe
            part[f"{prefix}__close_cloud_gap"] = (close - cloud_mid) / close_safe
            part[f"{prefix}__cloud_position"] = (close - cloud_bottom) / cloud_width
            part[f"{prefix}__cloud_thickness"] = (span_a_now - span_b_now) / close_safe
            part[f"{prefix}__cloud_slope_5"] = cloud_mid.pct_change(5)
            part[f"{prefix}__chikou_momentum"] = close / close.shift(spec.chikou) - 1.0
            part[f"{prefix}__tk_cross_state"] = np.sign(tenkan - kijun)

            components = [
                f"{prefix}__tenkan_kijun_gap",
                f"{prefix}__close_cloud_gap",
                f"{prefix}__cloud_slope_5",
                f"{prefix}__chikou_momentum",
            ]
            ranked = part[components].rank(axis=0, pct=True)
            part[f"{prefix}__composite_momentum"] = ranked.mean(axis=1)

        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    drop_cols = [column for column in out.columns if column.startswith("_ichi_")]
    return out.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan)


def spearman_corr(left: pd.Series, right: pd.Series) -> float:
    clean = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return float("nan")
    if clean.iloc[:, 0].nunique(dropna=True) < 2 or clean.iloc[:, 1].nunique(dropna=True) < 2:
        return float("nan")
    return float(clean.iloc[:, 0].rank(method="average").corr(clean.iloc[:, 1].rank(method="average")))


def build_signal_catalog(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in df.columns:
        if column.startswith("ichi_"):
            parts = column.split("__", maxsplit=1)
            rows.append({"signal": column, "family": parts[0], "component": parts[1]})
    for column in BASELINE_COLUMNS:
        if column in df.columns:
            rows.append({"signal": column, "family": "baseline_existing", "component": column})
    return pd.DataFrame(rows)


def daily_signal_metrics(
    df: pd.DataFrame,
    signal_catalog: pd.DataFrame,
    target_column: str,
    config: EvalConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, signal_row in signal_catalog.iterrows():
        signal = str(signal_row["signal"])
        previous_positions: dict[str, pd.Series] = {}
        for (split, date), group in df.groupby(["split", "Date"], sort=True):
            clean = group[["code", signal, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < config.min_names_per_day:
                continue
            ic = spearman_corr(clean[signal], clean[target_column])
            if not np.isfinite(ic):
                continue

            rank_pct = clean[signal].rank(method="first", pct=True)
            top = clean[rank_pct >= 1.0 - config.top_quantile]
            bottom = clean[rank_pct <= config.top_quantile]
            if top.empty or bottom.empty:
                long_short = float("nan")
                long_short_net = float("nan")
                top_return = float("nan")
                bottom_return = float("nan")
                turnover = float("nan")
            else:
                top_return = float(top[target_column].mean())
                bottom_return = float(bottom[target_column].mean())
                long_short = top_return - bottom_return
                position = pd.Series(0.0, index=clean["code"].astype(str))
                position.loc[top["code"].astype(str)] = 1.0 / len(top)
                position.loc[bottom["code"].astype(str)] = -1.0 / len(bottom)
                previous = previous_positions.get(str(split), pd.Series(dtype=float))
                aligned = pd.concat(
                    [previous.rename("previous"), position.rename("current")],
                    axis=1,
                ).fillna(0.0)
                turnover = float((aligned["current"] - aligned["previous"]).abs().sum())
                long_short_net = long_short - (config.cost_bps / 10_000.0) * turnover
                previous_positions[str(split)] = position

            rows.append(
                {
                    "signal": signal,
                    "family": signal_row["family"],
                    "component": signal_row["component"],
                    "split": split,
                    "Date": date,
                    "ic": ic,
                    "top_return": top_return,
                    "bottom_return": bottom_return,
                    "long_short_return": long_short,
                    "long_short_return_net": long_short_net,
                    "turnover": turnover,
                    "name_count": int(len(clean)),
                    "top_count": int(len(top)),
                    "bottom_count": int(len(bottom)),
                }
            )
    return pd.DataFrame(rows)


def summarize_values(values: pd.Series) -> tuple[float, float, float, float, int]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan")
    t_stat = float(mean / (std / np.sqrt(len(clean)))) if len(clean) > 1 and std > 0 else float("nan")
    positive_share = float(np.mean(clean > 0.0))
    return mean, std, t_stat, positive_share, int(len(clean))


def summarize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in daily.groupby(["signal", "family", "component", "split"], sort=True):
        ic_mean, ic_std, ic_t, ic_pos, days = summarize_values(group["ic"])
        ls_mean, ls_std, ls_t, ls_pos, ls_days = summarize_values(group["long_short_return"])
        ls_net_mean, ls_net_std, ls_net_t, ls_net_pos, ls_net_days = summarize_values(group["long_short_return_net"])
        ls_curve = (1.0 + group["long_short_return"].fillna(0.0)).cumprod()
        ls_net_curve = (1.0 + group["long_short_return_net"].fillna(0.0)).cumprod()
        rows.append(
            {
                **dict(zip(["signal", "family", "component", "split"], keys)),
                "days": days,
                "mean_ic": ic_mean,
                "ic_std": ic_std,
                "ic_t_stat": ic_t,
                "ic_positive_share": ic_pos,
                "mean_long_short_return": ls_mean,
                "long_short_std": ls_std,
                "long_short_t_stat": ls_t,
                "long_short_positive_share": ls_pos,
                "long_short_days": ls_days,
                "long_short_equity": float(ls_curve.iloc[-1]) if len(ls_curve) else float("nan"),
                "mean_long_short_return_net": ls_net_mean,
                "long_short_net_std": ls_net_std,
                "long_short_net_t_stat": ls_net_t,
                "long_short_net_positive_share": ls_net_pos,
                "long_short_net_days": ls_net_days,
                "long_short_net_equity": float(ls_net_curve.iloc[-1]) if len(ls_net_curve) else float("nan"),
                "avg_turnover": float(group["turnover"].mean()) if len(group) else float("nan"),
                "avg_names": float(group["name_count"].mean()) if len(group) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_stability_summary(summary: pd.DataFrame) -> pd.DataFrame:
    train = summary[summary["split"] == "train"].copy()
    val = summary[summary["split"] == "val"].copy()
    merged = val.merge(
        train,
        on=["signal", "family", "component"],
        suffixes=("_val", "_train"),
        how="inner",
    )
    if merged.empty:
        return merged
    merged["same_ic_sign"] = np.sign(merged["mean_ic_val"]) == np.sign(merged["mean_ic_train"])
    merged["ic_retention"] = np.where(
        merged["mean_ic_train"].abs() > 1e-12,
        merged["mean_ic_val"].abs() / merged["mean_ic_train"].abs(),
        np.nan,
    )
    merged["selection_score"] = (
        merged["mean_ic_val"].abs()
        * np.clip(merged["ic_t_stat_val"].abs() / 2.0, 0.0, 1.5)
        * np.where(merged["same_ic_sign"], 1.0, 0.35)
        * np.clip(merged["long_short_net_t_stat_val"].abs() / 1.5, 0.0, 1.5)
    )
    return merged.sort_values("selection_score", ascending=False, kind="stable")


def write_plot(output_dir: Path, stability: pd.DataFrame) -> None:
    if stability.empty:
        return
    val = stability.sort_values("mean_ic_val", ascending=True, kind="stable").tail(24)
    fig, axis = plt.subplots(figsize=(11, max(5, 0.32 * len(val))))
    colors = ["#386641" if value > 0 else "#bc4749" for value in val["mean_ic_val"]]
    axis.barh(val["signal"], val["mean_ic_val"], color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Validation mean daily Spearman IC")
    axis.set_title("Ichimoku Cycle Signal Validation IC")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_mean_ic_top24.png", dpi=170)
    plt.close(fig)


def write_markdown(output_dir: Path, stability: pd.DataFrame, args: argparse.Namespace, specs: tuple[IchimokuSpec, ...]) -> None:
    top = stability.head(18).copy()
    user_signal = stability[stability["family"] == "ichi_vn_8_22_44_22_22"].head(8)
    lines = [
        "# Ichimoku Cycle Signal Search",
        "",
        "Scope: VN train/validation only from the anchor run. Test/out-sample data is not used.",
        "",
        "Implementation note: Senkou spans are evaluated as causal cloud state known at date `t`; the charting-only forward displacement is not used as future information. Chikou is represented as lagged momentum.",
        "",
        f"Minimum names per day: `{args.min_names_per_day}`. Top/bottom quantile: `{args.top_quantile:.0%}`.",
        f"Cost proxy: `{args.cost_bps:.1f}` bps per traded notional, using daily top/bottom portfolio turnover.",
        "",
        "## Candidate Windows",
        "",
        "| Name | Conversion | Base | Span B | Displacement | Chikou |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in specs:
        lines.append(
            f"| `{spec.name}` | {spec.conversion} | {spec.base} | {spec.span_b} | {spec.displacement} | {spec.chikou} |"
        )

    lines.extend(
        [
            "",
            "## Best Validation Signals",
            "",
            "| Rank | Signal | Val IC | Val IC t | Val +days | Val net LS mean | Val net LS t | Val net equity | Avg turnover | Train IC | Same sign |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            "| "
            f"{rank} | `{row['signal']}` | {float(row['mean_ic_val']):+.4f} | {float(row['ic_t_stat_val']):+.2f} | "
            f"{float(row['ic_positive_share_val']):.1%} | {float(row['mean_long_short_return_net_val']):+.5f} | "
            f"{float(row['long_short_net_t_stat_val']):+.2f} | {float(row['long_short_net_equity_val']):.3f} | "
            f"{float(row['avg_turnover_val']):.2f} | "
            f"{float(row['mean_ic_train']):+.4f} | `{bool(row['same_ic_sign'])}` |"
        )

    lines.extend(
        [
            "",
            "## User 8/22/44/22/22 Variant",
            "",
            "| Signal | Val IC | Val IC t | Val +days | Val net LS mean | Val net LS t | Val net equity | Avg turnover | Train IC | Same sign |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in user_signal.iterrows():
        lines.append(
            "| "
            f"`{row['signal']}` | {float(row['mean_ic_val']):+.4f} | {float(row['ic_t_stat_val']):+.2f} | "
            f"{float(row['ic_positive_share_val']):.1%} | {float(row['mean_long_short_return_net_val']):+.5f} | "
            f"{float(row['long_short_net_t_stat_val']):+.2f} | {float(row['long_short_net_equity_val']):.3f} | "
            f"{float(row['avg_turnover_val']):.2f} | "
            f"{float(row['mean_ic_train']):+.4f} | `{bool(row['same_ic_sign'])}` |"
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Prefer signals with validation IC above zero, positive long-short spread, and same train/validation IC sign.",
            "- Treat component-level wins as feature candidates, not proof that changing all default windows will improve the LSTM.",
            "- If a signal passes this screen, the next step is a narrow feature ablation against the current `general_sector_full` anchor.",
            "",
            "![Validation mean IC top 24](validation_mean_ic_top24.png)",
        ]
    )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = RUN_ROOT / args.anchor_run
    run_config = resolve_run_config(run_dir)
    eval_config = EvalConfig(min_names_per_day=args.min_names_per_day, top_quantile=args.top_quantile, cost_bps=args.cost_bps)
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_market_frame(Path(str(run_config["data_path"])), run_config.get("stocks"))
    df = split_frame(df, run_config)
    target_column = str(run_config["target_column"])
    required = {"Date", "code", "high", "low", "close", target_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    with_signals = add_ichimoku_signals(df, DEFAULT_SPECS)
    signal_catalog = build_signal_catalog(with_signals)
    daily = daily_signal_metrics(with_signals, signal_catalog, target_column, eval_config)
    summary = summarize_daily(daily)
    stability = build_stability_summary(summary)

    signal_catalog.to_csv(output_dir / "signal_catalog.csv", index=False)
    daily.to_csv(output_dir / "daily_signal_metrics.csv", index=False)
    summary.to_csv(output_dir / "signal_summary_by_split.csv", index=False)
    stability.to_csv(output_dir / "stable_validation_summary.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "anchor_run": args.anchor_run,
                "data_path": run_config["data_path"],
                "train_end_date": run_config["train_end_date"],
                "val_end_date": run_config["val_end_date"],
                "target_column": target_column,
                "eval_config": eval_config.__dict__,
                "specs": [spec.__dict__ for spec in DEFAULT_SPECS],
                "top_validation": stability.head(20).to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_plot(output_dir, stability)
    write_markdown(output_dir, stability, args, DEFAULT_SPECS)
    print(json.dumps({"output_dir": str(output_dir), "signals": int(len(signal_catalog))}, indent=2))


if __name__ == "__main__":
    main()
