from __future__ import annotations

import math
import json
from pathlib import Path

import pandas as pd

from fk_lstm_classifier.evaluation import compute_classification_metrics

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "matplotlib is required for fk_lstm_classifier reporting.\n"
        "Install it in your environment, for example: pip install matplotlib"
    ) from exc


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def summarize_strategy(strategy_df: pd.DataFrame) -> dict[str, float]:
    if strategy_df.empty:
        return {
            "days": 0.0,
            "mean_daily_return": float("nan"),
            "vol_daily_return": float("nan"),
            "annualized_sharpe": float("nan"),
            "hit_rate": float("nan"),
            "max_drawdown": float("nan"),
            "final_equity": float("nan"),
            "mean_turnover": float("nan"),
            "total_transaction_cost": float("nan"),
        }

    returns = strategy_df["strategy_return"]
    vol = float(returns.std(ddof=0))
    sharpe = math.sqrt(252.0) * float(returns.mean()) / vol if vol > 0 else float("nan")
    return {
        "days": float(len(strategy_df)),
        "mean_daily_return": float(returns.mean()),
        "vol_daily_return": vol,
        "annualized_sharpe": float(sharpe),
        "hit_rate": float((returns > 0).mean()),
        "max_drawdown": _max_drawdown(strategy_df["equity_curve"]),
        "final_equity": float(strategy_df["equity_curve"].iloc[-1]),
        "mean_turnover": float(strategy_df["turnover"].mean()) if "turnover" in strategy_df else float("nan"),
        "total_transaction_cost": (
            float(strategy_df["transaction_cost"].sum()) if "transaction_cost" in strategy_df else float("nan")
        ),
    }


def summarize_fk_run(run_dir: Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    strategy_df = pd.read_csv(run_dir / "validation_long_short_returns.csv")
    predictions_df = pd.read_csv(run_dir / "validation_predictions.csv")
    market_rules_path = run_dir / "market_rules.json"
    market_rules = {}
    if market_rules_path.exists():
        market_rules = json.loads(market_rules_path.read_text(encoding="utf-8"))
    metrics = compute_classification_metrics(
        labels=predictions_df["target_class"].to_numpy(),
        prob_class_1=predictions_df["prob_class_1"].to_numpy(),
    )
    strategy_summary = summarize_strategy(strategy_df)
    return {
        "run_type": "fk",
        "run_name": run_dir.name,
        "market_scope": market_rules.get("market_scope", "unknown"),
        "forward_horizon_days": market_rules.get("forward_horizon_days", float("nan")),
        "allow_short": market_rules.get("allow_short", True),
        "strategy_mode": market_rules.get("strategy_mode", "unknown"),
        **metrics,
        **strategy_summary,
    }


def build_prob_decile_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    ranked = predictions_df.copy()
    ranked["prob_decile"] = pd.qcut(
        ranked["prob_class_1"].rank(method="first"),
        q=10,
        labels=False,
    ) + 1
    grouped = (
        ranked.groupby("prob_decile")
        .agg(
            count=("code", "size"),
            mean_prob_class_1=("prob_class_1", "mean"),
            positive_rate=("target_class", "mean"),
            mean_next_return=("next_return", "mean"),
        )
        .reset_index()
    )
    return grouped


def _plot_fit_history(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df.index + 1, history_df["loss"], label="train")
    axes[0].plot(history_df.index + 1, history_df["val_loss"], label="validation")
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical Crossentropy")
    axes[0].legend()

    axes[1].plot(history_df.index + 1, history_df["accuracy"], label="train accuracy")
    axes[1].plot(history_df.index + 1, history_df["val_accuracy"], label="validation accuracy")
    axes[1].plot(history_df.index + 1, history_df["auc"], label="train auc", linestyle="--")
    axes[1].plot(history_df.index + 1, history_df["val_auc"], label="validation auc", linestyle="--")
    axes[1].set_title("Accuracy and AUC by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_equity_and_drawdown(strategy_df: pd.DataFrame, output_path: Path) -> None:
    running_peak = strategy_df["equity_curve"].cummax()
    drawdown = strategy_df["equity_curve"] / running_peak - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(pd.to_datetime(strategy_df["realized_date"]), strategy_df["equity_curve"], color="#2563eb")
    axes[0].set_title("Long-Short Equity Curve")
    axes[0].set_ylabel("Equity")

    axes[1].fill_between(
        pd.to_datetime(strategy_df["realized_date"]),
        drawdown,
        0,
        color="#dc2626",
        alpha=0.35,
    )
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_return_distribution(strategy_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(strategy_df["strategy_return"], bins=40, color="#0f766e", alpha=0.85)
    axes[0].set_title("Daily Strategy Return Distribution")
    axes[0].set_xlabel("Daily return")

    rolling = strategy_df["strategy_return"].rolling(21).mean()
    axes[1].plot(pd.to_datetime(strategy_df["realized_date"]), rolling, color="#7c3aed")
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1].set_title("21-Day Rolling Mean Return")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_decile_returns(decile_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(decile_df["prob_decile"].astype(str), decile_df["mean_next_return"], color="#2563eb")
    axes[0].set_title("Mean Next Return by Probability Decile")
    axes[0].set_xlabel("Predicted probability decile")
    axes[0].set_ylabel("Mean next return")

    axes[1].bar(decile_df["prob_decile"].astype(str), decile_df["positive_rate"], color="#ea580c")
    axes[1].set_title("Positive Class Rate by Probability Decile")
    axes[1].set_xlabel("Predicted probability decile")
    axes[1].set_ylabel("Observed positive rate")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _format_metric_table(metrics: dict[str, float]) -> str:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, float):
            display = f"{value:.6f}" if math.isfinite(value) else "nan"
        else:
            display = str(value)
        rows.append(f"<tr><th>{key}</th><td>{display}</td></tr>")
    return "\n".join(rows)


def render_dashboard(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    history_df = pd.read_csv(run_dir / "fit_history.csv")
    strategy_df = pd.read_csv(run_dir / "validation_long_short_returns.csv")
    predictions_df = pd.read_csv(run_dir / "validation_predictions.csv")
    market_rules = {}
    market_rules_path = run_dir / "market_rules.json"
    if market_rules_path.exists():
        market_rules = json.loads(market_rules_path.read_text(encoding="utf-8"))
    strategy_heading = "Long-Short Strategy Summary" if market_rules.get("allow_short", True) else "Long-Only Strategy Summary"
    equity_heading = "Long-Short Equity Curve" if market_rules.get("allow_short", True) else "Long-Only Equity Curve"

    metrics = compute_classification_metrics(
        labels=predictions_df["target_class"].to_numpy(),
        prob_class_1=predictions_df["prob_class_1"].to_numpy(),
    )
    strategy_summary = summarize_strategy(strategy_df)
    decile_df = build_prob_decile_table(predictions_df)

    fit_plot = run_dir / "dashboard_fit_history.png"
    equity_plot = run_dir / "dashboard_equity_drawdown.png"
    return_plot = run_dir / "dashboard_return_distribution.png"
    decile_plot = run_dir / "dashboard_deciles.png"

    _plot_fit_history(history_df, fit_plot)
    _plot_equity_and_drawdown(strategy_df, equity_plot)
    _plot_return_distribution(strategy_df, return_plot)
    _plot_decile_returns(decile_df, decile_plot)
    top_deciles = decile_df[["prob_decile", "positive_rate", "mean_next_return"]].tail(3).copy()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FK LSTM Dashboard</title>
  <style>
    :root {{
      --bg: #f6f6f1;
      --card: #ffffff;
      --text: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #f8fafc, var(--bg));
      color: var(--text);
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      padding: 24px;
      border-radius: 20px;
      background: linear-gradient(135deg, #0f766e, #1d4ed8);
      color: white;
      margin-bottom: 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .kpi {{
      background: rgba(255,255,255,0.14);
      border: 1px solid rgba(255,255,255,0.2);
      border-radius: 14px;
      padding: 14px;
    }}
    .kpi-label {{
      font-size: 12px;
      opacity: 0.85;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .kpi-value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }}
    h1, h2, h3 {{
      margin-top: 0;
    }}
    img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid #e5e7eb;
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    .muted {{
      color: var(--muted);
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>FK LSTM Validation Dashboard</h1>
      <p class="muted">Run directory: {run_dir}</p>
      <div class="kpis">
        <div class="kpi"><div class="kpi-label">Accuracy</div><div class="kpi-value">{metrics['accuracy']:.3f}</div></div>
        <div class="kpi"><div class="kpi-label">AUC Rank</div><div class="kpi-value">{metrics['auc_rank']:.3f}</div></div>
        <div class="kpi"><div class="kpi-label">Final Equity</div><div class="kpi-value">{strategy_summary['final_equity']:.3f}</div></div>
        <div class="kpi"><div class="kpi-label">Sharpe</div><div class="kpi-value">{strategy_summary['annualized_sharpe']:.3f}</div></div>
      </div>
    </section>
    <section class="grid">
      <article class="card">
        <h2>Classification Metrics</h2>
        <div class="table-wrap"><table>{_format_metric_table(metrics)}</table></div>
      </article>
      <article class="card">
        <h2>{strategy_heading}</h2>
        <div class="table-wrap"><table>{_format_metric_table(strategy_summary)}</table></div>
      </article>
      <article class="card">
        <h2>Market Rules</h2>
        <div class="table-wrap"><table>{_format_metric_table(market_rules)}</table></div>
      </article>
      <article class="card full">
        <h2>Fit History</h2>
        <img src="{fit_plot.name}" alt="Fit history">
      </article>
      <article class="card full">
        <h2>{equity_heading} and Drawdown</h2>
        <img src="{equity_plot.name}" alt="Equity curve and drawdown">
      </article>
      <article class="card">
        <h2>Return Distribution</h2>
        <img src="{return_plot.name}" alt="Return distribution">
      </article>
      <article class="card full">
        <h2>Signal Shape</h2>
        <img src="{decile_plot.name}" alt="Decile analysis">
      </article>
      <article class="card">
        <h2>Top Decile Snapshot</h2>
        <div class="table-wrap">{top_deciles.to_html(index=False, float_format=lambda x: f"{x:.6f}", border=0)}</div>
      </article>
      <article class="card">
        <h2>Raw Files</h2>
        <p><a href="fit_history.csv">fit_history.csv</a></p>
        <p><a href="validation_long_short_returns.csv">validation_long_short_returns.csv</a></p>
        <p><a href="validation_predictions.csv">validation_predictions.csv</a></p>
        <p><a href="market_rules.json">market_rules.json</a></p>
      </article>
    </section>
  </div>
</body>
</html>
"""
    dashboard_path = run_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path
