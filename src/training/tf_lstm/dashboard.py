from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "matplotlib is required for tf_lstm dashboard rendering.\n"
        "Install it in your environment, for example: pip install matplotlib"
    ) from exc


LOWER_IS_BETTER = {"mae", "rmse", "mape_pct", "loss", "val_loss"}


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_tf_run(run_dir: Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    baseline_df = pd.read_csv(run_dir / "baseline_metrics.csv")
    lstm_df = pd.read_csv(run_dir / "lstm_metrics.csv")

    baseline_test = baseline_df[baseline_df["split"] == "test"].iloc[0].to_dict()
    lstm_test = lstm_df[lstm_df["split"] == "test"].iloc[0].to_dict()

    summary: dict[str, object] = {
        "run_type": "tf_lstm",
        "run_name": run_dir.name,
        "test_baseline_mae": _safe_float(baseline_test.get("mae")),
        "test_model_mae": _safe_float(lstm_test.get("mae")),
        "test_baseline_rmse": _safe_float(baseline_test.get("rmse")),
        "test_model_rmse": _safe_float(lstm_test.get("rmse")),
    }

    for metric in ["mae", "rmse", "mape_pct", "directional_accuracy", "thresholded_directional_accuracy"]:
        if metric in baseline_test and metric in lstm_test:
            baseline_value = _safe_float(baseline_test[metric])
            model_value = _safe_float(lstm_test[metric])
            if metric in LOWER_IS_BETTER:
                improvement = baseline_value - model_value
            else:
                improvement = model_value - baseline_value
            summary[f"test_{metric}_improvement"] = improvement

    return summary


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else "nan"
    return str(value)


def _metric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col != "split"]


def _build_comparison_df(baseline_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    merged = baseline_df.merge(model_df, on="split", suffixes=("_baseline", "_model"))
    rows: list[dict[str, object]] = []

    for _, row in merged.iterrows():
        entry: dict[str, object] = {"split": row["split"]}
        for metric in _metric_columns(baseline_df):
            baseline_value = _safe_float(row.get(f"{metric}_baseline"))
            model_value = _safe_float(row.get(f"{metric}_model"))
            if metric in LOWER_IS_BETTER:
                delta = baseline_value - model_value
            else:
                delta = model_value - baseline_value
            entry[f"{metric}_baseline"] = baseline_value
            entry[f"{metric}_model"] = model_value
            entry[f"{metric}_delta"] = delta
        rows.append(entry)

    return pd.DataFrame(rows)


def _plot_fit_history(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in history_df.columns:
        ax.plot(history_df.index + 1, history_df[col], label=col)
    ax.set_title("Fit History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_metric_grid(comparison_df: pd.DataFrame, metrics: list[str], output_path: Path) -> None:
    metrics = metrics[:4]
    rows = 2 if len(metrics) > 2 else 1
    cols = 2 if len(metrics) > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    x = range(len(comparison_df))
    labels = comparison_df["split"].tolist()
    for ax, metric in zip(axes_list, metrics):
        ax.bar([i - 0.18 for i in x], comparison_df[f"{metric}_baseline"], width=0.36, label="baseline", color="#94a3b8")
        ax.bar([i + 0.18 for i in x], comparison_df[f"{metric}_model"], width=0.36, label="model", color="#2563eb")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_title(metric)
    for ax in axes_list[len(metrics):]:
        ax.axis("off")
    axes_list[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_dashboard(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    baseline_df = pd.read_csv(run_dir / "baseline_metrics.csv")
    model_df = pd.read_csv(run_dir / "lstm_metrics.csv")
    history_df = pd.read_csv(run_dir / "fit_history.csv")

    comparison_df = _build_comparison_df(baseline_df, model_df)
    fit_plot = run_dir / "dashboard_fit_history.png"
    metric_plot = run_dir / "dashboard_metric_grid.png"

    _plot_fit_history(history_df, fit_plot)
    _plot_metric_grid(comparison_df, _metric_columns(baseline_df), metric_plot)

    test_summary = summarize_tf_run(run_dir)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TF LSTM Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: linear-gradient(135deg, #0f172a, #1d4ed8); color: white; padding: 24px; border-radius: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 20px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .kpi {{ background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.2); border-radius: 14px; padding: 14px; }}
    .kpi-label {{ font-size: 12px; text-transform: uppercase; opacity: 0.85; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .card {{ background: white; border: 1px solid #dbe4ee; border-radius: 16px; padding: 18px; }}
    .full {{ grid-column: 1 / -1; }}
    img {{ width: 100%; border-radius: 10px; border: 1px solid #dbe4ee; }}
    table {{ width: 100%; border-collapse: collapse; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 12px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>TF LSTM Regression Dashboard</h1>
      <p>{run_dir}</p>
      <div class="kpis">
        <div class="kpi"><div class="kpi-label">Test MAE</div><div class="kpi-value">{_format_value(test_summary.get('test_model_mae'))}</div></div>
        <div class="kpi"><div class="kpi-label">MAE Delta</div><div class="kpi-value">{_format_value(test_summary.get('test_mae_improvement', float('nan')))}</div></div>
        <div class="kpi"><div class="kpi-label">Test RMSE</div><div class="kpi-value">{_format_value(test_summary.get('test_model_rmse'))}</div></div>
        <div class="kpi"><div class="kpi-label">RMSE Delta</div><div class="kpi-value">{_format_value(test_summary.get('test_rmse_improvement', float('nan')))}</div></div>
      </div>
    </section>
    <section class="grid">
      <article class="card">
        <h2>Test Summary</h2>
        <div class="table-wrap"><table>{"".join(f"<tr><th>{k}</th><td>{_format_value(v)}</td></tr>" for k, v in test_summary.items() if k not in {'run_type', 'run_name'})}</table></div>
      </article>
      <article class="card">
        <h2>Quick Read</h2>
        <p>Negative delta on `mae` or `rmse` means the LSTM underperformed the baseline.</p>
        <p>Use the charts first. Open the raw files only when a run looks worth digging into.</p>
      </article>
      <article class="card full">
        <h2>Fit History</h2>
        <img src="{fit_plot.name}" alt="Fit history">
      </article>
      <article class="card full">
        <h2>Metric Grid</h2>
        <img src="{metric_plot.name}" alt="Metric grid">
      </article>
      <article class="card">
        <h2>Split Snapshot</h2>
        <div class="table-wrap">{comparison_df.to_html(index=False, float_format=lambda x: f"{x:.6f}", border=0)}</div>
      </article>
      <article class="card">
        <h2>Raw Files</h2>
        <p><a href="baseline_metrics.csv">baseline_metrics.csv</a></p>
        <p><a href="lstm_metrics.csv">lstm_metrics.csv</a></p>
        <p><a href="fit_history.csv">fit_history.csv</a></p>
        <p><a href="training_report.txt">training_report.txt</a></p>
      </article>
    </section>
  </div>
</body>
</html>
"""
    dashboard_path = run_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path
