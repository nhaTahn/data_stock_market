from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "matplotlib is required for model benchmark dashboard rendering.\n"
        "Install it in your environment, for example: pip install matplotlib"
    ) from exc


LOWER_IS_BETTER = {"mae", "rmse", "mape_pct"}


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_benchmark_run(run_dir: Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    summary_df = pd.read_csv(run_dir / "benchmark_summary.csv")
    test_df = summary_df[summary_df["split"] == "test"].copy()
    if test_df.empty:
        return {"run_type": "benchmark", "run_name": run_dir.name}

    metric = "rmse" if "rmse" in test_df.columns else test_df.columns[2]
    ascending = metric in LOWER_IS_BETTER
    best_row = test_df.sort_values(metric, ascending=ascending).iloc[0]
    summary: dict[str, object] = {
        "run_type": "benchmark",
        "run_name": run_dir.name,
        "best_model": best_row["model"],
        f"best_{metric}": _safe_float(best_row[metric]),
    }
    for candidate in ["directional_accuracy", "thresholded_directional_accuracy", "mae", "rmse"]:
        if candidate in best_row.index:
            summary[f"best_model_{candidate}"] = _safe_float(best_row[candidate])
    return summary


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else "nan"
    return str(value)


def _plot_overview(summary_df: pd.DataFrame, output_path: Path) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy()
    metric_candidates = [col for col in ["mae", "rmse", "directional_accuracy", "thresholded_directional_accuracy"] if col in test_df.columns]
    metric_candidates = metric_candidates[:4]
    rows = 2 if len(metric_candidates) > 2 else 1
    cols = 2 if len(metric_candidates) > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, metric in zip(axes_list, metric_candidates):
        ordered = test_df.sort_values(metric, ascending=(metric in LOWER_IS_BETTER))
        ax.bar(ordered["model"], ordered[metric], color="#2563eb")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
    for ax in axes_list[len(metric_candidates):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_transformer_fit(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in history_df.columns:
        ax.plot(history_df.index + 1, history_df[col], label=col)
    ax.set_title("Transformer Fit History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_dashboard(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    summary_df = pd.read_csv(run_dir / "benchmark_summary.csv")
    test_df = summary_df[summary_df["split"] == "test"].copy()
    compact_columns = [
        col
        for col in [
            "model",
            "mae",
            "rmse",
            "directional_accuracy",
            "thresholded_directional_accuracy",
        ]
        if col in test_df.columns
    ]
    compact_test_df = test_df[compact_columns].copy() if compact_columns else test_df.copy()

    overview_plot = run_dir / "dashboard_overview.png"
    _plot_overview(summary_df, overview_plot)

    transformer_fit_plot = None
    transformer_history_path = run_dir / "transformer_fit_history.csv"
    if transformer_history_path.exists():
        transformer_fit_plot = run_dir / "dashboard_transformer_fit.png"
        _plot_transformer_fit(pd.read_csv(transformer_history_path), transformer_fit_plot)

    image_tags = [f'<img src="{overview_plot.name}" alt="Benchmark overview">']
    for candidate in [
        "benchmark_test_mae.png",
        "benchmark_test_rmse.png",
        "benchmark_test_directional_accuracy.png",
        "benchmark_test_thresholded_directional_accuracy.png",
    ]:
        candidate_path = run_dir / candidate
        if candidate_path.exists():
            image_tags.append(f'<img src="{candidate_path.name}" alt="{candidate}">')
    if transformer_fit_plot is not None:
        image_tags.append(f'<img src="{transformer_fit_plot.name}" alt="Transformer fit history">')

    summary = summarize_benchmark_run(run_dir)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Benchmark Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: linear-gradient(135deg, #7c2d12, #c2410c); color: white; padding: 24px; border-radius: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 20px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .kpi {{ background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.2); border-radius: 14px; padding: 14px; }}
    .kpi-label {{ font-size: 12px; text-transform: uppercase; opacity: 0.85; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .card {{ background: white; border: 1px solid #dbe4ee; border-radius: 16px; padding: 18px; }}
    .full {{ grid-column: 1 / -1; }}
    img {{ width: 100%; border-radius: 10px; border: 1px solid #dbe4ee; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 12px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Benchmark Dashboard</h1>
      <p>{run_dir}</p>
      <div class="kpis">
        <div class="kpi"><div class="kpi-label">Best Model</div><div class="kpi-value">{summary.get('best_model', '-')}</div></div>
        <div class="kpi"><div class="kpi-label">Best RMSE</div><div class="kpi-value">{_format_value(summary.get('best_rmse', float('nan')))}</div></div>
        <div class="kpi"><div class="kpi-label">Dir Acc</div><div class="kpi-value">{_format_value(summary.get('best_model_directional_accuracy', float('nan')))}</div></div>
        <div class="kpi"><div class="kpi-label">Thr Dir Acc</div><div class="kpi-value">{_format_value(summary.get('best_model_thresholded_directional_accuracy', float('nan')))}</div></div>
      </div>
    </section>
    <section class="grid">
      <article class="card">
        <h2>Run Summary</h2>
        <div class="table-wrap"><table>{"".join(f"<tr><th>{k}</th><td>{_format_value(v)}</td></tr>" for k, v in summary.items() if k not in {'run_type', 'run_name'})}</table></div>
      </article>
      <article class="card">
        <h2>Test Split</h2>
        <div class="table-wrap">{compact_test_df.to_html(index=False, float_format=lambda x: f"{x:.6f}", border=0)}</div>
      </article>
      <article class="card full">
        <h2>Charts</h2>
        {"".join(image_tags)}
      </article>
      <article class="card">
        <h2>Quick Read</h2>
        <p>Start with the overview chart and compare only the test split. The full cross-split table is usually more useful for debugging than for decision-making.</p>
      </article>
      <article class="card">
        <h2>Raw Files</h2>
        <p><a href="benchmark_summary.csv">benchmark_summary.csv</a></p>
        <p><a href="transformer_metrics.csv">transformer_metrics.csv</a></p>
        <p><a href="baseline_metrics.csv">baseline_metrics.csv</a></p>
        <p><a href="transformer_report.txt">transformer_report.txt</a></p>
      </article>
    </section>
  </div>
</body>
</html>
"""
    dashboard_path = run_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path
