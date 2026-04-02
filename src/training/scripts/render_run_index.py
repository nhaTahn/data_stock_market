from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "matplotlib is required for run index rendering.\n"
        "Install it in your environment, for example: pip install matplotlib"
    ) from exc

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from fk_lstm_classifier.reporting import render_dashboard as render_fk_dashboard
from fk_lstm_classifier.reporting import summarize_fk_run
from model_benchmark.dashboard import render_dashboard as render_benchmark_dashboard
from model_benchmark.dashboard import summarize_benchmark_run
from tf_lstm.dashboard import render_dashboard as render_tf_dashboard
from tf_lstm.dashboard import summarize_tf_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a multi-run comparison index page.")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help='Comparison entry in the form "Label=path/to/run_dir". Repeat for multiple runs.',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "run_index" / "dashboard.html",
        help="Output HTML path for the aggregate dashboard.",
    )
    return parser.parse_args()


def _infer_run_type(run_dir: Path) -> str:
    if (run_dir / "validation_predictions.csv").exists():
        return "fk"
    if (run_dir / "benchmark_summary.csv").exists():
        return "benchmark"
    if (run_dir / "baseline_metrics.csv").exists() and (run_dir / "lstm_metrics.csv").exists():
        return "tf_lstm"
    raise ValueError(f"Could not infer run type for {run_dir}")


def _parse_entries(raw_entries: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for raw in raw_entries:
        if "=" not in raw:
            raise ValueError(f'Invalid --entry "{raw}". Expected "Label=path".')
        label, raw_path = raw.split("=", 1)
        parsed.append((label.strip(), Path(raw_path).expanduser()))
    return parsed


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else "nan"
    return str(value)


def _render_and_summarize(label: str, run_dir: Path) -> dict[str, object]:
    run_type = _infer_run_type(run_dir)
    if run_type == "fk":
        dashboard_path = render_fk_dashboard(run_dir)
        summary = summarize_fk_run(run_dir)
    elif run_type == "tf_lstm":
        dashboard_path = render_tf_dashboard(run_dir)
        summary = summarize_tf_run(run_dir)
    else:
        dashboard_path = render_benchmark_dashboard(run_dir)
        summary = summarize_benchmark_run(run_dir)

    return {
        "label": label,
        "run_dir": run_dir,
        "dashboard_path": dashboard_path,
        **summary,
    }


def _plot_fk_comparison(entries: list[dict[str, object]], output_path: Path) -> None:
    metrics = ["accuracy", "auc_rank", "final_equity", "annualized_sharpe"]
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
    axes_list = axes.flatten()

    labels = [str(entry["label"]) for entry in entries]
    for ax, metric in zip(axes_list, metrics):
        values = [float(entry.get(metric, float("nan"))) for entry in entries]
        ax.bar(labels, values, color="#2563eb")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_index(entries: list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_plot_tag = ""
    if entries and all(entry.get("run_type") == "fk" for entry in entries):
        comparison_plot = output_path.parent / "fk_comparison.png"
        _plot_fk_comparison(entries, comparison_plot)
        comparison_plot_tag = f"""
        <article class="card full">
          <h2>FK Run Comparison</h2>
          <img src="{comparison_plot.name}" alt="FK run comparison">
        </article>
        """

    output_parent = output_path.parent.resolve()
    cards = []
    for entry in entries:
        link_path = Path(entry["dashboard_path"]).resolve()
        link_href = Path(os.path.relpath(link_path, output_parent))
        summary_rows = "".join(
            f"<tr><th>{key}</th><td>{_format_value(value)}</td></tr>"
            for key, value in entry.items()
            if key not in {"label", "run_dir", "dashboard_path", "run_type", "run_name"}
        )
        cards.append(
            f"""
            <article class="card">
              <h2>{entry['label']}</h2>
              <p><strong>Type:</strong> {entry['run_type']}</p>
              <p><strong>Run dir:</strong> {entry['run_dir']}</p>
              <p><a href="{link_href}">Open run dashboard</a></p>
              <div class="table-wrap"><table>{summary_rows}</table></div>
            </article>
            """
        )

    summary_df = pd.DataFrame(entries)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Run Comparison Index</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: linear-gradient(135deg, #111827, #0284c7); color: white; padding: 24px; border-radius: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 20px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .kpi {{ background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.2); border-radius: 14px; padding: 14px; }}
    .kpi-label {{ font-size: 12px; text-transform: uppercase; opacity: 0.85; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .card {{ background: white; border: 1px solid #dbe4ee; border-radius: 16px; padding: 18px; }}
    .full {{ grid-column: 1 / -1; }}
    img {{ width: 100%; border-radius: 10px; border: 1px solid #dbe4ee; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 12px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Run Comparison Index</h1>
      <p>{len(entries)} runs</p>
      <div class="kpis">
        <div class="kpi"><div class="kpi-label">Runs</div><div class="kpi-value">{len(entries)}</div></div>
        <div class="kpi"><div class="kpi-label">Main FK Equity</div><div class="kpi-value">{_format_value(entries[0].get('final_equity', float('nan')) if entries else float('nan'))}</div></div>
        <div class="kpi"><div class="kpi-label">Main FK Sharpe</div><div class="kpi-value">{_format_value(entries[0].get('annualized_sharpe', float('nan')) if entries else float('nan'))}</div></div>
      </div>
    </section>
    <section class="grid">
      {comparison_plot_tag}
      {''.join(cards)}
      <article class="card full">
        <h2>How To Read This Page</h2>
        <p>Use each card as a headline summary and open the linked run dashboard only for the runs worth investigating. The raw summary table is intentionally removed from the default view.</p>
      </article>
    </section>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    parsed_entries = _parse_entries(args.entry)
    entries = [_render_and_summarize(label, run_dir) for label, run_dir in parsed_entries]
    output_path = render_index(entries, args.output)
    print(f"Run index saved to: {output_path}")


if __name__ == "__main__":
    main()
