from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_results(
    output_dir: Path,
    baseline_metrics: dict[str, dict[str, float]],
    lstm_metrics: dict[str, dict[str, float]],
    config: dict[str, object],
    history_df: pd.DataFrame,
    model,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.DataFrame(
        [{"split": split_name, **metrics} for split_name, metrics in baseline_metrics.items()]
    )
    lstm_df = pd.DataFrame([{"split": split_name, **metrics} for split_name, metrics in lstm_metrics.items()])

    baseline_df.to_csv(output_dir / "baseline_metrics.csv", index=False)
    lstm_df.to_csv(output_dir / "lstm_metrics.csv", index=False)
    history_df.to_csv(output_dir / "fit_history.csv", index=False)
    model.save(output_dir / "best_model.keras")

    report_lines = [
        "TensorFlow/Keras LSTM Next-Day Adjusted Return Training",
        "",
        "Problem definition:",
        "X_t = [adjust_t, volume_t] over a historical window.",
        "y_t = adjusted return on the next trading day.",
        "",
        "Baseline 1:",
        "Predict next-day return = 0, equivalent to reusing today's adjusted price.",
        "Thresholded directional accuracy uses |return| > 0.002.",
        "",
        "Configuration:",
    ]
    report_lines.extend([f"{key}: {value}" for key, value in config.items()])
    report_lines.extend(
        [
            "",
            "Baseline metrics:",
            baseline_df.to_string(index=False),
            "",
            "LSTM metrics:",
            lstm_df.to_string(index=False),
        ]
    )
    (output_dir / "training_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
