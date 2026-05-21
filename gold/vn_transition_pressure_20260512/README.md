# VN Transition Pressure Gold Freeze 2026-05-12

This folder freezes the current VN-first reporting candidate.

## Frozen Candidate

```text
base LSTM forecast
-> LSTM tradeability filter
-> committee selector
-> Wyckoff phase gate
-> transition pressure risk filter
-> holding-period execution
```

The two learned components are both LSTM-based. The committee and risk gates do not replace LSTM; they control when and how the LSTM signal is used.

Frozen policy:

```text
vn_legacy_acc_all_else_transition_pressure_nonneg
```

Advisor-facing execution variant:

```text
min_positions = 6
cost_bps = 15
validation = strict non-overlap rolling 126/21/21
```

## Why This Is Gold

The policy stays positive across four VN validation artifacts and passes the strict risk-control screen:

| Metric | Value |
| --- | ---: |
| artifacts positive | 4 / 4 |
| worst-artifact equity | 1.186 |
| mean equity | 1.460 |
| minimum Sharpe | +0.55 |
| worst max drawdown | -21.7% |
| maximum turnover | 0.179 |

Holdout/test is not used in this freeze.

## Key Files

| Path | Purpose |
| --- | --- |
| `advisor_report_format.md` | Report written in the advisor/math discussion format. |
| `feature_processing_and_standard_eval_reply.md` | Reply draft for feature-processing clarification and Tan's standard evaluation checklist. |
| `mathematical_report.md` | Short academic-style explanation of the frozen model. |
| `freeze_config.json` | Machine-readable freeze metadata. |
| `reports/vn_transition_pressure_gate_report_20260512.md` | Detailed research report. |
| `plots/error_hist_report_v2/summary.md` | Error histogram report in the old reporting style, clipped to the central 99% for readability. |
| `plots/next_day_prediction_plots/summary.md` | Base LSTM next-day return prediction plots across VN codes. |
| `artifacts/cross_artifact_m06/summary.md` | Four-artifact stability summary for the conservative policy. |
| `artifacts/cross_artifact_m06/cross_artifact_summary.csv` | Cross-artifact aggregate metrics. |
| `artifacts/cross_artifact_m06/cross_artifact_details.csv` | Artifact-level metrics. |
| `artifacts/fresh_no_leader_filter/summary.json` | Fresh no-leader filter artifact metadata. |
| `artifacts/fresh_no_leader_gate_m06/gate_map.json` | Frozen gate mapping. |

## Frozen Decision

Use `min_positions=6` for report and risk-control discussion.

Keep `min_positions=5` only as a research reference because it has higher return but misses the turnover screen by about `0.001`.

Do not tune more parameters before holdout. The next valid step is a leakage audit and then one controlled holdout read.
