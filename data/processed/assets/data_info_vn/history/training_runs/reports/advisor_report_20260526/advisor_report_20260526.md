# Báo Cáo Plot Cập Nhật — VN Hetero Combined + Portfolio Overlay

**Ngày**: 2026-05-26  
**Phạm vi**: VN validation only (`2020-04-01 → 2022-11-15`). Holdout/test chưa sử dụng.  
**Prediction anchor**: `hetero_combined_full5 → ensemble_mean_cal_each_traincal_clip`.  
**Portfolio overlay**: diagnostic overlay grid trên daily policy returns từ `hetero_long_finetune_batch_20260522`.

---

## 1. Metrics Tổng Hợp

| Component | Metric |
| --- | ---: |
| Prediction anchor rel_score | **0.04478** |
| Prediction anchor DA | **51.83%** |
| Q90(|E|) validation | **4.69%** |
| Static best overlay Sharpe | **3.25** |
| Static best overlay max DD | **-12.0%** |
| Static best overlay final equity | **5.29×** |
| Walk-forward overlay selector Sharpe | **2.19** |
| Walk-forward overlay selector max DD | **-12.5%** |

Ghi chú: Static overlay dùng để xem diagnostic/upper-bound trên validation; walk-forward selector chọn mỗi block 21 ngày bằng lịch sử trước đó trong validation.

---

## 2. Histogram Sai Số

![Error histogram](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/advisor_report_20260526/error_histogram.png)

---

## 3. Time-Series Q90(|E|)

![Q90 time series](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/advisor_report_20260526/timeseries_q90_validation.png)

---

## 4. Portfolio Overlay Equity / Drawdown

![Equity overlay comparison](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/advisor_report_20260526/equity_overlay_comparison.png)

---

## 5. Model Metrics Panel

![Metrics panel](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/advisor_report_20260526/metrics_panel.png)

---

## 6. Artifacts

- `error_histogram.png`
- `timeseries_q90_validation.png`
- `equity_overlay_comparison.png`
- `metrics_panel.png`
- Gold mirror: `/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/advisor_report_20260526/`
