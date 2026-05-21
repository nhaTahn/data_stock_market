# Models Code Map

Mục tiêu của tài liệu này là giúp đọc `src/models/` theo đúng luồng công việc hiện tại, sau khi đã tách các file lớn thành nhóm nhỏ hơn.

## 1. Điểm bắt đầu nên mở

Nếu đang đọc code train chính, đi theo thứ tự này:

1. [`main.py`](/Users/lap15111/Documents/research-paper/data_stock_market/main.py)
2. [`src/models/config.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/config.py)
3. [`src/models/training/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training)
4. [`src/models/architectures/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures)
5. [`src/models/components/losses.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/components/losses.py)
6. [`src/evaluation/metric.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/evaluation/metric.py)
7. [`src/models/reporting/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/reporting)
8. [`src/backtesting/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/backtesting)
9. [`experiments/`](/Users/lap15111/Documents/research-paper/data_stock_market/experiments)

## 2. Cấu trúc mới

### `src/models/architectures/`

Mỗi file là một family hoặc một phần backbone:

- [`backbone.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/backbone.py)
  Chỉ chứa backbone LSTM dùng chung.
- [`plain.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/plain.py)
  Plain LSTM một đầu ra.
- [`signmag.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/signmag.py)
  Family `sign + magnitude`.
- [`quantile.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/quantile.py)
  Head `q50/q90`.
- [`attention.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/attention.py)
  Attention family.
- [`event.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/event.py)
  Event-gated attention family.

### `src/models/training/`

Chỉ giữ logic train-time và data prep cho model:

- [`datasets.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/datasets.py)
  Build/split sequence dataset.
- [`scalers.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/scalers.py)
  Feature scaler, target scaler, local target normalizer.
- [`sample_weights.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/sample_weights.py)
  Sample-weight helpers.
- [`targets.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/targets.py)
  Build target dict cho `signmag` và `event`.
- [`prediction.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/prediction.py)
  Chuẩn hóa output `model.predict(...)`.
- [`seeds.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/seeds.py)
  Global random seed.
- [`fitters.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/fitters.py)
  Hàm train cho từng family.

### `src/models/reporting/`

- [`layout.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/reporting/layout.py)
  Toàn bộ logic map artifact vào `reports/core`, `reports/backtests`, `reports/plots`, ...

### `src/backtesting/`

- [`threshold_backtest.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/backtesting/threshold_backtest.py)
  Threshold backtest chính.
- [`multi_strategy_backtest.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/backtesting/multi_strategy_backtest.py)
  Portfolio backtest nhiều strategy.
- [`compare_equity_curves.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/backtesting/compare_equity_curves.py)
  So sánh equity curve giữa các run.
- [`summarize_horizon_backtests.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/backtesting/summarize_horizon_backtests.py)
  Tổng hợp backtest nhiều horizon/run.

### `experiments/`

- [`search/search_feature_combinations.py`](/Users/lap15111/Documents/research-paper/data_stock_market/experiments/search/search_feature_combinations.py)
  Search feature combinations bằng baseline tuyến tính.
- [`search/run_all_vn_feature_searches.py`](/Users/lap15111/Documents/research-paper/data_stock_market/experiments/search/run_all_vn_feature_searches.py)
  Batch search cho toàn bộ universe VN.
- [`search/summarize_vn_sector_features.py`](/Users/lap15111/Documents/research-paper/data_stock_market/experiments/search/summarize_vn_sector_features.py)
  Tổng hợp feature theo sector.
- [`analysis/compare_target_modes.py`](/Users/lap15111/Documents/research-paper/data_stock_market/experiments/analysis/compare_target_modes.py)
  So sánh `target_mode`.
- [`maintenance/archive_lstm_candidates.py`](/Users/lap15111/Documents/research-paper/data_stock_market/experiments/maintenance/archive_lstm_candidates.py)
  Gom shortlist run đạt ngưỡng.
- [`analysis/plot_run_trial_comparison.py`](/Users/lap15111/Documents/research-paper/data_stock_market/experiments/analysis/plot_run_trial_comparison.py)
  Plot so sánh nhiều run.

## 3. Các file cũ vẫn còn, nhưng chỉ là shim

Các path cũ vẫn được giữ để tránh gãy import cũ:

- [`src/models/sequence_utils.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/sequence_utils.py)
- [`src/models/trainer_wrapper.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/trainer_wrapper.py)
- [`src/models/report_layout.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/report_layout.py)
- [`src/models/dl_architectures/lstm_builder.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/dl_architectures/lstm_builder.py)

Nếu đang đọc để hiểu code, không nên bắt đầu từ các file shim này.

## 4. Đường đọc ngắn nhất theo câu hỏi

Nếu muốn biết:

- model train bằng loss gì:
  mở [`src/models/components/losses.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/components/losses.py)
- `signmag` là gì:
  mở [`src/models/architectures/signmag.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/signmag.py)
  rồi [`src/models/training/targets.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/targets.py)
- seed được dùng thế nào:
  mở [`src/models/training/seeds.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/seeds.py)
  rồi orchestration trong [`src/models/training/pipeline.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/pipeline.py)
- sequence được build ra sao:
  mở [`src/models/training/datasets.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/datasets.py)
- run sinh artifact nào:
  mở [`src/models/reporting/layout.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/reporting/layout.py)
- run backtest ở đâu:
  mở [`src/backtesting/threshold_backtest.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/backtesting/threshold_backtest.py)
- utility research ở đâu:
  mở [`experiments/`](/Users/lap15111/Documents/research-paper/data_stock_market/experiments)

## 5. Quy ước hiện tại nên giữ

- Plain `lstm` vẫn là family mặc định nên đọc trước.
- `signmag` là family phụ đáng giữ.
- `quantile`, `attention`, `event` là experimental.
- Khi thêm family mới, ưu tiên:
  - 1 file builder trong `architectures/`
  - 1 phần target/fit rõ ràng trong `training/`
  - không nhét thêm vào một file tổng hợp lớn.
