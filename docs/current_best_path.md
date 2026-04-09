# Current Best Path

Tài liệu này là phiên bản rút gọn của repo ở thời điểm hiện tại. Nếu muốn đọc ít nhưng đúng trọng tâm, bắt đầu từ đây.

## 1. Giữ lại gì

Hiện tại chỉ nên xem đây là đường chính:

- thị trường: VN
- bài toán: `target_mode=return`
- metric quyết định: `test rel_score`
- loss train mặc định: `rel_score`
- model family chính: `plain lstm`
- model family phụ đáng giữ: `lstm_signmag`
- đơn vị nghiên cứu chính: `mini-group`
- feature set: pool đã pruning theo correlation, sau đó chọn tập con theo group

## 2. Hạ xuống experimental

Các phần này chưa nên chi phối hướng phát triển tiếp:

- `lstm_quantile` như model family chính
- `lstm_attention`
- `lstm_event`
- Fischer-Krauss benchmark
- full sector pooling như phương án mặc định
- quá nhiều search trên hyperparameter cùng lúc

Ghi chú:

- quantile hiện chỉ đáng giữ như **sidecar** cho backtest qua `prediction_uncertainty = q90 - q50`
- với F&B mini-group mạnh nhất, `high spread` đang hữu ích hơn `low uncertainty`

## 3. Shortlist các run nên mở trước

### Run số 1: ổn định nhất hiện tại

`mini_tpdouong_g06_uncertainty_sidecar`

Lý do:

- `lstm_seed_52` vừa là `best by val` vừa là `best by test`
- `test rel_score = 0.0343`
- có thêm quantile sidecar để kiểm tra backtest

Nên đọc:

- `reports/core/config.json`
- `reports/core/metrics.json`
- `reports/core/family_selection_summary.json`
- `reports/backtests/threshold_backtest_summary_baseline_multi.json`
- `reports/backtests/threshold_backtest_summary_uncertainty_sidecar_multi_highspread.json`

### Run số 2: ceiling mạnh nhất theo test

`mini_bat_ong_san_g01_return_w20_pruned_v2`

Lý do:

- `lstm_signmag_seed_62` đạt `test rel_score = 0.0341`
- là ceiling mạnh nhất cho nhóm BĐS đang có

Cảnh báo:

- selection theo validation chưa ổn định
- nên dùng run này như bằng chứng tiềm năng, chưa phải template mặc định

### Run số 3: bản BĐS ổn định hơn để so sánh

`mini_bat_ong_san_g01_return_w15_pruned_v1`

Lý do:

- `best by val` và `best by test` gần nhau hơn bản `w20`
- dùng tốt để so sánh tác động của `window_size`

### Run số 4: best ngân hàng hiện tại

`mini_ngan_hang_g02_return_w20_pruned_v2`

Lý do:

- là mốc tốt nhất hiện có cho mini-group ngân hàng
- nhưng chưa vượt `0.03`

### Run số 5: best thực phẩm theo hướng pruned mini-group

`mini_thuc_pham_va_o_uong_g03_return_w20_pruned_v2`

Lý do:

- là case F&B gọn hơn để đọc khi không muốn nhìn run sidecar
- tín hiệu còn yếu nhưng dễ so hơn với ngân hàng

## 4. Cách đọc nhanh để không bị loãng

1. Mở `metrics.json`
2. Xem `best by val` và `best by test` có gần nhau không
3. Nếu ổn, mới mở `backtests`
4. Chỉ quay lại `predictions.csv` khi cần debug

## 5. Nếu chỉ giữ một command line để nghĩ tiếp

Đây là kiểu command line nên coi là baseline tinh gọn hiện tại:

```bash
venv/bin/python scripts/run_train.py \
  --target-mode return \
  --loss rel_score \
  --target-normalizer volatility_20 \
  --window-size 20 \
  --lstm-units 64,32 \
  --lstm-seeds 42,52,62,99
```

Sau đó chỉ thay:

- `--stocks`
- `--feature-columns` hoặc `--feature-selection-mode`
- có hoặc không dùng `signmag`

## 6. Điều chưa nên kết luận

Chưa nên kết luận rằng:

- pruning feature một mình là đủ
- `window=20` luôn tốt hơn `window=15`
- `signmag` luôn tốt hơn plain `lstm`
- `quantile` là family nên giữ như đường chính

Kết luận an toàn hơn là:

- `rel_score` là objective đúng để bám
- mini-group tốt hơn full sector trong nhiều case
- BĐS và F&B hiện là hai nơi cho tín hiệu đáng đọc nhất

## 7. Nhánh mở rộng đang thử

Nếu muốn đi ra khỏi `mini-group` nhưng chưa muốn nhảy vào một giant model cho toàn thị trường, nhánh hợp lý nhất hiện tại là:

- `shared market-context model`
- `sector / mini-group expert`
- `simple committee combiner`

Repo hiện đã có runner đầu tiên cho nhánh này:

- [`scripts/run_shared_vn30_committee.py`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_shared_vn30_committee.py)

Ý nghĩa của nhánh này:

- không thay `rel_score`
- không thay pipeline train chính
- chỉ thêm một tầng context chung `VN30` rồi xem khi ghép với expert thì `val-selected test rel_score` có vượt các mốc cũ không

Đây là bước trung gian hợp lý hơn giữa:

- `mini-group` quá cục bộ
- và `1 model lớn cho cả VN100` quá sớm

## 8. Baseline mới nên giữ

Sau batch overnight shared-context, baseline mở rộng đáng giữ nhất hiện tại là:

- `shared VN100`
- `window_size=20`
- expert: `F&B mini-group`
- committee nên chọn theo `stable band`, không nên chọn single-point theo `best val`:
  - expert model: `lstm_best_by_val`
  - market model: `lstm_signmag_best_by_val`
  - method: `agree_only`
  - `weight_expert=0.15`
  - overlap: `KDC,SAB,SBT,VNM`
  - stable weight band: `0.15 -> 0.90`
  - stable weight count: `5`
  - chosen-point `committee_test_rel_score = 0.0493`
  - stable-band `test rel_score` median khoảng `0.0516`

Artifacts gốc:

- [`best_committee_summary.json`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/committee_experiments/confirm_vn100_fnb_committee_20260408_235445_r01__committee__fnb/best_committee_summary.json)
- [`committee_rotation_active.csv`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/confirm_vn100_fnb_committee_20260408_235445_r01/reports/core/committee_rotation_active.csv)
- package báo cáo gọn cho giảng viên:
  - [`advisor_summary.md`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/advisor_shortlist/fnb_committee_best_20260409/advisor_summary.md)
  - [`manifest.json`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/advisor_shortlist/fnb_committee_best_20260409/manifest.json)

Điểm cần nhớ:

- context model standalone vẫn yếu
- giá trị hiện tại đến từ `shared VN100` đóng vai trò context cho expert F&B, không phải từ market model một mình
- `rotation_active` đã được siết lại để chỉ giữ preset có:
  - `code_count >= 3`
  - `committee_val_rel_score >= 0.015`
  - `stable_weight_count >= 2`
  - `stable_test_rel_score_median >= 0`
- nhánh `residual expert` đã thử và hiện chưa hiệu quả:
  - [`residual_fnb_vn100_context_signmag_r01`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/residual_fnb_vn100_context_signmag_r01)
  - [`residual_fnb_vn100_context_plain_r01`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/residual_fnb_vn100_context_plain_r01)

Runner xác nhận baseline:

- [`scripts/run_vn100_fnb_committee_confirmation.py`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_vn100_fnb_committee_confirmation.py)

Ví dụ:

```bash
venv/bin/python scripts/run_vn100_fnb_committee_confirmation.py --repeats 1
```

## 9. Nếu muốn trả lời câu hỏi "LSTM có hợp cho cả thị trường không?"

Không nên trả lời câu hỏi đó bằng một run duy nhất.

Tài liệu nên đọc là:

- [`whole_market_lstm_plan.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/whole_market_lstm_plan.md)

Watchlist sector tiếp theo:

- `Ngân hàng`
- `Dịch vụ tài chính`

## 10. Nếu muốn reset lại hướng F&B để cải thiện thật sự

Sau khi kiểm tra lại plot và bias, hướng nên đọc tiếp là:

- [`fnb_research_restart_plan.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/fnb_research_restart_plan.md)

Artifact tổng hợp candidate hiện tại:

- [`candidate_matrix.csv`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/research_restarts/fnb_restart_20260409/candidate_matrix.csv)
- [`candidate_matrix_summary.md`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/research_restarts/fnb_restart_20260409/candidate_matrix_summary.md)

Ý nghĩa:

- không tiếp tục tune mù quanh mọi nhánh
- giữ `sectorbase_committee_new` làm baseline decision layer
- giữ `plain_fnb_baseline` làm main standalone expert
- giữ `signmag_sector_base` làm auxiliary signal
- archive các candidate còn lại

Batch phase 1 hiện tại nên đọc tiếp ở:

- [`relscore_vn_walkthrough_phase1.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/relscore_vn_walkthrough_phase1.md)
- `Xây dựng và Vật liệu`
- `Điện, nước & xăng dầu khí đốt`
