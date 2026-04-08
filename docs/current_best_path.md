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
