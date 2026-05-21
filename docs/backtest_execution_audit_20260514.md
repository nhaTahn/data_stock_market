# Backtest Execution Audit — 2026-05-14

Scope: verify rằng prediction tại close `t` không bị pair với return của chính ngày `t` (instant-fill leak), mà chỉ được realize ở ngày `t+1` trở đi.

## Code paths đã audit

| File | Hàm | Vai trò |
| --- | --- | --- |
| [src/data_pipeline/quality_dataset_core.py](../src/data_pipeline/quality_dataset_core.py) | target build | `next_adjust = adjust.shift(-1); target_next_return = next_adjust / adjust - 1` |
| [src/models/training/datasets.py:53-54](../src/models/training/datasets.py:53) | `build_sequence_dataset` | `y_list.append(target_values[end_idx])` — target tại row `t` = return `D[t] → D[t+1]` |
| [src/models/training/pipeline.py:1314](../src/models/training/pipeline.py:1314) | `targets = y_split` | targets được pass nguyên sang prediction frame |
| [src/models/training/pipeline.py:679](../src/models/training/pipeline.py:679) | `enrich_prediction_frame` | `frame["actual"] = target_at_row_t` — CSV row Date=`D[t]` có `actual = return D[t]→D[t+1]` |
| [src/evaluation/metric.py:34-51](../src/evaluation/metric.py:34) | `_align_single_group` | drop row 0, sau đó pair `predict[i-1] ↔ actual[i]` |
| [experiments/analysis/analyze_lstm_filter_signal.py:364-374](../experiments/analysis/analyze_lstm_filter_signal.py:364) | `align_signal_actual` | `signal_rows = group.iloc[1:-1]`, `actual_rows = group.iloc[2:]`, `actual_aligned = actual_rows["actual"]` |
| [experiments/analysis/analyze_prediction_router.py:53-65](../experiments/analysis/analyze_prediction_router.py:53) | `_align_single_prediction_frame` | giống align_signal_actual |
| [src/models/selection/holding_period.py:79-88](../src/models/selection/holding_period.py:79) | `simulate_rebalance` | group theo `actual_date`, dùng `day["actual_aligned"]` làm return realized cùng `actual_date` |

## Convention thực tế

Pipeline lưu prediction tại Date `D[t]` với semantic:

```
prediction[D[t]] = model.predict(features ≤ D[t])  → forecast của return D[t]→D[t+1]
actual[D[t]]     = realized return D[t]→D[t+1]      (= target_next_return)
```

Tới analysis/backtest, có một bước **shift +1**:

```
signal_date    = D[t]
actual_date    = D[t+1]
actual_aligned = actual[D[t+1]] = realized return D[t+1]→D[t+2]
prediction     = prediction[D[t]]
```

→ prediction tại `D[t]` được pair với realized return ở **chu kỳ `t+1 → t+2`**, không phải `t → t+1`.

## Kết luận

**✓ NO EXECUTION LEAK.** Convention này conservative: model dự báo period `t→t+1` nhưng đánh giá/backtest trên period `t+1→t+2`. Tương đương giả định "đặt lệnh chậm 1 ngày" (entry sau khi đã thấy close `t+1`).

**⚠️ TRAIN/EVAL HORIZON MISMATCH.** Tuy không leak, prediction được optimize cho `D[t]→D[t+1]` nhưng evaluate trên `D[t+1]→D[t+2]`. Hệ quả:

- `rel_score = +0.0049` đo trên future-of-future, không phải on-target.
- IC `+0.054` cũng vậy.
- Edge thực sự trên on-target horizon có thể **lớn hơn** số đang báo cáo, hoặc **nhỏ hơn** nếu model chỉ predict được short-term mean reversion.

Bằng chứng convention là **intentional, không phải bug**:

- `metric.py` và `align_signal_actual` đều dùng cùng pattern (`iloc[1:-1]`, `iloc[2:]`).
- Convention nhất quán giữa train/val/test/backtest, nên tất cả số reported đều trên cùng một thước đo.
- Đây thường là cách backtest realistic: giả định không thể trade tại close `t` vì prediction được tính trên data ≤ close `t`.

## Đề xuất

1. **Document convention rõ trong README/code**: thêm docstring cho `_align_single_group` và `align_signal_actual` giải thích semantic 1-day shift.
2. **Cân nhắc A/B horizon mismatch**: chạy một experiment paired test:
   - Variant A (current): train target_next_return, eval với 1-day shift.
   - Variant B (on-target): train target_next_return, eval không shift (trực tiếp `actual[D[t]]`).
   - Mục tiêu: hiểu rel_score thực sự của model trên horizon nó được train.
3. **Synthetic oracle leak test** (R2 verification): `experiments/analysis/synthetic_oracle_leak_test.py` đã viết — inject `prediction = actual_aligned`, run `simulate_rebalance`, expect Sharpe → ∞. Nếu Sharpe hữu hạn → có leak. Nếu Sharpe `=NaN` hoặc inf → alignment đúng và backtest có execution lag.
