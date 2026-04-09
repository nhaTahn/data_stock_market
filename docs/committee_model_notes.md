# Committee Model Notes

Tài liệu này ghi lại thử nghiệm đầu tiên cho hướng `hội đồng model` thay vì chỉ dùng một mini-group riêng lẻ hoặc một shared model duy nhất.

## Ý tưởng

Committee tối giản hiện tại:

- một `expert model` chuyên cho cụm mạnh
- một `market/sector context model` rộng hơn
- ghép dự báo của hai model bằng rule đơn giản và chọn cấu hình theo `val rel_score`

Ở vòng đầu, tôi chưa train thêm `VN30/VN100 shared model`.
Thay vào đó, tôi dùng ngay các artifact out-of-sample đã có để test nguyên lý:

- `sector-wide model` đóng vai trò context
- `mini-group model` đóng vai trò expert

Điều này đủ để trả lời câu hỏi quan trọng nhất:

- committee có giúp `rel_score` tốt hơn không?

## Script

- [`src/research/committee_relscore_experiment.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/research/committee_relscore_experiment.py)

Script này:

- lấy `predictions.csv` từ hai run đã có
- merge theo `split`, `code`, `Date`
- search weight trên `val`
- thử các rule đơn giản như:
  - `avg`
  - `agree_only`
- chấm bằng `rel_score` thật của repo

## Kết quả vòng 1

### F&B: thành công rõ

Expert run:

- `mini_tpdouong_g06_uncertainty_sidecar`

Context run:

- `sector_thuc_pham_va_o_uong_return_w5_relscore`

Best committee theo `val`:

- expert model: `lstm_best_by_val`
- context model: `lstm_signmag_best_by_val`
- method: `avg`
- weight expert: `0.80`

Result trên overlap `KDC,SAB,SBT,VNM`:

- expert overlap test `rel_score`: `0.05048`
- context overlap test `rel_score`: `0.01407`
- committee val `rel_score`: `0.02987`
- committee test `rel_score`: `0.05242`

Artifacts:

- [`best_committee_summary.json`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/committee_experiments/fnb_sector_plus_mini/best_committee_summary.json)
- [`committee_grid_results.csv`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/committee_experiments/fnb_sector_plus_mini/committee_grid_results.csv)

Kết luận:

- committee không chỉ giữ được mốc `> 0.03`
- nó còn cải thiện thêm trên overlap test của case mạnh nhất

### BĐS: có tiềm năng nhưng chưa ổn định

Expert run:

- `mini_bat_ong_san_g01_return_w20_pruned_v2`

Context run:

- `sector_bat_ong_san_return_w5_relscore`

Best committee theo `val`:

- committee test `rel_score`: `0.02381`

Artifacts:

- [`best_committee_summary.json`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/committee_experiments/bds_sector_plus_mini/best_committee_summary.json)

Nhưng nếu nhìn top theo `test`, có các cấu hình vượt `0.04`.
Vấn đề là các cấu hình đó có `val rel_score` rất thấp hoặc gần 0, nên chưa đủ sạch để tin.

Kết luận:

- committee BĐS có thể nâng ceiling
- nhưng hiện chưa giải được bài toán stability theo validation

## Ý nghĩa cho roadmap

Điều rút ra từ vòng này:

- hướng `hội đồng model` là đáng làm tiếp
- không cần nhảy ngay sang giant LSTM
- bước đúng hơn là:
  - `shared context model`
  - `sector expert`
  - `simple combiner`

## Bước kế tiếp nên làm

1. Train một `shared_vn30_relscore` model làm tầng context thật sự.
2. Lặp lại committee experiment với:
   - `shared_vn30 + F&B expert`
   - `shared_vn30 + BĐS expert`
   - `shared_vn30 + Bank expert`
3. Chỉ khi `val-selected committee` ổn định ở nhiều case mới mở rộng lên `VN100`.

## Runner cho phase 2

Để bắt đầu hướng này mà không sửa thêm `run_train.py`, repo đã có runner riêng:

- [`scripts/run_shared_vn30_committee.py`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_shared_vn30_committee.py)

Runner này làm 2 việc:

- train một `shared_vn30` context model bằng chính pipeline hiện tại
- sau đó chạy committee experiment với các expert preset như `fnb`, `bds`, `bank`

Universe VN30 chuẩn được đọc từ:

- [`market_lists/vn30.txt`](/Users/lap15111/Documents/research-paper/data_stock_market/market_lists/vn30.txt)

Mặc định phase 1 đang cố ý giữ nhỏ:

- `target_mode=return`
- `loss=rel_score`
- `window_size=20`
- `lstm_units=64,32`
- `target_normalizer=volatility_20`
- `lstm_seeds=42,52,62`
- committee preset mặc định: `fnb`, `bds`

Ví dụ:

```bash
venv/bin/python scripts/run_shared_vn30_committee.py
```

Hoặc chỉ chạy committee trên một context run đã có:

```bash
venv/bin/python scripts/run_shared_vn30_committee.py \
  --skip-train \
  --context-run-dir data/processed/assets/data_info_vn/history/training_runs/shared_vn30_return_w20_relscore_YYYYMMDD_HHMMSS \
  --committee-preset fnb \
  --committee-preset bds
```

Artifacts tổng hợp của suite sẽ được lưu lại ngay trong context run:

- `reports/core/committee_suite_manifest.json`
- `reports/core/committee_suite_summary.csv`

## Overnight plan

Nếu muốn treo máy và lấy kết quả để đọc sau, dùng:

- [`scripts/run_shared_context_overnight.sh`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_shared_context_overnight.sh)

Batch này cố ý chỉ search trên 2 trục:

- universe: `VN30`, `VN100`
- `window_size`: `20`, `60`

Các tham số khác đang giữ cố định để tránh phình thêm:

- `loss=rel_score`
- `lstm_units=64,32`
- `target_normalizer=volatility_20`
- `lstm_seeds=42,52,62`
- committee presets: `bank`, `bds`, `fnb`

Lý do của matrix này:

- `VN30` kiểm tra context tập trung ở nhóm trụ
- `VN100` kiểm tra coverage rộng hơn cho expert mini-group
- `window=20` cho regime ngắn-vừa
- `window=60` cho context chậm hơn của thị trường

Sáng hôm sau nên đọc theo thứ tự:

1. log-level summary:
   - `overnight_logs/<stamp>_shared_context/shared_context_committee_stable_ge3codes.csv`
2. nếu có candidate tốt:
   - `shared_context_committee_all.csv`
3. sau đó mới mở từng run:
   - `reports/core/metrics.json`
   - `reports/core/committee_suite_summary.csv`

Tiêu chí đọc đúng:

- ưu tiên committee có `code_count >= 3`
- ưu tiên cấu hình có `committee_val_rel_score > 0`
- mục tiêu thực tế là `committee_test_rel_score > 0.03` nhưng chỉ có ý nghĩa khi overlap không quá hẹp

## Baseline confirmation

Sau overnight batch hiện tại, candidate mạnh nhất là:

- `VN100`
- `window=20`
- preset `fnb`

Repo có runner xác nhận riêng:

- [`scripts/run_vn100_fnb_committee_confirmation.py`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_vn100_fnb_committee_confirmation.py)

Runner này:

- cố định cấu hình winner hiện tại
- rerun một hoặc nhiều lần
- ghi `confirmation_summary.csv` dưới:
  - `data/processed/assets/data_info_vn/history/training_runs/reports/committee_confirmations/`

Mục đích là trả lời một câu duy nhất:

- winner `VN100 + F&B committee` có lặp lại đủ ổn định không?

## Stable weight search

Committee experiment giờ xuất thêm:

- `committee_stability_summary.csv`

File này tóm tắt theo từng cặp:

- `expert_model`
- `market_model`
- `method`

và cho biết:

- `best_weight_expert`
- `stable_weight_min`
- `stable_weight_max`
- `stable_weight_count`
- `stable_test_rel_score_mean`
- `stable_test_rel_score_median`

Mục đích là tránh đọc `weight_expert` như một điểm tối ưu quá mong manh. Nếu một band hẹp quanh điểm tốt vẫn giữ được `val rel_score` gần như nhau, band đó đáng tin hơn một spike đơn lẻ.

Kết quả đang đáng giữ nhất hiện tại là:

- expert: `mini_tpdouong_g06_uncertainty_sidecar / lstm_best_by_val`
- market: `confirm_vn100_fnb_committee_20260408_235445_r01 / lstm_signmag_best_by_val`
- method: `agree_only`
- chosen weight: `0.15`
- stable band: `0.15 -> 0.90`
- stable weight count: `5`
- chosen-point `test rel_score`: `0.0493`
- stable-band `test rel_score` median: khoảng `0.0516`

## Sector rotation layer

Shared runner giờ xuất thêm:

- `reports/core/committee_rotation_all.csv`
- `reports/core/committee_rotation_active.csv`

Ý nghĩa:

- `committee_rotation_all.csv`: toàn bộ preset committees đã đánh giá
- `committee_rotation_active.csv`: chỉ giữ preset đạt ngưỡng kích hoạt

Ngưỡng mặc định hiện tại:

- `code_count >= 3`
- `committee_val_rel_score >= 0.015`
- `stable_weight_count >= 2`
- `stable_test_rel_score_median >= 0`

Preset đang có:

- `fnb`
- `bank`
- `bds`
- `chung`

Ghi chú:

- `chung` hiện vẫn là preset tạm thời, dùng run `Dịch vụ tài chính` làm proxy cho nhóm chứng khoán
- nếu sau này có mini-group chứng sạch hơn, chỉ cần đổi `expert_run_name` của preset này
- sau khi siết ngưỡng, file `committee_rotation_active.csv` hiện chỉ còn lại `fnb`

## Context + residual proxy

Đã có một proxy rất nhỏ để kiểm tra xem việc tách `market context` và `expert residual` có giúp ngay không:

- [`src/research/context_residual_proxy_experiment.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/research/context_residual_proxy_experiment.py)

Script này chỉ dùng prediction đã có để fit một lớp tuyến tính trên `val`, rồi chấm lại bằng `rel_score`. Nó hữu ích để trả lời nhanh câu hỏi "có đáng đi tiếp theo hướng residual không?" mà chưa cần sửa pipeline train.

Kết quả F&B hiện tại:

- best candidate vẫn là `expert_only`
- `market_plus_residual_affine` chỉ đạt test khoảng `0.0281`
- `two_signal_ols` chỉ quanh `0.0212`

Kết luận hiện tại:

- `context + residual` vẫn đáng nghiên cứu
- nhưng phải là **retrain target residual thật sự**
- post-hoc linear combiner chưa đủ để thắng committee đang có

## Residual retrain experiment

Đã có runner train residual expert thật sự:

- [`scripts/run_context_residual_expert.py`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_context_residual_expert.py)

Ý tưởng:

- target mới của expert = `actual - market_prediction`
- train lại `plain lstm` trên residual target
- chọn seed theo `combined val rel_score` sau khi cộng ngược `market + residual`
- chấm cuối cùng vẫn bằng `rel_score` trên `actual`

Kết quả F&B hiện tại không tốt:

- với market `lstm_signmag_best_by_val`:
  - run: [`residual_fnb_vn100_context_signmag_r01`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/residual_fnb_vn100_context_signmag_r01)
  - best combined test `rel_score`: `0.0037`
  - ensemble combined test `rel_score`: `0.0033`
- với market `lstm_best_by_val`:
  - run: [`residual_fnb_vn100_context_plain_r01`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/residual_fnb_vn100_context_plain_r01)
  - best combined test `rel_score`: `0.0103`
  - ensemble combined test `rel_score`: `0.0133`

So với expert gốc:

- `expert_only` vẫn khoảng `0.0505`

Kết luận hiện tại:

- residual retrain kiểu cộng tuyến tính `market + residual expert` đang làm mô hình tệ đi
- bệnh gốc vẫn là bias âm / underreaction, residual target hiện chưa sửa được điều đó
- nếu còn theo hướng này, bước sau phải là:
  - cross-fitted residual targets cho train
  - hoặc regime gating trước khi residualize
  - không nên coi `market + residual expert` hiện tại là baseline mới
