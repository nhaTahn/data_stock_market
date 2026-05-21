# Tra Loi Ve Feature Processing Va Standard Evaluation

## 1. Lam ro feature processing

Trong bao cao, dong:

```text
X_process = F(X_raw)
```

nen duoc hieu la toan bo pipeline bien doi du lieu goc thanh chuoi feature dua vao LSTM. Cu the:

```text
X_raw
-> clean/sort theo code, Date
-> tao feature gia/khoi luong/ky thuat/context
-> map market feature de portable
-> scale feature bang scaler fit tren train
-> tao rolling window 15 ngay
-> X_process_{i,t} in R^{15 x 29}
```

Trong do `F(.)` gom cac buoc sau:

1. **Lam sach va canh thoi gian**
   - Sap xep theo `code`, `Date`.
   - Moi row ung voi mot co phieu `i` tai ngay giao dich `t`.
   - Target la return ngay tiep theo:

```text
y_{i,t+1} = r_{i,t+1}
```

2. **Tao feature tu gia va khoi luong**
   - Nhom level 20 ngay:
     `open_level_20`, `high_level_20`, `low_level_20`, `close_level_20`, `volume_level_20`.
   - Nhom bien dong 1 ngay:
     `open_delta_1`, `high_delta_1`, `low_delta_1`, `close_delta_1`, `volume_delta_1`.
   - Nhom hinh dang nen:
     `intraday_return`, `gap_open`, `close_position`.

3. **Tao feature ky thuat**
   - `bb_width`
   - `volume_ratio_20`
   - `volatility_20`
   - `momentum_20`
   - `macd_hist`

4. **Tao feature context thi truong**
   - O VN hien tai:

```text
vnindex_return = return cua chi so dai dien thi truong VN
```

   - Khi chuyen sang thi truong khac, logic nay duoc tong quat hoa thanh:

```text
vnindex_return -> market_proxy_return_1
```

   - Vi du:
     - JP: TOPIX hoac Nikkei return.
     - KR: KOSPI return.
     - US: S&P500 return hoac equal-weight market return.

5. **Tao feature sector va breadth**
   - `sector_momentum_rank`
   - `sector_momentum_rank_pct`
   - `sector_momentum_20`
   - `relative_sector_momentum_20`
   - `sector_return`
   - `alpha_sector`
   - `sector_positive_ratio`
   - `sector_ad_ratio`
   - `a_d_ratio`

6. **Feature lich**
   - `day_of_week`

7. **Chuan hoa**
   - Feature scaler duoc fit tren tap train.
   - Sau do dung cung scaler de bien doi validation/test.
   - Khong fit scaler tren toan bo du lieu de tranh leakage.

8. **Tao input sequence cho LSTM**
   - Voi moi co phieu `i`, tai ngay `t`, lay 15 ngay gan nhat:

```text
X_process_{i,t} = [x_{i,t-14}, ..., x_{i,t}]
X_process_{i,t} in R^{15 x 29}
```

   - Model dung `X_process_{i,t}` de du bao `y_{i,t+1}`.
   - Nghia la input chi dung thong tin den ngay `t`, target la ngay `t+1`.

## 2. Cau tra loi ngan co the dua vao bao cao

```text
Feature processing F(.) la qua trinh bien doi du lieu OHLCV/context tho thanh tensor chuoi cho LSTM.
Moi feature tai ngay t chi duoc tinh tu du lieu tai ngay t hoac qua khu, sau do scaler duoc fit tren train va apply sang validation/test.
Input cuoi cung cua base LSTM la X_process_{i,t} in R^{15 x 29}, gom 15 ngay nhin lai va 29 feature sau xu ly.
Target la next-day return y_{i,t+1}, nen setup nay giu dung cau truc time-series va tranh dung truc tiep du lieu tuong lai trong feature.
```

## 3. Huong dan Tan lam standard evaluation

Muc tieu cua standard evaluation la chot xem model freeze hien tai co can improve tiep hay khong. Khong duoc tiep tuc tune theo holdout/test.

### 3.1. Candidate can evaluate

Base model:

```text
Run = broad_signmag_portable_no_identity_20260428_allvn_r01
Model = lstm_seed_52
Checkpoint = model_seed_52.keras
```

Gold policy:

```text
Policy = vn_legacy_acc_all_else_transition_pressure_nonneg
Execution = min_positions = 6
Cost = 15 bps
```

### 3.2. Split dung de bao cao

Dung standard VN split:

```text
train      = [..., 2020-03-31]
in_sample  = [2020-04-01, 2022-11-15]
out_sample = hidden holdout, chi mo mot lan khi da freeze
```

Trong buoc hien tai:

```text
Khong mo out_sample/holdout.
```

### 3.3. Can chay/kiem tra nhung artifact nao

1. **Base LSTM evaluation**
   - `rel_score`
   - `base_score`
   - `abs_score`
   - `directional_accuracy`
   - histogram `E = prediction - actual`

2. **Filter LSTM evaluation**
   - So sanh `base_lstm` voi `lstm_filter_move_top_train_ic`.
   - Kiem tra `rel_score`, IC, quartile/equity neu co.

3. **Execution evaluation**
   - Dung policy freeze `min_positions=6`.
   - Kiem tra:
     - net equity
     - Sharpe
     - max drawdown
     - avg turnover
     - positive artifacts

4. **Leakage audit**
   - `feature scaler` fit train only.
   - `target` la `t+1`.
   - rolling features chi dung du lieu den `t`.
   - `pressure_delta_20` khong dung du lieu tuong lai.
   - Wyckoff/regime label khong duoc tinh bang future return.

### 3.4. Command goi y

Rebuild standard base report:

```bash
python3 main.py report update-run \
  data/processed/assets/data_info_vn/history/training_runs/broad_signmag_portable_no_identity_20260428_allvn_r01
```

Build histogram report trong gold:

```bash
venv/bin/python experiments/packaging/build_gold_vn_transition_error_hist_report.py \
  --output-name error_hist_report_v2
```

Doc ket qua:

```text
gold/vn_transition_pressure_20260512/plots/error_hist_report_v2/summary.md
gold/vn_transition_pressure_20260512/artifacts/cross_artifact_m06/summary.md
```

### 3.5. Tieu chi de chot co can improve nua khong

Neu tat ca dieu kien sau dat thi tam thoi khong improve tiep, chuyen sang leakage audit va holdout read:

| Nhom | Tieu chi |
| --- | --- |
| Base LSTM | `in_sample rel_score > 0` |
| Filter LSTM | filter/selector khong lam xau robust metric va co IC/quartile hop ly |
| Execution | net equity > 1 tren tat ca artifact |
| Risk | max drawdown khong xau hon `-25%` |
| Cost | avg turnover <= `0.20` sau cost `15 bps` |
| Stability | positive artifacts = `4 / 4` |
| Leakage | khong phat hien dung future data |

Neu fail mot trong cac diem tren:

- Neu fail leakage: sua pipeline truoc, khong train them.
- Neu fail base LSTM: moi quay lai feature/model training.
- Neu fail filter stability: uu tien multi-seed LSTM filter ensemble.
- Neu fail execution/risk: sua sizing/position rule, khong them raw feature vao base LSTM ngay.

## 4. Ket luan nen tra loi

```text
Hien tai phan feature processing can lam ro hon o buoc F(X_raw): day la pipeline tao feature causal, scale bang train-only scaler, va bien thanh tensor 15 x 29 cho LSTM.

De chot co can improve tiep hay khong, Tan nen chay standard evaluation tren candidate da freeze: base LSTM + LSTM filter + transition pressure gate, ban min_positions=6, cost 15 bps.

Neu standard evaluation va leakage audit deu pass, minh nen dung tune validation va moi mo holdout/test mot lan. Neu fail, moi quay lai improve theo dung diem fail, uu tien giu LSTM la core model.
```
