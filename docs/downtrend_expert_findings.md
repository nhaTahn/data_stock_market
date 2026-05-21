# Bao Cao Thuc Nghiem: Downtrend Expert

Updated: 2026-04-27

## 1. Muc Tieu

Danh gia viec xay dung mot "regime expert" rieng cho pha `downtrend`.

Y tuong da thu:

- Hard filtering: chi giu cac sequence co `target_date` thuoc `downtrend`.
- Loai bo hoan toan cac sequence khac khoi qua trinh train.
- Muc tieu la xem LSTM hoc rieng downtrend co tot hon anchor/router hien tai trong downtrend hay khong.

Ket luan hien tai: **chua hieu qua, khong nen chot hard-filter LSTM lam downtrend expert**.

Cap nhat 2026-04-27: da thu them sidecar va soft downtrend weighting. Ca hai deu khong du tot de dua vao active path.

## 2. Setup Da Chay

Run da kiem tra:

`data/processed/assets/data_info_vn/history/training_runs/downtrend_expert_phase_ic_sector19`

Thong tin thuc te cua run:

| Field | Value |
| --- | --- |
| Feature set | `phase_ic_sector19` |
| Regime filter | `downtrend` hard filter |
| Train sequences sau filter | `16,036` |
| Validation sequences sau filter | `6,253` |
| Validation downtrend dates | `68` |
| Window size | `5` |
| Loss | `rel_score_sharp` |
| LSTM units | `64` |
| Dropout | `0.1` |
| Seeds | `42, 52, 62, 72, 82` |
| Stock universe | `stocks=None` |
| Family | plain `lstm`, khong co `lstm_signmag` |

Ghi chu quan trong:

- `config.json` hien chua ghi lai `regime_filter`, nhung `predictions.csv` xac nhan toan bo train/val rows deu co regime `downtrend`.
- Day **khong phai A/B sach** voi anchor hien tai, vi anchor dang dung setup khac: window 15, `rel_score`, `lstm_signmag`, va feature set `general_sector_full`.
- Vi vay run nay chi ket luan duoc: **hard filtering + setup hien tai khong dat**. Chua du de ket luan moi hinh thuc downtrend expert deu vo ich.

## 3. Ket Qua Validation

Leaderboard cua run hard-filter:

| Model | Validation rel_score | Directional Accuracy | Quartile Equity | Nhan xet |
| --- | ---: | ---: | ---: | --- |
| `linear_regression` | `+0.0035` | `50.1%` | `1.245` | Baseline thang LSTM trong run nay |
| `arima` | `+0.0008` | `48.1%` | `1.278` | Yeu, nhung van hon LSTM theo rel_score |
| `lstm_best_by_val` | `-0.0062` | `48.4%` | `0.974` | Khong dat yeu cau |

Daily cross-sectional IC tren validation downtrend-only:

| Model | Mean daily IC | t-stat | Positive days | Nhan xet |
| --- | ---: | ---: | ---: | --- |
| `linear_regression` | `+0.0505` | `+2.28` | `57.4%` | Baseline don gian co ranking tot nhat trong run |
| `lstm_seed_42` | `+0.0423` | `+1.19` | `60.3%` | IC duong nhung t-stat yeu, rel_score rat xau |
| `lstm_ensemble` | `+0.0351` | `+1.38` | `58.8%` | Khong du manh |
| `lstm_best_by_val` | `+0.0047` | `+0.14` | `54.4%` | Gan nhu khong con edge |

Train vs validation cho thay overfit/variance:

| Model | Train mean IC | Validation mean IC | Nhan xet |
| --- | ---: | ---: | --- |
| `lstm_seed_42` | `+0.2211` | `+0.0423` | Suy giam manh, overfit cao |
| `lstm_ensemble` | `+0.1791` | `+0.0351` | Suy giam manh |
| `lstm_best_by_val` | `+0.0819` | `+0.0047` | Chon theo rel_score khong chon duoc IC tot |

## 4. So Sanh Voi Router Hien Tai

Report tham chieu:

`data/processed/assets/data_info_vn/history/training_runs/reports/cross_sectional_ic/anchor_sector19_cross_sectional_ic_20260426_r01`

Trong validation downtrend:

| Candidate | Downtrend mean IC | t-stat | Positive days |
| --- | ---: | ---: | ---: |
| Anchor | `-0.0066` | `-0.20` | `45.7%` |
| Challenger / sector19 in downtrend | `+0.0243` | `+0.84` | `54.3%` |
| `sector19_down_up_anchor_else` | `+0.0243` | `+0.84` | `54.3%` |

Doc ban dau ghi best seed downtrend expert `IC=+0.0215`, worst seed `IC=-0.1761`. Sau khi audit truc tiep `predictions.csv`, daily IC theo cach tinh cross-sectional cho thay mot vai seed co IC duong, nhung:

- `lstm_best_by_val` gan nhu khong co IC.
- Seed co IC tot lai co rel_score/equity xau.
- Baseline linear regression thang LSTM trong chinh run hard-filter.
- Ket qua khong on dinh giua seed va giua metric.

Ket luan khong doi: **hard-filter LSTM chua hieu qua**.

## 5. Bai Hoc

### 5.1. Data starvation

Hard filtering cat bo khoang 75% sequence. LSTM mat phan lon du lieu de hoc pattern nen, noise, volatility, bounce, va relation giua price action voi return.

Trong time-series DL, chia data theo regime qua manh thuong lam:

- giam sample size;
- tang variance giua seed;
- lam model hoc pattern cuc bo cua mot vai dot downtrend;
- lam train IC dep nhung validation IC roi manh.

### 5.2. Setup hien tai bi mismatch voi anchor

Run hard-filter dang dung `window_size=5` va `rel_score_sharp`, trong khi cac smoke gan day da cho thay:

- window 10 va 20 deu kem window 15;
- `rel_score_sharp` khong tot cho prediction anchor;
- `lstm_signmag` la family dang dang tin hon plain LSTM.

Vi vay neu muon test hard filtering lan nua, can A/B sach hon.

### 5.3. Ranking co tin hieu, calibration chua tot

Mot vai seed co IC duong, nhung rel_score/equity xau. Dieu nay khop voi ket luan truoc: edge hien tai co ve nam o cross-sectional ranking hon la raw return calibration.

## 6. Follow-up Da Thu Ngay 2026-04-27

### 6.1. Downtrend sidecar

Report:

`data/processed/assets/data_info_vn/history/training_runs/reports/downtrend_sidecar/anchor_downtrend_sidecar_20260427_r01`

Ket qua:

| Candidate | Validation rel_score | All-regime equity | Downtrend IC | Nhan xet |
| --- | ---: | ---: | ---: | --- |
| `sector19_down_anchor_else` | `+0.0038` | `3.924` | `+0.0243` | Baseline router can vuot |
| `downtrend_lstm_seed_42_else_anchor` | `-0.0001` | `4.115` | `+0.0239` | Equity tang nhe nhung rel_score hong |
| `downtrend_lstm_ensemble_else_anchor` | `+0.0060` | `3.869` | `+0.0182` | Rel_score tot hon anchor, nhung downtrend IC/equity thua sector19 router |
| `downtrend_linear_else_anchor` | `+0.0043` | `3.605` | `-0.0065` | Linear sidecar khong align duoc voi router hien tai |

Ket luan: sidecar co vai dau hieu prediction-side, nhung khong giai quyet downtrend ranking tot hon sector19 router. Khong dua vao active path.

### 6.2. Soft downtrend weighting

Report da giu lai:

`data/processed/assets/data_info_vn/history/training_runs/reports/feature_pruning/broad_signmag_prune_20260427_r02`

Regime report da giu lai:

`data/processed/assets/data_info_vn/history/training_runs/reports/regime_analysis/downtrend_focus_smoke_regime_20260427_r02`

Ba run heavy da bi xoa sau khi tong hop vi khong hieu qua.

| Case | Validation rel_score | Signmag quartile equity | Downtrend IC | Nhan xet |
| --- | ---: | ---: | ---: | --- |
| Anchor `general_sector_full` | `+0.0053` | `3.241` regime-recomputed | `-0.0066` | Van la standalone prediction anchor |
| `downtrend_w15` | `+0.0015` | `2.418` regime-recomputed | `+0.0116` | Downtrend IC tang nhe, nhung tong the hong |
| `downtrend_w20` | `+0.0024` | `2.969` regime-recomputed | `-0.0058` | Gan anchor ve IC tong the, nhung khong sua downtrend |
| `downtrend_w30` | `+0.0043` | `2.008` regime-recomputed | `-0.0186` | Rel_score gan hon, nhung trade/downtrend xau |

Ket luan: **rollback code `downtrend_focus`; khong giu lam feature training chinh**.

## 7. De Xuat Lam Them

### Uu tien 1: A/B sach cho hard filtering

Neu van muon kiem tra hard filtering cong bang hon, chay lai cung setup voi anchor/challenger:

- `window_size=15`
- `loss=rel_score`
- `lstm_units=64,32`
- bat `lstm_signmag`
- cung universe VN30/broad dang dung cho anchor
- dung `phase_ic_sector19` hoac `general_sector_full`

Neu A/B sach van thua, moi nen archive huong hard-filter hoan toan. Sau ket qua soft weighting am, A/B nay chi nen lam neu can xac nhan, khong con la uu tien cao.

### Uu tien 2: Rank/portfolio objective

Huong co co so hon la chuyen muc tieu ve ranking:

- daily Spearman IC;
- top-bottom quartile return;
- differentiable pairwise rank loss hoac portfolio spread loss;
- theo doi worst-year equity va drawdown.

Khong nen chi toi uu raw `rel_score` neu muc tieu thuc te la ranking de trade.

### Uu tien 3: Regime-conditioned router, khong train them expert

Viec train rieng expert chua thuyet phuc. Huong tot hon la hoc/validate cong thuc router dua tren:

- regime;
- rolling IC cua tung model tren train only;
- volatility/breadth filter;
- worst-year constraint.

Neu router khong vuot `sector19_down_up_anchor_else`, quay lai anchor/sector19 hien tai.

## 8. Khong Nen Lam Ngay

- Khong chot hard-filter LSTM lam downtrend expert.
- Khong giu soft downtrend weighting lam code path chinh.
- Khong mo out-sample.
- Khong tang seed/epoch cho hard-filter setup hien tai.
- Khong chon seed tot nhat theo IC neu seed do co `rel_score` hoac equity xau.
- Khong them feature moi neu chua chi ra no tang IC trong downtrend.

## 9. Ket Luan

Downtrend expert theo hard filtering va soft weighting **chua dat**.

Huong co co so hon la:

1. Giu anchor `general_sector_full` va router `sector19_down_up_anchor_else`.
2. Chuyen sang rank/portfolio objective neu muon tiep tuc cai thien LSTM.
3. Hoc router theo rolling IC/regime tren train only, thay vi train them downtrend expert.
4. Chi A/B hard-filter sach neu can xac nhan truoc khi archive hoan toan.
