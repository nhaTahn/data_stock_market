# Bao Cao Dong Bang Mo Hinh VN - Transition Pressure Gate

## 1. Tong quan

Model tot nhat hien tai de bao cao van lay LSTM lam loi chinh, nhung khong chi la mot LSTM don le. Kien truc hien tai co hai tang hoc bang LSTM:

```text
Base LSTM = du bao return
Filter LSTM = danh gia du bao co tradeable khong
```

Sau hai tang LSTM moi den cac lop chon lenh va kiem soat rui ro:

```text
Base LSTM forecast
-> LSTM filter signal
-> committee selector
-> Wyckoff phase gate
-> transition pressure risk filter
-> portfolio execution
```

Ban duoc freeze:

```text
Policy = vn_legacy_acc_all_else_transition_pressure_nonneg
Execution = min_positions = 6
Cost = 15 bps
Holdout/test = chua dung
```

Ly do chon ban nay:

- LSTM van la thanh phan hoc chinh cua he thong.
- Duong di hai tang LSTM tot hon viec chi them feature vao LSTM goc.
- Ket qua con duong tren song duoc qua 4 artifact validation.
- Ban `min_positions=6` qua duoc risk screen ve drawdown va turnover.

## 2. Du lieu

Du lieu goc:

```text
X_raw = vn_gold_recommended.csv
```

Bien doi du lieu:

```text
X_process = F(X_raw)
```

Trong do:

```text
X_{i,t} in R^{15 x 29}
```

Ky hieu:

| Ky hieu | Y nghia |
| --- | --- |
| `i` | ma co phieu |
| `t` | ngay giao dich |
| `15` | so ngay nhin lai |
| `29` | so feature dau vao cua base LSTM |

Target:

```text
y_{i,t+1} = r_{i,t+1}
```

Trong do `r_{i,t+1}` la return ngay tiep theo cua co phieu `i`.

## 3. Features cua base LSTM

Base LSTM dung 29 feature:

| STT | Feature |
| ---: | --- |
| 1 | `open_level_20` |
| 2 | `high_level_20` |
| 3 | `low_level_20` |
| 4 | `close_level_20` |
| 5 | `volume_level_20` |
| 6 | `open_delta_1` |
| 7 | `high_delta_1` |
| 8 | `low_delta_1` |
| 9 | `close_delta_1` |
| 10 | `volume_delta_1` |
| 11 | `intraday_return` |
| 12 | `gap_open` |
| 13 | `close_position` |
| 14 | `bb_width` |
| 15 | `volume_ratio_20` |
| 16 | `volatility_20` |
| 17 | `momentum_20` |
| 18 | `macd_hist` |
| 19 | `vnindex_return` |
| 20 | `sector_momentum_rank` |
| 21 | `sector_momentum_rank_pct` |
| 22 | `sector_momentum_20` |
| 23 | `relative_sector_momentum_20` |
| 24 | `sector_return` |
| 25 | `alpha_sector` |
| 26 | `sector_positive_ratio` |
| 27 | `sector_ad_ratio` |
| 28 | `a_d_ratio` |
| 29 | `day_of_week` |

Ghi chu portability:

```text
vnindex_return -> market_proxy_return_1
```

Khi dua sang thi truong khac, `vnindex_return` khong nen duoc giu nhu ten logic. No nen duoc map thanh chi so dai dien thi truong:

| Thi truong | Market proxy |
| --- | --- |
| VN | VNINDEX hoac equal-weight market return |
| JP | TOPIX / Nikkei / equal-weight market return |
| KR | KOSPI / KOSDAQ / equal-weight market return |
| US | S&P 500 / Russell / equal-weight market return |

## 4. Base model

Base model duoc freeze:

```text
Run = broad_signmag_portable_no_identity_20260428_allvn_r01
Model = lstm_seed_52
Checkpoint = model_seed_52.keras
Window = 15
Features = 29
Target = target_next_return
Target normalizer = volatility_20
Identity feature = false
```

Cong thuc:

```text
h_{i,t} = LSTM_{64,32}(X_{i,t})
f_{i,t+1} = W h_{i,t} + b
```

Trong do:

| Ky hieu | Y nghia |
| --- | --- |
| `X_{i,t}` | chuoi feature 15 ngay cua co phieu `i` |
| `h_{i,t}` | hidden representation sau LSTM |
| `f_{i,t+1}` | du bao return ngay tiep theo |

Metric chinh:

```text
loss(z) = Q_0.5(|z|) + 0.5 Q_0.9(|z|)
rel_score = 1 - loss(y - f) / loss(y)
```

Y nghia:

- `Q_0.5` la median absolute error.
- `Q_0.9` phat hien loi lon o tail.
- `rel_score > 0` nghia la model tot hon baseline zero prediction theo robust loss.

## 5. LSTM filter signal

Base LSTM tao du bao return. Tang filter cung la LSTM, duoc them sau do de tra loi cau hoi:

```text
Du bao nay co dang tradeable khong?
```

Input cua filter:

```text
Z_{i,t} in R^{10 x 15}
```

Filter model:

```text
p_{i,t} = FilterLSTM(Z_{i,t-9:t})
```

Trong do:

| Ky hieu | Y nghia |
| --- | --- |
| `Z_{i,t}` | feature cua filter tai ngay `t` |
| `p_{i,t}` | xac suat tin hieu cua base LSTM la tradeable |

Filter features trong artifact stress moi:

| STT | Feature |
| ---: | --- |
| 1 | `base_prediction` |
| 2 | `base_abs_prediction` |
| 3 | `base_prediction_rank_pct` |
| 4 | `base_prediction_zscore` |
| 5 | `market_proxy_return_1` |
| 6 | `market_proxy_return_5` |
| 7 | `market_proxy_return_20` |
| 8 | `market_proxy_return_60` |
| 9 | `market_proxy_volatility_20` |
| 10 | `market_proxy_volatility_ratio_20` |
| 11 | `market_breadth_20` |
| 12 | `market_ad_ratio_20` |
| 13 | `market_proxy_drawdown_60` |
| 14 | `market_liquidity_zscore_20` |
| 15 | `ichi_8_22_44_tenkan_kijun_gap` |

Feature bi loai trong stress artifact:

```text
market_leader_return
```

Ly do:

- Kiem tra xem policy co phu thuoc vao market-leader feature khong.
- Ket qua van positive 4 / 4 artifact, nen policy khong chi song nho feature nay.

## 6. Committee selector

Tu base forecast va filter probability:

```text
s_{i,t} = |f_{i,t+1}| p_{i,t}
```

Trong do:

| Ky hieu | Y nghia |
| --- | --- |
| `|f_{i,t+1}|` | do manh cua du bao |
| `p_{i,t}` | xac suat tin hieu tradeable |
| `s_{i,t}` | expected move score de xep hang co phieu |

Tap ung vien giao dich:

```text
C_t = SelectTop(s_{i,t}, constraints)
```

Muc tieu cua tang nay khong thay the LSTM. No dung forecast va filter probability tu hai tang LSTM de chon duoc nhom co phieu co tin hieu tot hon sau cost.

## 7. Wyckoff phase gate

Goi:

```text
phi_t in {accumulation, markup, distribution, markdown, transition}
```

La regime thi truong theo cach doc Wyckoff.

Policy duoc freeze:

```text
g(phi_t) =
  legacy_filter_shortlist,  neu phi_t = accumulation
  all_committee_candidates, neu phi_t in {markup, distribution, markdown}
  transition_rule,          neu phi_t = transition
```

Dang bang:

| Phase | Policy |
| --- | --- |
| accumulation | `legacy_filter_shortlist` |
| markup | `all_committee_candidates` |
| distribution | `all_committee_candidates` |
| markdown | `all_committee_candidates` |
| transition | `all_committee_candidates_if_pressure_delta_20_gte_0` |

## 8. Transition pressure risk filter

Khi thi truong o phase `transition`, model khong trade mac dinh. No chi trade khi ap luc mua khong am.

Dinh nghia:

```text
pressure_delta_20(t)
  = MA20(mean_i(buying_pressure_{i,t} - selling_pressure_{i,t}))
```

Rule:

```text
transition_rule =
  all_committee_candidates, neu pressure_delta_20(t) >= 0
  cash,                     neu pressure_delta_20(t) < 0
```

Y nghia:

- Neu thi truong dang transition nhung buying pressure van khong am, cho phep committee trade.
- Neu transition va pressure am, dung ngoai thi truong de giam risk.

## 9. Portfolio execution

Ban bao cao duoc freeze:

```text
min_positions = 6
cost_bps = 15
```

Trong do:

```text
TO_t = sum_i |w_{i,t} - w_{i,t-1}|
R_t^{net} = sum_i w_{i,t} r_{i,t+1} - 0.0015 TO_t
```

Y nghia:

| Ky hieu | Y nghia |
| --- | --- |
| `w_{i,t}` | ty trong co phieu `i` tai ngay `t` |
| `TO_t` | turnover L1 cua danh muc |
| `R_t^{net}` | return sau chi phi giao dich |

## 10. Validation

Protocol:

```text
source split = validation
train_days = 126
test_days = 21
step_days = 21
strict non-overlap = true
holdout/test = not used
```

Risk screen:

```text
max drawdown >= -25%
avg turnover <= 0.20
```

Ket qua freeze `min_positions=6` qua 4 artifact:

| Metric | Value |
| --- | ---: |
| positive artifacts | 4 / 4 |
| worst-artifact equity | 1.186 |
| mean equity | 1.460 |
| minimum Sharpe | +0.55 |
| mean Sharpe | +0.98 |
| worst max drawdown | -21.7% |
| max average turnover | 0.179 |
| risk screen | pass |

Ket qua tung artifact:

| Artifact | Net equity | Sharpe | Max DD | Avg turnover |
| --- | ---: | ---: | ---: | ---: |
| `r06_selector_module` | 1.724 | +1.27 | -21.7% | 0.173 |
| `r05_signmag` | 1.356 | +0.92 | -19.5% | 0.135 |
| `market_leader_k3w60` | 1.572 | +1.18 | -21.7% | 0.179 |
| `no_leader_seed43` | 1.186 | +0.55 | -21.7% | 0.126 |

## 11. So sanh voi ban manh hon

Ban `min_positions=5` co return tot hon nhung khong qua turnover screen:

| Variant | Worst equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Risk screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `min_positions=5` | 1.240 | 1.679 | +0.59 | -24.3% | 0.201 | fail nhe |
| `min_positions=6` | 1.186 | 1.460 | +0.55 | -21.7% | 0.179 | pass |

Ket luan:

- `min_positions=5` la ban research co return cao hon.
- `min_positions=6` la ban nen dua vao bao cao vi sach hon ve risk-control.

## 12. Ket luan hoc thuat

Ket qua hien tai khong nen duoc dien giai la:

```text
LSTM du bao gia rat tot.
```

Nen dien giai la:

```text
Day la he thong LSTM hai tang.
Base LSTM tao forecast return.
Filter LSTM hoc xem forecast nao co tinh tradeable.
Committee va Wyckoff/pressure gate khong thay the LSTM, ma dieu khien cach dung tin hieu LSTM trong portfolio.
Phan cai thien den tu viec khai thac tin hieu LSTM tot hon va kiem soat rui ro tot hon, khong phai tu viec bo LSTM.
```

Trang thai:

| Muc | Danh gia |
| --- | --- |
| Phu hop bao cao advisor | Co |
| Phu hop noi la production-ready | Chua |
| Da dung holdout/test | Chua |
| Buoc tiep theo | Leakage audit, sau do mo holdout mot lan |

## 13. Cau noi ngan gon de dua vao bao cao

```text
Mo hinh duoc freeze la mot kien truc LSTM hai tang: base LSTM du bao return va filter LSTM danh gia do tradeable cua tin hieu.

Base LSTM tao du bao next-day return. Sau do, filter LSTM danh gia do tradeable cua tung du bao, committee selector chon nhom co phieu co expected move tot hon, va Wyckoff phase gate dieu kien hoa hanh vi giao dich theo regime thi truong.

Trong phase transition, mo hinh chi giao dich khi pressure_delta_20 >= 0. Dieu nay giup tranh giao dich trong giai do thi truong chuyen pha nhung ap luc mua yeu.

Voi cau hinh conservative min_positions=6 va chi phi 15 bps, policy duong tren 4/4 artifact validation, worst equity 1.186, mean equity 1.460, min Sharpe +0.55, worst drawdown -21.7%, max turnover 0.179.

Ket qua nay du de dua vao thao luan hoc thuat ve mot he thong LSTM forecast-to-execution, nhung chua nen xem la mo hinh production vi holdout/test chua duoc mo.
```
