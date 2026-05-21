# LSTM Error-Control Readout 2026-05-20

Muc tieu hien tai: giu LSTM la forecast model chinh, nhung giam cac ngay spike cua
`q90(|actual_return - predicted_return|)`. Muc tieu `<3.5%` khong nen hieu la full
coverage all-stock/all-day, vi error-floor diagnostic cho thay ngay ca zero/oracle-style
floor van co nhieu ngay lon hon 3.5%. Cach hop ly hon la selective prediction:

```text
return_hat = LSTM(X)
risk_hat   = g(X, return_hat, market_state)
accept     = risk_hat <= tau
```

`tau` phai duoc chon tren train/calibration, sau do moi bao cao tren validation.

## Ket qua chinh

| Huong thu | Ket qua | Ket luan |
|:--|:--|:--|
| `stressaux_w20` full coverage | multiseed rel_score khoang `0.0248`, daily p90 khoang `6.39%`, spike >=8% khoang `7.7` ngay | Van la LSTM forecast base tot nhat hien tai |
| Post-hoc `risk_logistic` / target 3.0, coverage_q40 | obs coverage `16.54%`, rel_score `0.00496`, daily p90 `3.35%`, days >=8% `0` | Dat muc <3.5% khi chap nhan selective coverage |
| Post-hoc `risk_input_noise` / coverage_q30 | obs coverage `13.29%`, rel_score `0.00429`, daily p90 `3.40%`, days >=8% `0` | Xac nhan gia thuyet input-noise co lien quan den spike error |
| Joint RiskAux LSTM | full coverage rel_score am, selective co giam spike nhung return head bi keo xau | Khong promote |
| Detached RiskAux LSTM seed 52 | full coverage rel_score `0.0035`; selective q20 daily p90 `3.81%` | Tot hon joint mot chut nhung kem post-hoc risk model |
| Input z-score clipping seed 52 | `clip5` tang rel_score trong probe seed 52, nhung daily max/spike >=8% xau hon | Clipping don gian khong phai loi giai |

## Cau tra loi ngan cho huong nghien cuu

1. `feature selection co noise?` Co. Cac score dua tren `input_noise_score` tach duoc nhom
   error thap tot hon random va loai duoc spike >=8%. Dieu nay ung ho viec them tang
   error-control dua tren chat luong input, thay vi chi them feature vao LSTM.
2. `chua process tot?` Dung mot phan. Multimarket/rolling normalization va clipping don
   gian chua tao breakthrough, nhung cac chi bao noise cua input co gia tri lam risk gate.
   Buoc dung hon la hoc/calibrate risk tren residual cua LSTM, khong ep LSTM sua tat ca
   samples full coverage.
3. `model chua toi uu?` Co, nhung gioi han lon nam o predictability cua daily return.
   Them joint head lam model xau hon. Base nen giu la `stressaux_w20`, sau do them
   two-stage confidence/error-control.

## Huong nen promote

Promote tam thoi:

```text
Layer 1: LSTM return forecaster = stressaux_w20
Layer 2: Error-control model = logistic/HGB/input-noise risk score
Calibration: choose tau on train/calibration to target daily q90(|E|) p90
Report: rel_score full coverage + coverage/error frontier
```

Day co tinh hoc thuat hon viec noi "model fail spike": bai toan duoc chuyen thanh
heteroskedastic/selective forecasting. LSTM van la model du bao, con confidence layer
la co che quyet dinh luc nao du lieu dau vao/market state dang qua noisy de tin vao
forecast.

## Buoc tiep theo

Chay standard multi-seed cho `stressaux_w20 + post-hoc error-control`, gom:

- full coverage rel_score cua LSTM;
- accepted coverage;
- accepted daily `q90(|E|)` median/p90/max;
- spike days `>=3.5%`, `>=5%`, `>=8%`;
- yearly plots cho accepted vs full coverage.

Neu can dat `<3.5%`, report nen ghi ro la muc nay dat tren selective accepted samples,
khong phai all-stock/all-day full coverage.
