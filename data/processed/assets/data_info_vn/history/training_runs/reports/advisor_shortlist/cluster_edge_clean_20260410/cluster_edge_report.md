# Cluster-Level Edge Report

## Executive View

- Winner chính hiện tại là `F&B committee` trên cụm `KDC,SAB,SBT,VNM`.
- Candidate phụ còn đáng giữ là `VN30 BDS expert` trên cụm `VHM,BCM,VIC,VRE`.
- `Broad VN30 forecasting` vẫn chưa đứng vững: walk-forward yếu và panel LSTM bước 1 vẫn âm nhẹ trên test.

## 1. Primary Edge: F&B Committee

- standalone `test rel_score`: `0.034253`
- committee `test rel_score`: `0.051016`
- committee `val rel_score`: `0.023671`
- stable-band median: `0.050897`
- rule: `method=avg`, `weight_expert=0.90`
- overlap codes: `KDC,SAB,SBT,VNM`

## 2. Secondary Edge: VN30 BDS Expert

- run: `vn30gold_expert_bds_20260410_105821`
- standalone `val rel_score`: `0.013337`
- standalone `test rel_score`: `0.039268`
- backtest final equity: `2.285065`
- backtest directional accuracy: `0.590713`
- bias `pred_abs_over_actual_abs`: `0.155596`

## 3. Broad VN30 Is Still Weak

- VN30 panel baseline `test rel_score`: `-0.008646`
- best broad VN30 paper-style run `best_test rel_score`: `+0.001336`
- broad VN30 representation improved, but the edge is still too small to claim a stable full-universe forecasting model.

## 4. Reading Note

- Có edge sạch nhất ở `cluster-level`, không phải `broad VN30`.
- `F&B committee` hiện là candidate đủ đẹp nhất để mang đi trao đổi về hướng nghiên cứu.
- `VN30 BDS expert` là candidate phụ để chứng minh vẫn có edge trong chính universe VN30, nhưng chưa đủ để claim forecasting cho toàn bộ VN30.
