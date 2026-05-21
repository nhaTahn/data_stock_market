# High-Error Segment Diagnosis

Scope: `VN100`, split `train`, year `2017`, trading-day segment `200-250`.
Date range: `2017-10-23` to `2017-12-29`. Holdout/test is not used.

## What Happened

- Segment days: `50`.
- Spike rule: `q90(|actual_return - predicted_return|) > 3.5%`.
- Spike days: `20` / `50`.
- Segment median q90(|E|): `3.24%`.
- Spike-day median q90(|E|): `4.16%`.
- Spike dates: 2017-11-01, 2017-11-02, 2017-11-03, 2017-11-09, 2017-11-10, 2017-11-14, 2017-11-15, 2017-11-20, 2017-11-21, 2017-11-23, ....

## Main Diagnostic

- Median q90 actual absolute return in segment: `3.25%`.
- Median q90 predicted absolute return in segment: `0.25%`.
- Median predicted/actual q90 tail ratio: `0.079`.
- Median sign-mismatch share on spike days: `52.69%`.

The high q90 error is therefore not just one bad stock. On spike days, the cross-section has many stocks moving several percent while the base LSTM prediction remains much closer to zero. This is a shrinkage/tail-response problem in an uptrend regime with strong rotation.

## Repeated High-Error Codes

| code   | feature_sector       |   top_error_appearances | mean_abs_error   | max_abs_error   | sign_mismatch_share   |
|:-------|:---------------------|------------------------:|:-----------------|:----------------|:----------------------|
| CMG    | Công nghệ Thông tin  |                      10 | 5.68%            | 7.10%           | 60.00%                |
| DIG    | Bất động sản         |                       8 | 4.66%            | 6.68%           | 25.00%                |
| STB    | Ngân hàng            |                       8 | 4.55%            | 5.26%           | 62.50%                |
| PVD    | Dầu khí              |                       8 | 4.18%            | 6.69%           | 50.00%                |
| VIX    | Dịch vụ tài chính    |                       7 | 6.18%            | 10.07%          | 28.57%                |
| SAB    | Thực phẩm và đồ uống |                       6 | 5.29%            | 7.18%           | 83.33%                |
| PLX    | Dầu khí              |                       6 | 4.93%            | 6.71%           | 50.00%                |
| HCM    | Dịch vụ tài chính    |                       6 | 4.15%            | 5.46%           | 33.33%                |
| SHB    | Ngân hàng            |                       5 | 5.71%            | 6.87%           | 60.00%                |
| VNM    | Thực phẩm và đồ uống |                       5 | 5.34%            | 6.92%           | 20.00%                |
| HAG    | Thực phẩm và đồ uống |                       5 | 5.26%            | 6.65%           | 40.00%                |
| CTG    | Ngân hàng            |                       5 | 5.24%            | 6.96%           | 60.00%                |

## Sector Concentration

| sector                        |   top_error_rows |   unique_codes | mean_abs_error   | max_abs_error   | sign_mismatch_share   | mean_actual_return   |
|:------------------------------|-----------------:|---------------:|:-----------------|:----------------|:----------------------|:---------------------|
| Thực phẩm và đồ uống          |               38 |             10 | 5.09%            | 7.18%           | 57.89%                | 2.38%                |
| Bất động sản                  |               37 |             10 | 4.80%            | 6.99%           | 37.84%                | 2.85%                |
| Ngân hàng                     |               35 |             10 | 5.02%            | 7.94%           | 60.00%                | 2.74%                |
| Dịch vụ tài chính             |               26 |              7 | 4.94%            | 10.07%          | 42.31%                | 1.85%                |
| Xây dựng và Vật liệu          |               19 |              7 | 5.26%            | 8.33%           | 57.89%                | 2.75%                |
| Dầu khí                       |               14 |              2 | 4.50%            | 6.71%           | 50.00%                | 0.95%                |
| Điện, nước & xăng dầu khí đốt |               13 |              5 | 4.71%            | 6.83%           | 30.77%                | 0.90%                |
| Công nghệ Thông tin           |               12 |              2 | 5.60%            | 7.10%           | 66.67%                | 2.24%                |
| Hàng & Dịch vụ Công nghiệp    |               11 |              4 | 4.65%            | 6.66%           | 36.36%                | 2.97%                |
| Tài nguyên Cơ bản             |                9 |              4 | 5.07%            | 6.64%           | 66.67%                | 2.81%                |

## Hypotheses To Test Next

1. **Target/objective shrinkage**: `rel_score` makes the LSTM learn a conservative conditional mean. It helps robust next-day error but under-reacts when the market enters broad uptrend rotation or single-name tail moves.
2. **Feature processing gap**: current features may not encode cross-sectional dispersion, limit-like moves, leader breadth, and market/sector rotation strongly enough. The model sees the uptrend but does not know which stocks are becoming high-move candidates.
3. **Feature selection noise**: portable features are useful, but a few VN-specific microstructure signals may be needed as filter features, not necessarily as base LSTM features.
4. **Model optimization is secondary for now**: simply making the LSTM larger can reduce training error but may not fix tail timing. A small tail-risk/filter head is the cleaner next test.

## Next Ablation

Keep the base LSTM frozen, then train/evaluate a sidecar filter on in-sample/validation with new regime features:

- daily cross-sectional return dispersion
- q90 absolute market/universe return
- market breadth and advance/decline
- sector dispersion and top-sector concentration
- leader/liquidity-weighted return from top traded-value stocks
- limit-like move count or high absolute return count

Success criterion: reduce `q90(|E|)` and high-error trade exposure in this 2017 uptrend segment and similar high-dispersion buckets, while not materially hurting full validation `rel_score`.

## Files

- `segment_daily.csv`
- `segment_daily_distribution.csv`
- `segment_spike_days.csv`
- `segment_spike_top_errors.csv`
- `segment_code_error_summary.csv`
- `segment_sector_error_summary.csv`