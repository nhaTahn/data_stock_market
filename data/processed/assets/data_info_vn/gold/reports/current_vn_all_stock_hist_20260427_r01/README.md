# Current VN All-Stock Histogram Report

Scope: train and in-sample validation only. No out-sample/test data is used.

Universe: all stock rows available in the current VN anchor/router prediction frame.

Conventions:

- `E = prediction - actual`
- `q2` and `q8` are the 20% and 80% quantiles of `E` over all stocks and days in the split
- relative_score histogram uses a stabilized local proxy; aggregate `rel_score` uses the repo q50/q90 loss formula

## Quick Read

| Candidate | In-sample rel_score | Direction | Error q2/q8 | Proxy > 0 | Rows | Stocks |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `anchor` | +0.0053 | 48.9% | -0.0160 / +0.0133 | 91.7% | 18279 | 28 |
| `train_rank_regime_ic_weight` | +0.0037 | 48.4% | -0.0163 / +0.0130 | 91.8% | 18279 | 28 |
| `sector19_down_up_anchor_else` | +0.0034 | 48.7% | -0.0163 / +0.0131 | 91.8% | 18279 | 28 |

## Weakest In-Sample Stocks For Anchor

| Code | rel_score | Direction | Error q2/q8 | Error band | Bias mean | Rows |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `SAB` | -0.0340 | 47.6% | -0.0133 / +0.0103 | 0.0235 | -0.0018 | 656 |
| `STB` | -0.0263 | 48.1% | -0.0206 / +0.0178 | 0.0384 | -0.0012 | 657 |
| `ACB` | -0.0201 | 50.3% | -0.0143 / +0.0111 | 0.0254 | -0.0012 | 648 |
| `PLX` | -0.0166 | 46.6% | -0.0141 / +0.0137 | 0.0279 | +0.0004 | 657 |
| `VPB` | -0.0156 | 51.1% | -0.0176 / +0.0130 | 0.0306 | -0.0012 | 657 |
| `CTG` | -0.0155 | 50.1% | -0.0187 / +0.0137 | 0.0324 | -0.0010 | 657 |
| `VIC` | -0.0124 | 45.8% | -0.0107 / +0.0102 | 0.0210 | -0.0005 | 657 |
| `TPB` | -0.0114 | 50.1% | -0.0159 / +0.0150 | 0.0309 | +0.0004 | 657 |

## Overlay

![overlay](plots/candidate_overlay_val_relscore_error_band.png)

## Candidate Histograms

### `anchor`

![anchor](plots/anchor__all_stock_hist.png)

### `sector19_down_up_anchor_else`

![sector19_down_up_anchor_else](plots/sector19_down_up_anchor_else__all_stock_hist.png)

### `train_rank_regime_ic_weight`

![train_rank_regime_ic_weight](plots/train_rank_regime_ic_weight__all_stock_hist.png)
