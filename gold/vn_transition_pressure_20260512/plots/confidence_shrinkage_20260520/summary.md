# Confidence Shrinkage & Volatility Abstention Readout

Step 4: post-processing on raw_baseline predictions (no retraining).
Scope: VN train/validation only. Holdout/test is not used.

## Aggregate Validation (3 seeds)

| strategy         |   rel_score_mean |   rel_score_std |   daily_max_mean |   spike_days_ge_8pct_mean |   directional_accuracy_mean |   coverage_mean |
|:-----------------|-----------------:|----------------:|-----------------:|--------------------------:|----------------------------:|----------------:|
| mag_shrink_g2    |        0.0290711 |      0.00586214 |        0.111242  |                  11.6667  |                    0.509405 |        1        |
| mag_shrink_g1    |        0.0289048 |      0.00629179 |        0.111242  |                  11.6667  |                    0.509405 |        1        |
| clip             |        0.0287292 |      0.00568305 |        0.109259  |                  11.6667  |                    0.509405 |        1        |
| baseline         |        0.02837   |      0.00531529 |        0.111242  |                  11.6667  |                    0.509405 |        1        |
| vol_mag_shrink   |        0.0280033 |      0.00699032 |        0.107045  |                  11       |                    0.509405 |        1        |
| clip_vol_shrink  |        0.0274516 |      0.00616447 |        0.104559  |                  11       |                    0.509405 |        1        |
| vol_shrink       |        0.0271998 |      0.00590864 |        0.107045  |                  11       |                    0.509405 |        1        |
| clip_vol_abstain |        0.0147556 |      0.00501761 |        0.0906847 |                   2.33333 |                    0.434025 |        0.844007 |
| vol_abstain      |        0.0146883 |      0.00500799 |        0.0906847 |                   2.33333 |                    0.434025 |        0.844007 |

## Fitted Parameters

{
  "43": {
    "clip_k": 1.5,
    "vol_abstain_threshold": 0.02661484368366409,
    "vol_median": 0.01814861721928876,
    "vol_shrink_power": 0.0,
    "pred_q75": 0.003660288775
  },
  "52": {
    "clip_k": 1.5,
    "vol_abstain_threshold": 0.02661484368366409,
    "vol_median": 0.01814861721928876,
    "vol_shrink_power": 0.25,
    "pred_q75": 0.0043181562
  },
  "71": {
    "clip_k": 2.0,
    "vol_abstain_threshold": 0.02661484368366409,
    "vol_median": 0.01814861721928876,
    "vol_shrink_power": 0.25,
    "pred_q75": 0.003845315275
  }
}

## Decision

Best strategy = highest rel_score_mean that ALSO reduces spike_days_ge_8pct vs baseline.
If no strategy improves both, the raw baseline remains gold.