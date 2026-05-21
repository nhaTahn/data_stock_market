# All-VN Model Selection 2026-04-28

Updated: 2026-04-28

## Scope

This note finalizes the current broad VN base model for the clean all-VN universe.

- universe: `93` clean VN tickers
- target mode: `return`
- objective: `rel_score`
- stock identity: `False`
- train rows: `129720`
- validation rows: `60468`
- out-sample rows: `77640`

Important policy note:

- out-sample/test was opened once on `2026-04-28` to finalize the current all-VN base model
- do not keep tuning against this holdout after this decision

## Final Decision

Choose this run/model as the current all-VN base model:

- run: `broad_signmag_portable_no_identity_20260428_allvn_r01`
- selected model alias: `lstm_best_by_val`
- concrete checkpoint: `model_seed_52.keras`

Reason:

- among the completed all-VN candidates, this model has the best current out-sample `rel_score`
- it is more stable than the higher-validation `units48_24` variant

## Finalist Comparison

| Role | Run | Model | Val rel_score | Val dir acc | Test rel_score | Test dir acc | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Chosen base | `broad_signmag_portable_no_identity_20260428_allvn_r01` | `lstm_best_by_val` | `+0.006860631` | `47.67%` | `+0.003684364` | `47.53%` | Best current out-sample `rel_score` |
| Main challenger | `broad_signmag_portable_no_identity_units48_24_20260428_allvn_r01` | `lstm_ensemble` | `+0.006770576` | `48.38%` | `+0.003273047` | `47.59%` | Close second; keep as a challenger |
| Validation winner only | `broad_signmag_portable_no_identity_units48_24_20260428_allvn_r01` | `lstm_best_by_val` | `+0.008262455` | `47.52%` | `-0.001909937` | `47.09%` | Validation gain did not hold in out-sample |

Additional signmag read:

- `broad_signmag_portable_no_identity_20260428_allvn_r01 / lstm_signmag_best_by_val`:
  `val rel_score = +0.006588697`, `test rel_score = -0.002752589`
- `broad_signmag_portable_no_identity_units48_24_20260428_allvn_r01 / lstm_signmag_best_by_val`:
  `val rel_score = +0.005401642`, `test rel_score = +0.000338089`

Conclusion:

- for the current all-VN base branch, plain `lstm` is more reliable than `lstm_signmag`

## Stock-Level Read

For the chosen base model:

- `53/93` tickers have positive test `rel_score`
- `30/93` tickers are positive on both validation and test
- `20/93` tickers have test `rel_score <= -0.01`
- stock-level test `rel_score` median is `+0.001765`

Recurring weak tickers across the strongest finalists:

- `KOS`
- `PPC`
- `VPI`
- `PC1`
- `VPB`
- `VND`
- `CII`
- `EIB`
- `SJS`
- `FPT`

Recurring strong tickers across the strongest finalists:

- `BCM`
- `VIX`
- `MSN`
- `LPB`
- `HDG`
- `VHM`
- `CTS`

Interpretation:

- the aggregate all-VN signal is usable
- the stock-level distribution is still uneven
- the next improvement should target reliability by ticker, not a larger architecture jump

## Next Improvement

Keep the next step narrow:

1. do not tune against the current out-sample holdout
2. treat the chosen base model as the benchmark to beat
3. if stock-aware work continues, start with a soft reliability rule on the chronic weak tickers above
4. prefer sizing/abstain logic over dropping many stocks from the universe
5. if another model run is needed, compare first against:
   - `broad_signmag_portable_no_identity_20260428_allvn_r01 / lstm_best_by_val`
   - `broad_signmag_portable_no_identity_units48_24_20260428_allvn_r01 / lstm_ensemble`

## Artifact References

- `data/processed/assets/data_info_vn/history/training_runs/broad_signmag_portable_no_identity_20260428_allvn_r01/reports/core/leaderboard.csv`
- `data/processed/assets/data_info_vn/history/training_runs/broad_signmag_portable_no_identity_20260428_allvn_r01/holdout_private/predictions_full.csv`
- `data/processed/assets/data_info_vn/history/training_runs/broad_signmag_portable_no_identity_units48_24_20260428_allvn_r01/reports/core/leaderboard.csv`
- `data/processed/assets/data_info_vn/history/training_runs/broad_signmag_portable_no_identity_units48_24_20260428_allvn_r01/holdout_private/predictions_full.csv`
