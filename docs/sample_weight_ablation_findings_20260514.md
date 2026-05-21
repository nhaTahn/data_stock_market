# Sample-Weight Ablation Findings — 2026-05-14

> **REVISION 2026-05-15**: The WF-CV winner `none` was **REVERTED** to
> `magnitude` after a follow-up isolated A/B test on the canonical single
> split showed `none` underperformed by ~25%. See bottom of this file for
> details. Keep the original analysis below for reference, but the operational
> decision changed.


Walk-forward CV, 4 folds × 3 seeds × 24 epochs, on VN training period
(`<= 2020-03-31`). Configured per [configs/lstm_config.json](../configs/lstm_config.json)
with `window=15`, `lstm_units=[64,32]`, `dropout=0.05`,
`recurrent_dropout=0.1`, `use_layer_norm=True`, `loss=rel_score`,
`target_normalizer=volatility_20`. See full output at
`data/processed/assets/data_info_vn/history/training_runs/reports/sample_weight_ablation/sweight_ablation_20260514/`.

## Results

| variant | mean | median | std | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | +0.00223 | +0.00156 | 0.00159 | +0.00123 | +0.00458 |
| `magnitude` | +0.00263 | +0.00440 | 0.00457 | −0.00409 | +0.00580 |
| `magnitude_balance` | +0.00263 | +0.00440 | 0.00457 | −0.00409 | +0.00580 |
| `inv_volatility` | −0.00139 | +0.00117 | 0.00759 | −0.01250 | +0.00461 |

## Decisions

- **Winner: `sample_weight_mode = "none"`** (selected, applied to
  [configs/lstm_config.json](../configs/lstm_config.json)).
- Reasoning: mean gap to `magnitude` (~0.0004) is within bootstrap noise
  (std 0.002–0.005), but `none` has 1/3 the std, never produces a negative
  fold, and removes a hyperparameter family from the search space.

## Findings

- **`inv_volatility` rejected.** Hypothesis "current magnitude weighting
  over-emphasises outlier high-vol days" is **not supported** by data.
  Mean dropped to `−0.00139`, worst fold collapsed to `−0.01250`.
  Mechanism: `rel_score` loss is `q50(|err|) + 0.5·q90(|err|)`. The `q90`
  term explicitly cares about the right tail, exactly where high-vol
  samples live. Downweighting them removes the model's incentive to fit
  the tail well.
- **`magnitude_balance` is a no-op on VN-only data.** Identical numbers to
  `magnitude`. Pipeline path:
  [src/models/training/pipeline.py:1353](../src/models/training/pipeline.py:1353).
  `infer_balance_group_labels` returns a single market label when stocks
  are all VN, so `balance_sample_weights_by_group` becomes the identity.
  Re-run as multi-market group (e.g. VN+US+JP) is the only way to test
  the intended effect.
- **`magnitude` had higher mean but high variance.** This is interesting:
  the previously-canonical setting wins on average but loses on robustness.
  Suspicion: previous single-split val results may have been a lucky draw
  within the `magnitude` distribution. WF-CV breaks that illusion.

## What this changes about the project narrative

Previous research notes
([docs/current_best_path.md](current_best_path.md)) treated
`sample_weight_mode=magnitude` as a settled default. This ablation shows
the choice is not statistically distinguishable from no weighting at all
once we use walk-forward CV. Recommended reframing:

- Old narrative: "magnitude weighting helps the model focus on large
  moves which matters for rel_score." → not falsifiable on single split.
- New narrative: "Sample weighting is regime-dependent. On VN-only data
  with current feature set and rel_score loss, uniform weighting is just
  as good on average and more reliable across regime. Magnitude weighting
  remains an option if a future feature set / multi-market run shows it
  helps."

## Next step

Proceed to L5 — single-split final fit using:

```
window_size: 15
lstm_units: [64, 32]
dropout: 0.05
recurrent_dropout: 0.1
use_layer_norm: True
loss: rel_score
target_normalizer: volatility_20
sample_weight_mode: none           # ← L4 winner
ensemble_strategy: mean             # ← R6
lstm_seeds: [42, 52, 62, 72, 82]    # ← 5 seeds
epochs: 60
patience: 15
batch_size: 64
```

Train on full train period (`<= 2020-03-31`), evaluate on the canonical
val (Apr-2020 → Nov-2022). After agreeing on the result, open the holdout
exactly once.

---

## Revision 2026-05-15: L4 winner reverted

After the 5-seed × 60-epoch final fit produced `mean_ensemble val rel_score
= +0.00301` (below anchor `+0.00490`), we ran 3 isolated A/B tests with
`seed=42, 24 epochs` to disentangle which change hurt:

| Test | use_layer_norm | recurrent_dropout | sample_weight_mode | val rel_score |
| --- | :-: | :-: | :-: | ---: |
| baseline (canonical) | False | 0.0 | magnitude | **+0.01111** |
| L1 only | True | 0.1 | magnitude | +0.00563 |
| L4 only | False | 0.0 | none | +0.00837 |
| (final 5-seed L1+L4, seed_42) | True | 0.1 | none | +0.00585 |

Both L1 and L4 individually **hurt** val rel_score by 49% and 25%
respectively. Combined, the L1 effect dominated.

Why the original WF-CV picked `none`:
- Mean gap (0.0004) sat well inside the bootstrap noise floor (std ~0.003).
- 4 folds × 3 seeds × 24 epochs was insufficient power to discriminate.
- Single-split val with seed_42 × 24 epochs gives a clearer signal — the
  fold-averaged WF-CV result masked variant differences.

Lesson learned:
- When variant gaps are O(0.0004) on rel_score and WF-CV std is O(0.002–0.005),
  the WF-CV result is **not directly actionable** without a paired test or
  larger sample. Report CI overlap, not just the mean.
- For variants this close, prefer the simpler baseline (Occam's razor) and
  treat the WF-CV "winner" as a tie.

Action:
- `configs/lstm_config.json` reverted: `sample_weight_mode = "magnitude"`.
- `inv_volatility` remains rejected (its WF-CV underperformance was outside
  noise, so that finding stands).
- See also `docs/lstm_isolated_ablation_findings_20260515.md` for the L1 +
  L4 isolation analysis.
