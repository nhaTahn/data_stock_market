# VN30 Paper Phase 1 - 2026-04-12

## Goal

Test whether a paper-inspired representation and causal denoise features can improve VN30 `rel_score` over the existing panel baseline.

Baseline:

- Run: `vn30_panel_probe_step1_listrefresh_20260411`
- Model: `panel_lstm_best_by_val`
- Test `rel_score`: `-0.008646`

## What Changed

Three new VN30 runs were trained:

1. `vn30_paper_repr_20260412_paper_phase1`
   - Paper-style level + delta channels
   - No sequence normalization
2. `vn30_paper_repr_instnorm_20260412_paper_phase1`
   - Same representation
   - Per-window instance z-score normalization
3. `vn30_paper_denoise_20260412_paper_phase1`
   - Same representation + causal denoise features
   - Instance normalization

## Results

| Case | Best by val | Test rel_score | Best by test | Test rel_score |
|---|---:|---:|---|---:|
| baseline_existing_panel | panel_lstm_best_by_val | -0.008646 | panel_lstm_best_by_val | -0.008646 |
| vn30_paper_repr_20260412_paper_phase1 | lstm_signmag_top2_by_val | -0.005029 | lstm_signmag_seed_52 | +0.001336 |
| vn30_paper_repr_instnorm_20260412_paper_phase1 | lstm_signmag_seed_52 | -0.007352 | lstm_signmag | +0.000425 |
| vn30_paper_denoise_20260412_paper_phase1 | lstm_ensemble | -0.004318 | lstm_signmag_seed_52 | -0.001761 |

## Interpretation

- The paper-style representation is worth keeping.
- The strongest result in this phase came from `paper_repr` without denoise: best-test `rel_score = +0.001336`.
- Instance normalization improved validation ranking but did not improve the best test result versus plain `paper_repr`.
- The causal denoise bundle improved validation metrics but did not produce a positive best-test `rel_score`.

## Practical Takeaway

For the next phase on VN30:

1. Keep the paper representation as a candidate base.
2. Keep instance normalization as an optional toggle, not a default winner.
3. Drop full denoise bundle from the main branch for now.
4. Focus next on signal quality and cross-sectional ranking, not heavier smoothing.
