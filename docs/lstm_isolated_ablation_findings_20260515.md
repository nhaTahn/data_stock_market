# LSTM Isolated Ablation Findings — 2026-05-15

After the 5-seed × 60-epoch final fit (config: L1 + L4 + R6) returned
`mean_ensemble val rel_score = +0.00301` below anchor `+0.00490`, three
isolated A/B tests with `seed=42, 24 epochs` were run to attribute the
regression to a single change.

## Setup

Each test changes exactly one dimension from the canonical signmag config
(`window=15`, `lstm_units=[64,32]`, `dropout=0.05`, `lr=5e-4`,
`loss=rel_score`, `target_normalizer=volatility_20`, default feature set
of 26 columns):

| Test | use_layer_norm | recurrent_dropout | sample_weight_mode |
| --- | :-: | :-: | :-: |
| baseline | False | 0.0 | magnitude |
| L1 only | True | 0.1 | magnitude |
| L4 only | False | 0.0 | none |

## Results

| variant | val rel | val dir_acc | val mae | train rel |
| --- | ---: | ---: | ---: | ---: |
| baseline (no L1, no L4) | **+0.01111** | 48.69% | 0.01913 | +0.00426 |
| L1 only (LN + rec_drop) | +0.00563 | 48.50% | 0.01923 | +0.00248 |
| L4 only (sw=none) | +0.00837 | 48.10% | 0.01936 | −0.00204 |
| reference: L1+L4 (final, seed_42) | +0.00585 | 48.27% | 0.01913 | — |
| reference: L1+L4 mean_ensemble | +0.00301 | 48.37% | 0.01908 | — |

## Findings

### L1 (LayerNorm + recurrent_dropout) — **REJECTED**

- Drop of `49%` on val rel_score (`+0.01111 → +0.00563`).
- Train rel_score also dropped (`+0.00426 → +0.00248`) → it's not an
  overfitting issue; L1 hurt the model's ability to fit train too.
- Likely mechanism: LayerNorm after the final LSTM normalises the hidden
  state going into `magnitude_raw = softplus(Dense(...))`. Softplus-based
  magnitude head depends on the *scale* of incoming features. Normalising
  away that scale loses signal needed for magnitude prediction.
- recurrent_dropout=0.1 on top of dropout=0.05 also over-regularises on
  this universe size (~29 stocks × ~10 years × ~250 days = ~70k sequences).

### L4 (sample_weight_mode = "none") — **REJECTED**

- Drop of `25%` on val rel_score (`+0.01111 → +0.00837`).
- Train rel_score went *negative* (`−0.00204`) — model genuinely failed to
  learn the training distribution well without magnitude weighting.
- Mechanism: `rel_score = 1 − (q50(|err|) + 0.5·q90(|err|)) / loss(base)`.
  The `q90` term explicitly cares about the upper tail of error magnitude.
  Magnitude sample weighting upweights samples with large `|y|`, biasing
  the optimiser toward correctly predicting tail moves — exactly what
  rel_score rewards.
- The previous WF-CV ablation incorrectly favoured `none` because the
  variant gap (`0.0004`) sat well inside the noise floor (std `~0.003`).

### Direction is the structural bottleneck

Across all 4 variants, directional accuracy stayed in `[48.10%, 48.69%]`
— uniformly below 50%. The rel_score signal lives entirely in magnitude
calibration. Any future architectural search should consider:

- A dedicated sign-head ablation: separate the optimisation of
  `sign_prob` from `signed_prediction` and `magnitude` (currently they
  share a backbone via signmag).
- Adding **cross-sectional rank loss** as a sidecar (already implemented
  via `CrossSectionalPairwiseRankLoss` but weight default = 0). Past tests
  in `docs/current_best_path.md` rejected it inside the main signmag, but
  a downstream rank head on a frozen backbone is unexplored.

## Actions

1. **Revert L1 and L4 in `configs/lstm_config.json`**:
   - `use_layer_norm`: not present (defaults to False) ✓
   - `recurrent_dropout`: not present (defaults to 0.0) ✓
   - `sample_weight_mode`: restored to `"magnitude"` ✓
2. **Keep R6 (multi-seed mean ensemble)**: variance reduction is orthogonal
   to L1/L4 choice and still useful.
3. **Keep WF-CV infrastructure (R3a/R3b)** for future searches, but pair
   any variant gap < `2σ` with a bootstrap CI before acting on it.
4. **Re-run the 5-seed × 60-epoch final fit** with the reverted config to
   establish the production `mean_ensemble val rel_score`.

## Re-baseline expectations

With seed_42 alone reaching `+0.01111` on canonical baseline (24 epochs),
the 5-seed × 60-epoch final fit should land somewhere around:

- `mean_ensemble val rel_score ≈ +0.008–0.012` (5-seed averaging on
  high-variance metric typically gives 60–90% of the best single seed).
- `directional_accuracy` will likely remain `~48–49%` regardless of L1/L4
  state — structural bottleneck noted above.

If the re-baseline lands above `+0.005` on `mean_ensemble`, we **can**
open the holdout for a single final readout. If it lands below, hold
and address the directional bottleneck first.
