# US Context Hetero LSTM Smoke Readout

Protocol: US portable + context adapter features, train `<= 2020-03-31`, validation `2020-04-01..2022-11-15`. Holdout/test not used.

## Result

| Variant | Seeds | raw rel_score mean | alpha_rel_score mean | DA mean | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| portable hetero_combined | 43,52,62 | 0.00229 | -0.00510 | 51.38% | better raw rel_score |
| context hetero_combined | 43,52,62 | 0.00080 | -0.00247 | 51.75% | better alpha, worse raw |

## Interpretation

- Context adapter is not ready to replace the portable/raw model for US raw rel_score.
- It does improve date-demeaned alpha_rel_score, matching the Ridge adapter probe direction.
- Next experiment should not simply append context features to the raw-return loss. Better directions:
  1. add an explicit alpha/demeaned auxiliary loss,
  2. train a two-head raw-return + alpha-return model,
  3. use context features in a selection/ranking head rather than directly in raw-return prediction.
