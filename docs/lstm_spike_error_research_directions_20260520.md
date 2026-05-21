# LSTM spike-error research directions

Problem in current VN work:

```text
LSTM point forecast has acceptable rel_score, but daily q90(|E|) can spike.
E = actual_return - predicted_return
```

The literature usually does not solve this by forcing one point forecast to fit all
days. The recurring solutions are:

1. Predict a conditional distribution or quantiles.
2. Predict volatility/tail risk explicitly.
3. Add calibrated uncertainty or selective prediction.
4. Add volatility/econometric structure such as GARCH into the recurrent model.
5. Decompose noisy/non-stationary series before feeding the LSTM.

## Sources

| Direction | Source | Main idea | Fit to current repo |
|:--|:--|:--|:--|
| Probabilistic RNN | DeepAR, Salinas et al. https://arxiv.org/abs/1704.04110 | Train recurrent model to output a predictive distribution over related time series. | Replace single return head with distribution parameters or quantile heads. |
| Financial LSTM quantile/tail dynamics | LSTM-HTQF, https://arxiv.org/abs/2010.08263 | LSTM learns conditional quantile/tail dynamics of financial returns and VaR. | Strong fit: target the tail/spike behavior directly, not only mean return. |
| VaR/ES LSTM | LSTM-AL, https://arxiv.org/abs/2001.08374 | Bayesian LSTM with asymmetric Laplace likelihood for joint VaR/ES. | Fit for a risk head: estimate loss quantile and expected shortfall. |
| Selective prediction | SelectiveNet, https://proceedings.mlr.press/v97/geifman19a.html | Learn prediction and reject/accept option together to optimize risk-coverage. | Directly matches current selective error-control framing. |
| Conformal time-series forecasting | NeurIPS 2021, https://papers.nips.cc/paper/2021/file/312f1ba2a72318edaaa995a67835fad5-Paper.pdf | Wrap RNN forecasts with calibrated intervals and coverage guarantees. | Fit for post-hoc uncertainty intervals around LSTM forecasts. |
| Adaptive conformal inference | Gibbs & Candes, https://papers.nips.cc/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf | Online conformal calibration under distribution shift. | Fit for markets because error regime changes over time. |
| Ensemble conformalized quantile regression | EnCQR, https://arxiv.org/abs/2202.08756 | Prediction intervals for heteroscedastic/nonstationary series, can sit on deep models. | Good next experiment: quantile LSTM + conformal correction. |
| GARCH-LSTM volatility | Kim & Won, https://www.sciencedirect.com/science/article/pii/S0957417418301416 | Feed GARCH/EGARCH/EWMA volatility information into LSTM; best hybrid reduces volatility forecast errors. | Strong fit: add GARCH-style volatility state as feature/head. |
| GARCH-NN / GARCH-LSTM | AAAI 2024, https://ojs.aaai.org/index.php/AAAI/article/view/29643 | Embed GARCH stylized facts into neural architecture. | Medium-high fit: more architectural, but academically strong. |
| LSTM mixture density | Temporal MDN, https://www.sciencedirect.com/science/article/pii/S0957417425012072 | LSTM-MDN models return uncertainty and volatility/stability regimes. | Fit if moving from point return to mixture distribution. |
| Deep quantile risk + heterogeneous market | Financial Innovation 2024, https://link.springer.com/article/10.1186/s40854-023-00564-5 | Quantile deep learning plus heterogeneous time horizons for VaR/ES. | Fit: multi-horizon features/heads can address regime reaction lag. |
| Decomposition + LSTM | CEEMDAN-LSTM, https://www.sciencedirect.com/science/article/pii/S095741742201017X | Decompose noisy realized volatility series, then forecast components with LSTM. | Fit for input processing, but risk of complexity/overfit. |

## Recommended next experiments

Priority 1: Quantile LSTM head

```text
Return head: y_hat
Quantile heads: q10_hat, q50_hat, q90_hat or abs_error_q90_hat
Loss = rel_score_loss + lambda_q * pinball_loss + monotonic_penalty
```

Use the quantile width or predicted q90 absolute error as confidence. This is closer
to LSTM-HTQF and DeepAR thinking than the current point-only forecast.

Priority 2: Conformal calibration on residuals

```text
calibration residual: e_t = |y_t - y_hat_t|
score_t = e_t / predicted_scale_t
interval_t = y_hat_t +/- conformal_quantile(score) * predicted_scale_t
accept if interval_width <= threshold
```

This is the most defensible way to say the model is calibrated, not just filtered.

Priority 3: GARCH/volatility sidecar

Add daily/stock volatility states:

```text
garch_sigma_1
ewma_sigma_20
egarch_like_leverage
market_sigma_proxy
```

Use them as LSTM inputs and/or to normalize target:

```text
z_return = return / sigma_hat
return_hat = z_hat * sigma_hat
```

Priority 4: Regime/scale adaptation

Train the LSTM to output:

```text
mu_hat      = expected return
sigma_hat   = expected abs error / volatility
tail_hat    = P(|E| > threshold)
```

Then evaluate both:

```text
full coverage rel_score
accepted coverage
accepted daily q90(|E|)
max daily q90(|E|)
```

## Best framing for report

The current work should be framed as:

```text
Stage 1: LSTM return forecaster
Stage 2: calibrated uncertainty / risk-control layer
Goal: reduce high-error days through selective forecasting, not force full-coverage point prediction.
```

The next academically stronger version is:

```text
LSTM distributional forecaster + conformal residual calibration
```

This directly targets the observed issue: the model is slow to adapt when the market
enters a high-volatility or distribution-shift regime.
