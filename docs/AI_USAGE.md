# AI Usage and Code Review Notes

This project used AI assistance mainly for code organization, documentation drafts, and initial implementation suggestions. The team did not treat the AI-generated code as final without review. We checked the mathematical meaning of the pricing functions, the realized variance convention, and the control variate construction.

## How AI was prompted

The prompts asked AI to help organize a Python package for Monte Carlo pricing of variance derivatives under the Heston model. The requested structure included Heston simulation, realized variance calculation, variance swap pricing, variance option pricing, control variates, and experiments. Follow-up prompts asked AI to make the implementation modular and consistent with shared dataclasses such as `HestonParams`, `MonteCarloConfig`, `SimulationResult`, and `PricingResult`.

After feedback on the GitHub issue, AI was specifically asked to review the control variate implementation and the calculation of beta. The main correction was to avoid estimating beta from the same Monte Carlo sample used for the final reported estimator.

## Reuse of existing code and libraries

The project reuses standard numerical tools from NumPy rather than reimplementing array operations, random number generation, covariance estimation, discounting, or standard error calculations from scratch. The implementation also uses Python dataclasses to keep model parameters and result containers clear.

The project did not rely on a black-box pricing library for the main Monte Carlo pricing workflow. The Heston simulation, realized variance calculation, variance swap pricing, variance option pricing, and control variate logic are implemented directly so that the numerical method is transparent.

## What was changed after review

The original control variate code estimated beta as `Cov(target, control) / Var(control)` using the same Monte Carlo sample used for pricing. This is not ideal because the final estimator then depends on a random beta fitted on the reported sample.

The revised implementation makes beta more explicit:

1. For the variance swap, beta is fixed at 1. This follows from the linear payoff `RV - K` when the control variable is realized variance.
2. For the variance option, beta is estimated from an independent pilot simulation and then held fixed for the final pricing simulation.
3. The control mean is based on the analytic expected average Heston variance rather than the sample mean from the final pricing paths.

These changes make the control variate implementation more consistent with financial Monte Carlo practice.

## Added beyond the reference paper

The project focuses on a reproducible Python implementation and compares plain Monte Carlo with control-variate Monte Carlo for both variance swaps and variance options. It also includes modular interfaces, reusable pricing functions, and experiment code that reports price, standard error, runtime, and variance reduction.

The project uses the analytic expectation of average variance under the Heston variance process as a practical control mean. This connects the numerical implementation to analytic results while keeping the final estimator simple and easy to audit.

## Reference added after feedback

Bernard, C., & Cui, Z. (2014). Prices and asymptotics for discrete variance swaps. *Applied Mathematical Finance*, 21(2), 140–173. https://doi.org/10.1080/1350486X.2013.820524
