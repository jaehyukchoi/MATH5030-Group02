from __future__ import annotations

import time

import numpy as np

from .interfaces import PricingResult, SimulationResult
from .params import HestonParams, MonteCarloConfig
from .realized_variance import realized_variance_from_prices
from .simulation import HestonModelSimulator
from .utils import discount, standard_error
from .variance_option import variance_option_payoff
from .variance_swap import price_variance_swap, variance_swap_payoff


def expected_average_variance(params: HestonParams, maturity: float) -> float:
    r"""Return E[(1/T) int_0^T v_t dt] under the Heston variance process.

    For
        dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t,
    we have
        E[v_t] = theta + (v0 - theta) exp(-kappa t).

    Therefore,
        E[(1/T) int_0^T v_t dt]
        = theta + (v0 - theta) * (1 - exp(-kappa T)) / (kappa T).

    This analytic quantity is used as the mean of the realized-variance
    control variate. It avoids estimating the control mean from the same
    Monte Carlo sample used for pricing.
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if params.kappa <= 0:
        raise ValueError("params.kappa must be positive")

    return float(
        params.theta
        + (params.v0 - params.theta)
        * (1.0 - np.exp(-params.kappa * maturity))
        / (params.kappa * maturity)
    )


def optimal_control_variate_coefficient(
    target_samples: np.ndarray,
    control_samples: np.ndarray,
) -> float:
    """Estimate beta = Cov(Y, X) / Var(X) from a separate pilot sample.

    This helper is still useful, but it should not be applied to the same
    sample that is used for the final pricing estimator. For the final
    estimator, beta should be fixed in advance or estimated from an
    independent pilot simulation.
    """
    if target_samples.ndim != 1:
        raise ValueError("target_samples must be 1D")
    if control_samples.ndim != 1:
        raise ValueError("control_samples must be 1D")
    if target_samples.shape != control_samples.shape:
        raise ValueError("target_samples and control_samples must have the same shape")
    if target_samples.size < 2:
        return 0.0

    control_var = float(np.var(control_samples, ddof=1))
    if control_var == 0.0:
        return 0.0

    covariance = float(np.cov(target_samples, control_samples, ddof=1)[0, 1])
    return covariance / control_var


def apply_control_variate(
    target_samples: np.ndarray,
    control_samples: np.ndarray,
    control_mean: float,
    beta: float,
) -> np.ndarray:
    """Apply a control variate with a beta fixed outside this final sample."""
    if target_samples.ndim != 1:
        raise ValueError("target_samples must be 1D")
    if control_samples.ndim != 1:
        raise ValueError("control_samples must be 1D")
    if target_samples.shape != control_samples.shape:
        raise ValueError("target_samples and control_samples must have the same shape")

    return target_samples - beta * (control_samples - control_mean)


def standard_error_improvement_ratio(
    plain_std_error: float,
    control_variate_std_error: float,
) -> float:
    if plain_std_error < 0 or control_variate_std_error < 0:
        raise ValueError("standard errors must be nonnegative")
    if control_variate_std_error == 0.0:
        return float("inf")
    return float(plain_std_error / control_variate_std_error)


def _discounted_realized_variance_control(
    sim_result: SimulationResult,
    rate: float,
    maturity: float,
    params: HestonParams,
) -> tuple[np.ndarray, float]:
    realized_variance = realized_variance_from_prices(
        sim_result.stock_paths,
        sim_result.dt,
    )
    discounted_control = discount(realized_variance, rate, maturity)
    discounted_control_mean = float(
        discount(expected_average_variance(params, maturity), rate, maturity)
    )
    return discounted_control, discounted_control_mean


def _simulate_pilot_result(
    params: HestonParams,
    config: MonteCarloConfig,
    pilot_n_paths: int,
    pilot_seed: int,
) -> SimulationResult:
    if pilot_n_paths <= 1:
        raise ValueError("pilot_n_paths must be greater than 1")

    pilot_config = MonteCarloConfig(
        maturity=config.maturity,
        n_steps=config.n_steps,
        n_paths=pilot_n_paths,
        seed=pilot_seed,
        periods_per_year=config.periods_per_year,
    )
    stock_paths, variance_paths = HestonModelSimulator(params, pilot_config).simulate()
    dt = pilot_config.maturity / pilot_config.n_steps
    return SimulationResult(stock_paths=stock_paths, variance_paths=variance_paths, dt=dt)


def estimate_variance_option_beta_from_pilot(
    params: HestonParams,
    config: MonteCarloConfig,
    strike: float,
    rate: float,
    maturity: float,
    pilot_n_paths: int = 2_000,
    pilot_seed: int = 202405,
) -> float:
    """Estimate option CV beta from an independent pilot simulation.

    The final pricing run must use a different random seed/sample. This avoids
    the professor's concern that beta is calculated from the same Monte Carlo
    results used for the reported estimator.
    """
    pilot_result = _simulate_pilot_result(
        params=params,
        config=config,
        pilot_n_paths=pilot_n_paths,
        pilot_seed=pilot_seed,
    )

    realized_variance = realized_variance_from_prices(
        pilot_result.stock_paths,
        pilot_result.dt,
    )
    target_samples = discount(
        variance_option_payoff(realized_variance, strike),
        rate,
        maturity,
    )
    control_samples = discount(realized_variance, rate, maturity)

    return optimal_control_variate_coefficient(target_samples, control_samples)


def price_variance_swap_control_variate(
    sim_result: SimulationResult,
    strike: float,
    rate: float,
    maturity: float,
    params: HestonParams,
) -> PricingResult:
    """Price a variance swap using a fixed analytic beta.

    For a variance swap, the discounted target is
        exp(-rT) * (RV - K),
    and the control is
        exp(-rT) * RV.
    Hence the linear coefficient is exactly beta = 1.
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive")

    start_time = time.time()

    realized_variance = realized_variance_from_prices(
        sim_result.stock_paths,
        sim_result.dt,
    )
    discounted_payoff = discount(
        variance_swap_payoff(realized_variance, strike),
        rate,
        maturity,
    )

    discounted_control, discounted_control_mean = _discounted_realized_variance_control(
        sim_result=sim_result,
        rate=rate,
        maturity=maturity,
        params=params,
    )

    adjusted_samples = apply_control_variate(
        target_samples=discounted_payoff,
        control_samples=discounted_control,
        control_mean=discounted_control_mean,
        beta=1.0,
    )

    return PricingResult(
        price=float(np.mean(adjusted_samples)),
        std_error=standard_error(adjusted_samples),
        n_paths=sim_result.stock_paths.shape[0],
        n_steps=sim_result.stock_paths.shape[1] - 1,
        runtime_seconds=time.time() - start_time,
        method_name="control_variate_variance_swap_beta_1",
    )


def price_variance_option_control_variate(
    sim_result: SimulationResult,
    strike: float,
    rate: float,
    maturity: float,
    params: HestonParams,
    beta: float,
) -> PricingResult:
    """Price a variance option using a pre-set or pilot-estimated beta.

    The function intentionally requires beta as an argument. This design makes
    it clear that beta should be determined before the final pricing sample,
    not re-estimated from the same sample being reported.
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive")

    start_time = time.time()

    realized_variance = realized_variance_from_prices(
        sim_result.stock_paths,
        sim_result.dt,
    )
    discounted_payoff = discount(
        variance_option_payoff(realized_variance, strike),
        rate,
        maturity,
    )

    discounted_control, discounted_control_mean = _discounted_realized_variance_control(
        sim_result=sim_result,
        rate=rate,
        maturity=maturity,
        params=params,
    )

    adjusted_samples = apply_control_variate(
        target_samples=discounted_payoff,
        control_samples=discounted_control,
        control_mean=discounted_control_mean,
        beta=beta,
    )

    return PricingResult(
        price=float(np.mean(adjusted_samples)),
        std_error=standard_error(adjusted_samples),
        n_paths=sim_result.stock_paths.shape[0],
        n_steps=sim_result.stock_paths.shape[1] - 1,
        runtime_seconds=time.time() - start_time,
        method_name="control_variate_variance_option_pilot_beta",
    )


def compare_variance_swap_methods(
    sim_result: SimulationResult,
    strike: float,
    rate: float,
    maturity: float,
    params: HestonParams,
) -> dict[str, float]:
    plain_result = price_variance_swap(
        sim_result=sim_result,
        strike=strike,
        rate=rate,
        maturity=maturity,
    )
    cv_result = price_variance_swap_control_variate(
        sim_result=sim_result,
        strike=strike,
        rate=rate,
        maturity=maturity,
        params=params,
    )

    return {
        "plain_price": plain_result.price,
        "plain_std_error": plain_result.std_error,
        "control_variate_price": cv_result.price,
        "control_variate_std_error": cv_result.std_error,
        "std_error_improvement_ratio": standard_error_improvement_ratio(
            plain_result.std_error,
            cv_result.std_error,
        ),
    }
