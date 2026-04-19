from __future__ import annotations
import time
import numpy as np
from .interfaces import PricingResult, SimulationResult
from .params import HestonParams
from .realized_variance import realized_variance_from_prices
from .utils import discount, standard_error
from .variance_option import variance_option_payoff
from .variance_swap import price_variance_swap, variance_swap_payoff


def expected_average_variance(params: HestonParams, maturity: float) -> float:
    r"""
    Expected time-averaged variance under the Heston variance process.

    For
        dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t,
    we have
        E[v_t] = theta + (v0 - theta) exp(-kappa t).

    Therefore,
        E[(1/T) \int_0^T v_t dt]
        = theta + (v0 - theta) * (1 - exp(-kappa T)) / (kappa T).
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive")

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
) -> tuple[np.ndarray, float]:
    if target_samples.ndim != 1:
        raise ValueError("target_samples must be 1D")
    if control_samples.ndim != 1:
        raise ValueError("control_samples must be 1D")
    if target_samples.shape != control_samples.shape:
        raise ValueError("target_samples and control_samples must have the same shape")

    beta = optimal_control_variate_coefficient(target_samples, control_samples)
    adjusted_samples = target_samples - beta * (control_samples - control_mean)
    return adjusted_samples, float(beta)


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



def price_variance_swap_control_variate(
    sim_result: SimulationResult,
    strike: float,
    rate: float,
    maturity: float,
    params: HestonParams,
) -> PricingResult:
    if maturity <= 0:
        raise ValueError("maturity must be positive")

    start_time = time.time()

    realized_variance = realized_variance_from_prices(
        sim_result.stock_paths,
        sim_result.dt,
    )
    payoff = variance_swap_payoff(realized_variance, strike)
    discounted_payoff = discount(payoff, rate, maturity)

    discounted_control, discounted_control_mean = _discounted_realized_variance_control(
        sim_result=sim_result,
        rate=rate,
        maturity=maturity,
        params=params,
    )

    adjusted_samples, _ = apply_control_variate(
        target_samples=discounted_payoff,
        control_samples=discounted_control,
        control_mean=discounted_control_mean,
    )

    price = float(np.mean(adjusted_samples))
    std_err = standard_error(adjusted_samples)
    runtime = time.time() - start_time

    return PricingResult(
        price=price,
        std_error=std_err,
        n_paths=sim_result.stock_paths.shape[0],
        n_steps=sim_result.stock_paths.shape[1] - 1,
        runtime_seconds=runtime,
        method_name="control_variate_variance_swap",
    )



def price_variance_option_control_variate(
    sim_result: SimulationResult,
    strike: float,
    rate: float,
    maturity: float,
    params: HestonParams,
) -> PricingResult:
    if maturity <= 0:
        raise ValueError("maturity must be positive")

    start_time = time.time()

    realized_variance = realized_variance_from_prices(
        sim_result.stock_paths,
        sim_result.dt,
    )
    payoff = variance_option_payoff(realized_variance, strike)
    discounted_payoff = discount(payoff, rate, maturity)

    discounted_control, discounted_control_mean = _discounted_realized_variance_control(
        sim_result=sim_result,
        rate=rate,
        maturity=maturity,
        params=params,
    )

    adjusted_samples, _ = apply_control_variate(
        target_samples=discounted_payoff,
        control_samples=discounted_control,
        control_mean=discounted_control_mean,
    )

    price = float(np.mean(adjusted_samples))
    std_err = standard_error(adjusted_samples)
    runtime = time.time() - start_time

    return PricingResult(
        price=price,
        std_error=std_err,
        n_paths=sim_result.stock_paths.shape[0],
        n_steps=sim_result.stock_paths.shape[1] - 1,
        runtime_seconds=runtime,
        method_name="control_variate_variance_option",
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
