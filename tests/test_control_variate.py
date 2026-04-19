import numpy as np

from heston_mc.control_variate import (
    apply_control_variate,
    expected_average_variance,
    price_variance_option_control_variate,
    price_variance_swap_control_variate,
    standard_error_improvement_ratio,
)
from heston_mc.interfaces import PricingResult, SimulationResult
from heston_mc.params import HestonParams
from heston_mc.realized_variance import realized_variance_from_prices
from heston_mc.utils import standard_error
from heston_mc.variance_option import price_variance_option
from heston_mc.variance_swap import price_variance_swap



def test_expected_average_variance_matches_theta_when_v0_equals_theta():
    params = HestonParams(
        s0=100.0,
        v0=0.04,
        r=0.03,
        q=0.0,
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
    )

    assert np.isclose(expected_average_variance(params, maturity=1.0), 0.04)



def test_apply_control_variate_reduces_standard_error_for_correlated_samples():
    rng = np.random.default_rng(123)
    control = rng.normal(size=4000)
    noise = 0.1 * rng.normal(size=4000)
    target = 2.0 + 1.5 * control + noise

    adjusted, beta = apply_control_variate(
        target_samples=target,
        control_samples=control,
        control_mean=0.0,
    )

    assert beta > 0.0
    assert standard_error(adjusted) < standard_error(target)



def test_control_variate_variance_swap_runs_and_improves_se():
    stock_paths = np.array([
        [100.0, 108.0, 97.2, 106.92],
        [100.0, 92.0, 99.36, 90.4176],
        [100.0, 105.0, 110.25, 104.7375],
        [100.0, 95.0, 90.25, 94.7625],
    ])
    variance_paths = np.zeros_like(stock_paths)
    sim_result = SimulationResult(stock_paths=stock_paths, variance_paths=variance_paths, dt=1.0 / 3.0)

    params = HestonParams(
        s0=100.0,
        v0=0.04,
        r=0.0,
        q=0.0,
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
    )

    plain = price_variance_swap(sim_result=sim_result, strike=0.03, rate=0.0, maturity=1.0)
    cv = price_variance_swap_control_variate(
        sim_result=sim_result,
        strike=0.03,
        rate=0.0,
        maturity=1.0,
        params=params,
    )

    assert isinstance(cv, PricingResult)
    assert cv.n_paths == plain.n_paths
    assert cv.n_steps == plain.n_steps
    assert cv.std_error <= plain.std_error + 1e-12



def test_control_variate_variance_option_runs():
    stock_paths = np.array([
        [100.0, 108.0, 97.2, 106.92],
        [100.0, 92.0, 99.36, 90.4176],
        [100.0, 105.0, 110.25, 104.7375],
        [100.0, 95.0, 90.25, 94.7625],
        [100.0, 115.0, 97.75, 112.4125],
        [100.0, 85.0, 97.75, 83.0875],
    ])
    variance_paths = np.zeros_like(stock_paths)
    sim_result = SimulationResult(stock_paths=stock_paths, variance_paths=variance_paths, dt=1.0 / 3.0)

    params = HestonParams(
        s0=100.0,
        v0=0.04,
        r=0.0,
        q=0.0,
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
    )

    rv = realized_variance_from_prices(stock_paths, 1.0 / 3.0)
    strike = float(np.quantile(rv, 0.4))

    plain = price_variance_option(sim_result=sim_result, strike=strike, rate=0.0, maturity=1.0)
    cv = price_variance_option_control_variate(
        sim_result=sim_result,
        strike=strike,
        rate=0.0,
        maturity=1.0,
        params=params,
    )

    assert isinstance(cv, PricingResult)
    assert cv.n_paths == plain.n_paths
    assert cv.n_steps == plain.n_steps
    assert cv.std_error >= 0.0



def test_standard_error_improvement_ratio_is_at_least_one_when_cv_helps():
    assert standard_error_improvement_ratio(0.20, 0.10) == 2.0
