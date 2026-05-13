import numpy as np
import pytest

from heston_mc.interfaces import PricingResult, SimulationResult
from heston_mc.variance_swap import price_variance_swap


def test_variance_swap_mathematical_accuracy():
    
    # Path 1: [0.1, 0.1] -> RV = (0.01 + 0.01) / 1.0 = 0.02
    # Path 2: [-0.1, 0.1] -> RV = (0.01 + 0.01) / 1.0 = 0.02
    stock_paths = np.array([
        [100.0, 110.0, 121.0],
        [100.0, 90.0, 99.0],
    ])
    variance_paths = np.zeros_like(stock_paths)
    dt = 0.5
    maturity = 1.0
    strike = 0.015
    rate = 0.03  

    sim_result = SimulationResult(stock_paths=stock_paths, variance_paths=variance_paths, dt=dt)

    result = price_variance_swap(
        sim_result=sim_result,
        strike=strike,
        rate=rate,
        maturity=maturity,
    )

   
    expected_price = 0.005 * np.exp(-0.03 * maturity)

    assert isinstance(result, PricingResult)
    assert result.n_paths == 2
    assert result.n_steps == 2
    
    assert np.isclose(result.price, expected_price)
    
    assert np.isclose(result.std_error, 0.0)


def test_variance_swap_zero_volatility_negative_payoff():
    
    stock_paths = np.ones((50, 100)) * 100.0
    variance_paths = np.zeros_like(stock_paths)
    dt = 0.01
    maturity = 0.99
    strike = 0.04
    rate = 0.0

    sim_result = SimulationResult(stock_paths=stock_paths, variance_paths=variance_paths, dt=dt)
    result = price_variance_swap(sim_result, strike, rate, maturity)

    expected_price = -0.04 
    
    assert np.isclose(result.price, expected_price)
    assert np.isclose(result.std_error, 0.0)


def test_variance_swap_validation():
    
    sim_result = SimulationResult(
        stock_paths=np.array([[100.0, 101.0]]), 
        variance_paths=np.array([[0.04, 0.04]]), 
        dt=1.0
    )
    
   
    with pytest.raises(ValueError, match="maturity"):
        price_variance_swap(sim_result, strike=0.01, rate=0.0, maturity=-1.0)
        
    with pytest.raises(ValueError, match="maturity"):
        price_variance_swap(sim_result, strike=0.01, rate=0.0, maturity=0.0)