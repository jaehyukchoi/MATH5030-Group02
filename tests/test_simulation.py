import numpy as np
import pytest

from heston_mc.params import default_heston_params, default_mc_config
from heston_mc.simulation import HestonModelSimulator


def test_simulation_output_shape_and_validity():
    params = default_heston_params()
    config = default_mc_config()
    config.n_paths = 100
    config.n_steps = 50
    
    S, v = HestonModelSimulator(params, config).simulate()
    
    assert S.shape == (100, 51)
    assert v.shape == (100, 51)
    assert not np.any(np.isnan(S))
    assert not np.any(np.isnan(v))
    assert np.all(S >= 0.0)


def test_simulation_initial_conditions():
    params = default_heston_params()
    config = default_mc_config()
    S, v = HestonModelSimulator(params, config).simulate()
    
    
    np.testing.assert_array_equal(S[:, 0], params.s0)
    np.testing.assert_array_equal(v[:, 0], params.v0)


def test_simulation_reproducibility():
    params = default_heston_params()
    config1 = default_mc_config()
    config1.seed = 2026
    
    config2 = default_mc_config()
    config2.seed = 2026
    
    config3 = default_mc_config()
    config3.seed = 9999
    
    S1, v1 = HestonModelSimulator(params, config1).simulate()
    S2, v2 = HestonModelSimulator(params, config2).simulate()
    S3, v3 = HestonModelSimulator(params, config3).simulate()
    
    
    np.testing.assert_array_equal(S1, S2)
    np.testing.assert_array_equal(v1, v2)
    

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(S1, S3)



def test_simulation_zero_volatility_limit():
    params = default_heston_params()
    params.v0 = 0.0
    params.theta = 0.0
    params.sigma = 1e-8
    
    config = default_mc_config()
    config.n_paths = 10
    config.maturity = 1.0
    
    dt = config.maturity / config.n_steps
    
    S, v = HestonModelSimulator(params, config).simulate()
    
    
    np.testing.assert_array_almost_equal(v, np.zeros_like(v), decimal=5)
    
    
    expected_S_T = params.s0 * ((1 + params.r * dt) ** config.n_steps)
    np.testing.assert_array_almost_equal(S[:, -1], np.full(config.n_paths, expected_S_T), decimal=5)


def test_simulation_martingale_property():
    params = default_heston_params()
    config = default_mc_config()
    
    config.n_paths = 50000 
    config.maturity = 1.0
    
    S, _ = HestonModelSimulator(params, config).simulate()
    
    simulated_expected_S_T = np.mean(S[:, -1])
    theoretical_expected_S_T = params.s0 * np.exp(params.r * config.maturity)
    
   
    assert np.isclose(simulated_expected_S_T, theoretical_expected_S_T, rtol=0.005)