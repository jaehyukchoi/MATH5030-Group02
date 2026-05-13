import numpy as np
import pytest

from heston_mc.variance_option import variance_option_payoff


def test_variance_option_payoff_vectorized():
    realized_variance = np.array([0.02, 0.04, 0.08])
    strike = 0.04
    
    expected_payoff = np.array([0.00, 0.00, 0.04])
    actual_payoff = variance_option_payoff(realized_variance, strike)
    
    np.testing.assert_array_almost_equal(actual_payoff, expected_payoff)


def test_variance_option_payoff_single_element():
    assert np.isclose(variance_option_payoff(np.array([0.06]), 0.04)[0], 0.02)
    assert np.isclose(variance_option_payoff(np.array([0.02]), 0.04)[0], 0.00)
    assert np.isclose(variance_option_payoff(np.array([0.04]), 0.04)[0], 0.00)


def test_variance_option_payoff_all_otm():
    realized_variance = np.zeros(100)
    strike = 0.05
    
    expected_payoff = np.zeros(100)
    actual_payoff = variance_option_payoff(realized_variance, strike)
    
    np.testing.assert_array_almost_equal(actual_payoff, expected_payoff)


def test_variance_option_payoff_validation():
    with pytest.raises(ValueError, match="nonnegative"):
        variance_option_payoff(np.array([0.05]), -0.01)
        
    
    with pytest.raises(ValueError, match="1D"):
        variance_option_payoff(np.array([[0.05]]), 0.04)
        
    
    with pytest.raises(AttributeError):
        variance_option_payoff(0.05, 0.04)