import numpy as np
import pytest

from heston_mc.control_variate import (
    apply_control_variate,
    optimal_control_variate_coefficient
)

def test_optimal_control_variate_coefficient():
    
    target = np.array([10.0, 20.0, 30.0])
    control = np.array([2.0, 4.0, 6.0])
    
    
    beta = optimal_control_variate_coefficient(target, control)
    assert np.isclose(beta, 5.0)

def test_apply_control_variate():
    
    target = np.array([1.0, 2.0, 3.0])
    control = np.array([2.0, 4.0, 6.0])
    control_mean = 4.0
    
    
    beta = 0.5
    
    
    expected = np.array([2.0, 2.0, 2.0])
    
    adjusted = apply_control_variate(target, control, control_mean, beta)
    
    np.testing.assert_array_almost_equal(adjusted, expected)