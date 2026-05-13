import numpy as np
import pytest

from heston_mc.control_variate import (
    apply_control_variate,
    optimal_control_variate_coefficient
)

def test_optimal_control_variate_coefficient_positive():
    target = np.array([10.0, 20.0, 30.0])
    control = np.array([2.0, 4.0, 6.0])
    beta = optimal_control_variate_coefficient(target, control)
    assert np.isclose(beta, 5.0)

def test_optimal_control_variate_coefficient_negative():
    target = np.array([10.0, 20.0, 30.0])
    control = np.array([6.0, 4.0, 2.0])
    beta = optimal_control_variate_coefficient(target, control)
    assert np.isclose(beta, -5.0)

def test_optimal_control_variate_coefficient_zero_variance():
    target = np.array([10.0, 20.0, 30.0])
    control = np.array([4.0, 4.0, 4.0])
    beta = optimal_control_variate_coefficient(target, control)
    assert beta == 0.0

def test_optimal_control_variate_coefficient_small_sample():
    target = np.array([10.0])
    control = np.array([2.0])
    beta = optimal_control_variate_coefficient(target, control)
    assert beta == 0.0

def test_optimal_control_variate_coefficient_validation():
    with pytest.raises(ValueError, match="1D"):
        optimal_control_variate_coefficient(np.array([[10.0]]), np.array([2.0]))
        
    with pytest.raises(ValueError, match="1D"):
        optimal_control_variate_coefficient(np.array([10.0]), np.array([[2.0]]))
        
    with pytest.raises(ValueError, match="same shape"):
        optimal_control_variate_coefficient(np.array([10.0, 20.0]), np.array([2.0]))

def test_apply_control_variate_basic():
    target = np.array([1.0, 2.0, 3.0])
    control = np.array([2.0, 4.0, 6.0])
    control_mean = 4.0
    beta = 0.5
    expected = np.array([2.0, 2.0, 2.0])
    adjusted = apply_control_variate(target, control, control_mean, beta)
    np.testing.assert_array_almost_equal(adjusted, expected)

def test_apply_control_variate_zero_beta():
    target = np.array([1.0, 2.0, 3.0])
    control = np.array([2.0, 4.0, 6.0])
    adjusted = apply_control_variate(target, control, 4.0, 0.0)
    np.testing.assert_array_equal(adjusted, target)

def test_apply_control_variate_validation():
    with pytest.raises(ValueError, match="1D"):
        apply_control_variate(np.array([[1.0]]), np.array([2.0]), 4.0, 0.5)
        
    with pytest.raises(ValueError, match="same shape"):
        apply_control_variate(np.array([1.0, 2.0]), np.array([2.0]), 4.0, 0.5)