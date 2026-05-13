import numpy as np
import pytest

from heston_mc.utils import simple_returns_from_prices, standard_error


def test_simple_returns_calculation():
    prices = np.array([
        [100.0, 110.0, 121.0],
        [100.0, 90.0,  99.0],
    ])
    expected_returns = np.array([
        [0.10, 0.10],
        [-0.10, 0.10],
    ])
    
    rets = simple_returns_from_prices(prices)
    np.testing.assert_array_almost_equal(rets, expected_returns)


def test_simple_returns_validation_dimensions():
    with pytest.raises(ValueError, match="2D"):
        simple_returns_from_prices(np.array([100.0, 101.0]))
        
    with pytest.raises(ValueError, match="2D"):
        simple_returns_from_prices(np.array([[[100.0, 101.0]]]))


def test_simple_returns_validation_time_points():
    with pytest.raises(ValueError, match="time points"):
        simple_returns_from_prices(np.array([[100.0], [101.0]]))


def test_simple_returns_zero_price_warning():
    prices = np.array([[100.0, 0.0, 50.0]])
    with pytest.warns(RuntimeWarning):
        simple_returns_from_prices(prices)


def test_standard_error_calculation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    se = standard_error(x)
    expected_se = np.std(x, ddof=1) / np.sqrt(len(x))
    
    assert np.isclose(se, expected_se)


def test_standard_error_zero_variance():
    x = np.array([5.0, 5.0, 5.0, 5.0])
    se = standard_error(x)
    assert se == 0.0


def test_standard_error_single_element():
    x = np.array([1.0])
    se = standard_error(x)
    assert np.isnan(se)
    
def test_standard_error_empty_array():
    x = np.array([])
    se = standard_error(x)
    assert np.isnan(se)