import numpy as np
import pytest

from heston_mc.realized_variance import realized_variance_from_prices

def test_realized_variance_basic():
    stock_paths = np.array([
        [100.0, 110.0, 121.0],
        [100.0, 90.0, 99.0],
    ])
    dt = 0.5
    rv = realized_variance_from_prices(stock_paths, dt)

    
    expected = np.array([
        (0.1**2 + 0.1**2) / 1.0,
        ((-0.1)**2 + 0.1**2) / 1.0,
    ])

    assert rv.shape == (2,)
    np.testing.assert_array_almost_equal(rv, expected)

def test_realized_variance_constant_price():
    paths = np.ones((50, 100)) * 100.0
    rv = realized_variance_from_prices(paths, 0.01)
    np.testing.assert_array_almost_equal(rv, np.zeros(50))

def test_realized_variance_1d_input_rejected():
    path = np.array([100.0, 110.0, 121.0])
    with pytest.raises(ValueError, match="2D"):
        realized_variance_from_prices(path, 0.5)

@pytest.mark.parametrize("bad_dt", [0.0, -0.5])
def test_realized_variance_invalid_dt(bad_dt):
    paths = np.array([[100.0, 101.0], [100.0, 102.0]])
    with pytest.raises(ValueError):
        realized_variance_from_prices(paths, bad_dt)

def test_realized_variance_insufficient_steps():
    paths = np.array([[100.0], [101.0]])
    with pytest.raises(ValueError, match="time points"):
        realized_variance_from_prices(paths, 0.5)

def test_realized_variance_zero_prices_warning():
    paths = np.array([[100.0, 0.0, 100.0]])
    with pytest.warns(RuntimeWarning):
        realized_variance_from_prices(paths, 0.5)