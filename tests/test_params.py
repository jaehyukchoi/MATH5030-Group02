import pytest
from heston_mc.params import HestonParams, MonteCarloConfig

def test_heston_params_valid():
    params = HestonParams(s0=100.0, v0=0.04, r=0.03, q=0.0, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7)
    params.validate()

@pytest.mark.parametrize("bad_val", [{"rho": -1.5}, {"rho": 1.2}, {"v0": -0.01}, {"theta": -0.05}, {"sigma": -0.1}, {"s0": -50.0},{"kappa": -2.0}])
def test_heston_params_invalid(bad_val):
    base = {"s0": 100.0, "v0": 0.04, "r": 0.03, "q": 0.0, "kappa": 2.0, "theta": 0.04, "sigma": 0.5, "rho": -0.7}
    base.update(bad_val)
    with pytest.raises(ValueError):
        HestonParams(**base).validate()

def test_mc_config_valid():
    cfg = MonteCarloConfig(maturity=1.0, n_steps=252, n_paths=10000, seed=42, periods_per_year=252)
    cfg.validate()

@pytest.mark.parametrize("bad_val", [{"maturity": -1.0}, {"maturity": 0.0},{"n_steps": 0}, {"n_paths": -100}, {"n_paths": 0},{"periods_per_year": 0}])
def test_mc_config_invalid(bad_val):
    base = {"maturity": 1.0, "n_steps": 252, "n_paths": 10000, "seed": 42, "periods_per_year": 252}
    base.update(bad_val)
    with pytest.raises(ValueError):
        MonteCarloConfig(**base).validate()