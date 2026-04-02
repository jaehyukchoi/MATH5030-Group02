from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HestonParams:
    s0: float
    v0: float
    r: float
    q: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def validate(self) -> None:
        if self.s0 <= 0:
            raise ValueError("s0 must be positive")
        if self.v0 < 0:
            raise ValueError("v0 must be nonnegative")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta < 0:
            raise ValueError("theta must be nonnegative")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be between -1 and 1")


@dataclass
class MonteCarloConfig:
    maturity: float
    n_steps: int
    n_paths: int
    seed: int = 42
    periods_per_year: int = 252

    def validate(self) -> None:
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")


def default_heston_params() -> HestonParams:
    return HestonParams(
        s0=100.0,
        v0=0.04,
        r=0.03,
        q=0.0,
        kappa=2.0,
        theta=0.04,
        sigma=0.50,
        rho=-0.70,
    )


def default_mc_config() -> MonteCarloConfig:
    return MonteCarloConfig(
        maturity=1.0,
        n_steps=252,
        n_paths=10000,
        seed=42,
        periods_per_year=252,
    )
