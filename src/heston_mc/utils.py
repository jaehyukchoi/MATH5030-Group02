from __future__ import annotations

import numpy as np

from .params import HestonParams


def make_correlated_normals(
    n_paths: int,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two correlated standard normal vectors.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be between -1 and 1")

    z1 = rng.standard_normal(n_paths)
    z2 = rng.standard_normal(n_paths)
    z_v = z2
    z_s = rho * z2 + np.sqrt(1.0 - rho**2) * z1
    return z_s, z_v


def discounted_call_payoff(
    terminal_spots: np.ndarray,
    strike: float,
    rate: float,
    maturity: float,
) -> np.ndarray:
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be nonnegative")

    payoffs = np.maximum(terminal_spots - strike, 0.0)
    return np.exp(-rate * maturity) * payoffs


def standard_error(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 0.0
    return float(samples.std(ddof=1) / np.sqrt(samples.size))


def default_heston_params() -> HestonParams:
    """
    Baseline parameter set used across the project.
    """
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
