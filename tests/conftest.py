"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from mc_pricing.core import EuropeanOption, AsianOption
from mc_pricing.monte_carlo import MonteCarloSimulator


@pytest.fixture
def european_call():
    """Create a standard European call option."""
    return EuropeanOption(
        option_type='call',
        strike=100.0,
        maturity=1.0,
        spot=100.0,
        rate=0.05,
        volatility=0.2
    )


@pytest.fixture
def european_put():
    """Create a standard European put option."""
    return EuropeanOption(
        option_type='put',
        strike=100.0,
        maturity=1.0,
        spot=100.0,
        rate=0.05,
        volatility=0.2
    )


@pytest.fixture
def asian_call():
    """Create a standard Asian call option."""
    return AsianOption(
        option_type='call',
        strike=100.0,
        maturity=1.0,
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        averaging_type='arithmetic'
    )


@pytest.fixture
def mc_simulator():
    """Create a Monte Carlo simulator with fixed seed."""
    return MonteCarloSimulator(n_simulations=10000, seed=42)


@pytest.fixture
def sample_paths():
    """Generate sample price paths for testing."""
    np.random.seed(42)
    n_paths = 1000
    n_steps = 252
    spot = 100.0

    # Generate random paths
    returns = np.random.normal(0, 0.01, (n_paths, n_steps))
    log_paths = np.cumsum(returns, axis=1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(log_paths)

    return paths
