"""Tests for Monte Carlo simulator."""

import pytest
import numpy as np
from mc_pricing.core import EuropeanOption, BlackScholes
from mc_pricing.monte_carlo import MonteCarloSimulator


class TestMonteCarloSimulator:
    """Test suite for Monte Carlo simulator."""

    def test_initialization(self):
        """Test simulator initialization."""
        sim = MonteCarloSimulator(n_simulations=10000, seed=42)

        assert sim.n_simulations == 10000
        assert sim.n_steps == 252
        assert sim.path_generator is not None

    def test_european_call_pricing(self, european_call):
        """Test pricing of European call option."""
        sim = MonteCarloSimulator(n_simulations=50000, seed=42)
        price, std_error = sim.price(european_call)

        # Get Black-Scholes price for comparison
        bs_price = BlackScholes.price(
            option_type='call',
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2
        )

        # MC price should be close to BS price
        assert abs(price - bs_price) < 3 * std_error
        # Sanity check
        assert price > 0
        assert std_error > 0

    def test_european_put_pricing(self, european_put):
        """Test pricing of European put option."""
        sim = MonteCarloSimulator(n_simulations=50000, seed=42)
        price, std_error = sim.price(european_put)

        # Get Black-Scholes price for comparison
        bs_price = BlackScholes.price(
            option_type='put',
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2
        )

        # MC price should be close to BS price
        assert abs(price - bs_price) < 3 * std_error

    def test_reproducibility(self, european_call):
        """Test that same seed gives same results."""
        sim1 = MonteCarloSimulator(n_simulations=10000, seed=42)
        sim2 = MonteCarloSimulator(n_simulations=10000, seed=42)

        price1, _ = sim1.price(european_call)
        price2, _ = sim2.price(european_call)

        np.testing.assert_almost_equal(price1, price2)

    def test_different_seeds_give_different_results(self, european_call):
        """Test that different seeds give different results."""
        sim1 = MonteCarloSimulator(n_simulations=10000, seed=42)
        sim2 = MonteCarloSimulator(n_simulations=10000, seed=123)

        price1, _ = sim1.price(european_call)
        price2, _ = sim2.price(european_call)

        # Prices should be different (but close)
        assert price1 != price2
        # But within reasonable range
        assert abs(price1 - price2) < 1.0

    def test_confidence_interval(self, european_call):
        """Test confidence interval calculation."""
        sim = MonteCarloSimulator(n_simulations=10000, seed=42)
        price, lower, upper = sim.price_with_confidence_interval(
            european_call, confidence=0.95
        )

        assert lower < price < upper
        assert upper - lower > 0

    def test_asian_option_pricing(self, asian_call):
        """Test pricing of Asian option."""
        sim = MonteCarloSimulator(n_simulations=10000, seed=42)
        price, std_error = sim.price(asian_call)

        # Asian call should be cheaper than European call (less volatile)
        european_call = EuropeanOption(
            option_type='call',
            strike=100.0,
            maturity=1.0,
            spot=100.0,
            rate=0.05,
            volatility=0.2
        )
        european_price, _ = sim.price(european_call)

        # This should generally hold
        assert price <= european_price + 2 * std_error

    def test_convergence_test(self, european_call):
        """Test convergence analysis."""
        sim = MonteCarloSimulator(n_simulations=10000, seed=42)
        n_trials = np.array([1000, 5000, 10000, 50000])

        prices, errors = sim.convergence_test(european_call, n_trials)

        assert len(prices) == len(n_trials)
        assert len(errors) == len(n_trials)

        # Errors should generally decrease with more trials
        assert errors[-1] < errors[0]
