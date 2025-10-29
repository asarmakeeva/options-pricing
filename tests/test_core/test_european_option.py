"""Tests for European option pricing."""

import pytest
import numpy as np
from mc_pricing.core import EuropeanOption, BlackScholes


class TestEuropeanOption:
    """Test suite for European options."""

    def test_initialization(self):
        """Test option initialization."""
        option = EuropeanOption(
            option_type='call',
            strike=100.0,
            maturity=1.0,
            spot=100.0,
            rate=0.05,
            volatility=0.2
        )

        assert option.option_type == 'call'
        assert option.strike == 100.0
        assert option.maturity == 1.0
        assert option.spot == 100.0
        assert option.rate == 0.05
        assert option.volatility == 0.2

    def test_invalid_option_type(self):
        """Test that invalid option type raises error."""
        with pytest.raises(ValueError):
            EuropeanOption(
                option_type='invalid',
                strike=100.0,
                maturity=1.0,
                spot=100.0,
                rate=0.05,
                volatility=0.2
            )

    def test_call_payoff(self, european_call):
        """Test call option payoff calculation."""
        terminal_prices = np.array([90, 100, 110, 120])
        payoffs = european_call.payoff(terminal_prices)

        expected = np.array([0, 0, 10, 20])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_put_payoff(self, european_put):
        """Test put option payoff calculation."""
        terminal_prices = np.array([80, 90, 100, 110])
        payoffs = european_put.payoff(terminal_prices)

        expected = np.array([20, 10, 0, 0])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_payoff_with_paths(self, european_call):
        """Test payoff with full price paths."""
        # Create paths with shape (n_paths, n_steps)
        paths = np.array([
            [100, 105, 110],
            [100, 95, 90],
            [100, 102, 115]
        ])

        payoffs = european_call.payoff(paths)
        expected = np.array([10, 0, 15])  # Only terminal prices matter

        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_atm_option(self):
        """Test at-the-money option."""
        option = EuropeanOption(
            option_type='call',
            strike=100.0,
            maturity=1.0,
            spot=100.0,
            rate=0.05,
            volatility=0.2
        )

        terminal = np.array([100.0])
        payoff = option.payoff(terminal)

        assert payoff[0] == 0.0

    def test_deep_itm_call(self):
        """Test deep in-the-money call."""
        option = EuropeanOption(
            option_type='call',
            strike=50.0,
            maturity=1.0,
            spot=100.0,
            rate=0.05,
            volatility=0.2
        )

        terminal = np.array([100.0])
        payoff = option.payoff(terminal)

        assert payoff[0] == 50.0

    def test_deep_otm_put(self):
        """Test deep out-of-the-money put."""
        option = EuropeanOption(
            option_type='put',
            strike=50.0,
            maturity=1.0,
            spot=100.0,
            rate=0.05,
            volatility=0.2
        )

        terminal = np.array([100.0])
        payoff = option.payoff(terminal)

        assert payoff[0] == 0.0
