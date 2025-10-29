"""Tests for Black-Scholes pricing."""

import pytest
import numpy as np
from mc_pricing.core import BlackScholes


class TestBlackScholes:
    """Test suite for Black-Scholes pricing."""

    def test_atm_call_price(self):
        """Test ATM call option price."""
        price = BlackScholes.price(
            option_type='call',
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2
        )

        # ATM call should be positive
        assert price > 0
        # Rough sanity check (BS price should be around 10.45)
        assert 9.0 < price < 12.0

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)."""
        spot = 100.0
        strike = 100.0
        maturity = 1.0
        rate = 0.05
        volatility = 0.2

        call_price = BlackScholes.price(
            'call', spot, strike, maturity, rate, volatility
        )
        put_price = BlackScholes.price(
            'put', spot, strike, maturity, rate, volatility
        )

        parity = call_price - put_price
        expected = spot - strike * np.exp(-rate * maturity)

        np.testing.assert_almost_equal(parity, expected, decimal=10)

    def test_delta_call(self):
        """Test call delta is between 0 and 1."""
        greeks = BlackScholes.greeks(
            option_type='call',
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2
        )

        delta = greeks['delta']
        assert 0 < delta < 1
        # ATM call delta should be around 0.5-0.6
        assert 0.4 < delta < 0.7

    def test_delta_put(self):
        """Test put delta is between -1 and 0."""
        greeks = BlackScholes.greeks(
            option_type='put',
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.2
        )

        delta = greeks['delta']
        assert -1 < delta < 0
        # ATM put delta should be around -0.5 to -0.4
        assert -0.6 < delta < -0.3

    def test_gamma_positive(self):
        """Test gamma is positive for both calls and puts."""
        call_greeks = BlackScholes.greeks(
            'call', 100.0, 100.0, 1.0, 0.05, 0.2
        )
        put_greeks = BlackScholes.greeks(
            'put', 100.0, 100.0, 1.0, 0.05, 0.2
        )

        assert call_greeks['gamma'] > 0
        assert put_greeks['gamma'] > 0
        # Gamma should be same for calls and puts
        np.testing.assert_almost_equal(
            call_greeks['gamma'], put_greeks['gamma'], decimal=10
        )

    def test_vega_positive(self):
        """Test vega is positive for both calls and puts."""
        call_greeks = BlackScholes.greeks(
            'call', 100.0, 100.0, 1.0, 0.05, 0.2
        )
        put_greeks = BlackScholes.greeks(
            'put', 100.0, 100.0, 1.0, 0.05, 0.2
        )

        assert call_greeks['vega'] > 0
        assert put_greeks['vega'] > 0

    def test_theta_call(self):
        """Test call theta (time decay)."""
        greeks = BlackScholes.greeks(
            'call', 100.0, 100.0, 1.0, 0.05, 0.2
        )

        # ATM call typically has negative theta
        assert greeks['theta'] < 0

    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        true_vol = 0.25
        price = BlackScholes.price(
            'call', 100.0, 100.0, 1.0, 0.05, true_vol
        )

        implied_vol = BlackScholes.implied_volatility(
            option_type='call',
            market_price=price,
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05
        )

        np.testing.assert_almost_equal(implied_vol, true_vol, decimal=6)

    def test_otm_call_price_increases_with_volatility(self):
        """Test that OTM call price increases with volatility."""
        price_low_vol = BlackScholes.price(
            'call', 100.0, 110.0, 1.0, 0.05, 0.1
        )
        price_high_vol = BlackScholes.price(
            'call', 100.0, 110.0, 1.0, 0.05, 0.3
        )

        assert price_high_vol > price_low_vol

    def test_price_decreases_with_time(self):
        """Test that ATM option price decreases as time to maturity decreases."""
        price_long = BlackScholes.price(
            'call', 100.0, 100.0, 1.0, 0.05, 0.2
        )
        price_short = BlackScholes.price(
            'call', 100.0, 100.0, 0.25, 0.05, 0.2
        )

        assert price_long > price_short
