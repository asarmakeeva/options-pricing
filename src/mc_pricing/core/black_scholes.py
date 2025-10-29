"""Black-Scholes closed-form solutions for European options."""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional


class BlackScholes:
    """Black-Scholes model for European option pricing.

    Provides closed-form solutions for European options and Greeks.
    Used as a benchmark for Monte Carlo pricing.
    """

    @staticmethod
    def d1(spot: float, strike: float, maturity: float, rate: float,
           volatility: float, dividend: float = 0.0) -> float:
        """Calculate d1 parameter in Black-Scholes formula.

        Args:
            spot: Current spot price
            strike: Strike price
            maturity: Time to maturity
            rate: Risk-free rate
            volatility: Volatility
            dividend: Dividend yield

        Returns:
            d1 value
        """
        numerator = np.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity
        denominator = volatility * np.sqrt(maturity)
        return numerator / denominator

    @staticmethod
    def d2(spot: float, strike: float, maturity: float, rate: float,
           volatility: float, dividend: float = 0.0) -> float:
        """Calculate d2 parameter in Black-Scholes formula.

        Args:
            spot: Current spot price
            strike: Strike price
            maturity: Time to maturity
            rate: Risk-free rate
            volatility: Volatility
            dividend: Dividend yield

        Returns:
            d2 value
        """
        d1_val = BlackScholes.d1(spot, strike, maturity, rate, volatility, dividend)
        return d1_val - volatility * np.sqrt(maturity)

    @staticmethod
    def price(
        option_type: str,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        volatility: float,
        dividend: float = 0.0
    ) -> float:
        """Calculate European option price using Black-Scholes formula.

        Args:
            option_type: 'call' or 'put'
            spot: Current spot price
            strike: Strike price
            maturity: Time to maturity
            rate: Risk-free rate
            volatility: Volatility
            dividend: Dividend yield

        Returns:
            Option price

        Raises:
            ValueError: If option_type is invalid
        """
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

        d1_val = BlackScholes.d1(spot, strike, maturity, rate, volatility, dividend)
        d2_val = BlackScholes.d2(spot, strike, maturity, rate, volatility, dividend)

        if option_type == 'call':
            price = (spot * np.exp(-dividend * maturity) * norm.cdf(d1_val) -
                    strike * np.exp(-rate * maturity) * norm.cdf(d2_val))
        else:  # put
            price = (strike * np.exp(-rate * maturity) * norm.cdf(-d2_val) -
                    spot * np.exp(-dividend * maturity) * norm.cdf(-d1_val))

        return price

    @staticmethod
    def greeks(
        option_type: str,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        volatility: float,
        dividend: float = 0.0
    ) -> Dict[str, float]:
        """Calculate all Greeks using closed-form formulas.

        Args:
            option_type: 'call' or 'put'
            spot: Current spot price
            strike: Strike price
            maturity: Time to maturity
            rate: Risk-free rate
            volatility: Volatility
            dividend: Dividend yield

        Returns:
            Dictionary with Delta, Gamma, Vega, Theta, and Rho
        """
        d1_val = BlackScholes.d1(spot, strike, maturity, rate, volatility, dividend)
        d2_val = BlackScholes.d2(spot, strike, maturity, rate, volatility, dividend)

        sqrt_T = np.sqrt(maturity)
        exp_div = np.exp(-dividend * maturity)
        exp_rate = np.exp(-rate * maturity)

        # Delta
        if option_type == 'call':
            delta = exp_div * norm.cdf(d1_val)
        else:
            delta = -exp_div * norm.cdf(-d1_val)

        # Gamma (same for calls and puts)
        gamma = (exp_div * norm.pdf(d1_val)) / (spot * volatility * sqrt_T)

        # Vega (same for calls and puts, convert to per 1% change)
        vega = spot * exp_div * norm.pdf(d1_val) * sqrt_T / 100

        # Theta
        first_term = -(spot * norm.pdf(d1_val) * volatility * exp_div) / (2 * sqrt_T)

        if option_type == 'call':
            second_term = -rate * strike * exp_rate * norm.cdf(d2_val)
            third_term = dividend * spot * exp_div * norm.cdf(d1_val)
            theta = (first_term + second_term + third_term) / 365  # Per day
        else:
            second_term = rate * strike * exp_rate * norm.cdf(-d2_val)
            third_term = -dividend * spot * exp_div * norm.cdf(-d1_val)
            theta = (first_term + second_term + third_term) / 365  # Per day

        # Rho (per 1% change)
        if option_type == 'call':
            rho = strike * maturity * exp_rate * norm.cdf(d2_val) / 100
        else:
            rho = -strike * maturity * exp_rate * norm.cdf(-d2_val) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

    @staticmethod
    def implied_volatility(
        option_type: str,
        market_price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend: float = 0.0,
        initial_guess: float = 0.2,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> Optional[float]:
        """Calculate implied volatility using Newton-Raphson method.

        Args:
            option_type: 'call' or 'put'
            market_price: Observed market price
            spot: Current spot price
            strike: Strike price
            maturity: Time to maturity
            rate: Risk-free rate
            dividend: Dividend yield
            initial_guess: Initial volatility guess
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations

        Returns:
            Implied volatility or None if not converged
        """
        vol = initial_guess

        for _ in range(max_iterations):
            # Calculate price and vega
            price = BlackScholes.price(option_type, spot, strike, maturity, rate, vol, dividend)

            # Vega for Newton-Raphson (actual vega, not per 1%)
            d1_val = BlackScholes.d1(spot, strike, maturity, rate, vol, dividend)
            vega = spot * np.exp(-dividend * maturity) * norm.pdf(d1_val) * np.sqrt(maturity)

            # Check convergence
            diff = market_price - price
            if abs(diff) < tolerance:
                return vol

            # Newton-Raphson update
            if vega < 1e-10:  # Avoid division by very small numbers
                return None

            vol = vol + diff / vega

            # Keep volatility positive
            if vol <= 0:
                return None

        return None  # Failed to converge
