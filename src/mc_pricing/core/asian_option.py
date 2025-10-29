"""Asian option implementation."""

import numpy as np
from .base_option import BaseOption
from typing import Literal


class AsianOption(BaseOption):
    """Asian option with payoff based on average price over the lifetime.

    Supports both arithmetic and geometric averaging:
    - Arithmetic Call: max(A - K, 0) where A = (1/n) * sum(S_i)
    - Arithmetic Put: max(K - A, 0)
    - Geometric Call: max(G - K, 0) where G = (prod(S_i))^(1/n)
    - Geometric Put: max(K - G, 0)

    Attributes:
        averaging_type: 'arithmetic' or 'geometric'
    """

    def __init__(
        self,
        option_type: str,
        strike: float,
        maturity: float,
        spot: float,
        rate: float,
        volatility: float,
        averaging_type: Literal['arithmetic', 'geometric'] = 'arithmetic',
        dividend: float = 0.0
    ):
        """Initialize Asian option.

        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            maturity: Time to maturity in years
            spot: Current spot price
            rate: Risk-free interest rate
            volatility: Volatility (sigma)
            averaging_type: 'arithmetic' or 'geometric'
            dividend: Continuous dividend yield

        Raises:
            ValueError: If averaging_type is invalid
        """
        super().__init__(option_type, strike, maturity, spot, rate, volatility, dividend)

        if averaging_type not in ['arithmetic', 'geometric']:
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'")

        self.averaging_type = averaging_type

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculate Asian option payoff.

        Args:
            paths: Array of price paths, shape (n_paths, n_steps)

        Returns:
            Array of payoffs, shape (n_paths,)
        """
        if paths.ndim == 1:
            # If only final prices given, can't calculate average
            raise ValueError("Asian options require full price paths, not just final prices")

        # Calculate average price
        if self.averaging_type == 'arithmetic':
            average_price = np.mean(paths, axis=1)
        else:  # geometric
            # Use log-space for numerical stability
            log_prices = np.log(paths)
            log_mean = np.mean(log_prices, axis=1)
            average_price = np.exp(log_mean)

        # Calculate payoff
        if self.option_type == 'call':
            payoff = np.maximum(average_price - self.strike, 0.0)
        else:  # put
            payoff = np.maximum(self.strike - average_price, 0.0)

        return payoff

    def __repr__(self) -> str:
        """String representation of the option."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.option_type}, "
            f"K={self.strike}, "
            f"T={self.maturity}, "
            f"S0={self.spot}, "
            f"r={self.rate}, "
            f"sigma={self.volatility}, "
            f"avg={self.averaging_type})"
        )
