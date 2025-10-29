"""Base class for all option types."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseOption(ABC):
    """Abstract base class for option contracts.

    All option types should inherit from this class and implement
    the payoff method.

    Attributes:
        option_type: 'call' or 'put'
        strike: Strike price
        maturity: Time to maturity in years
        spot: Current spot price
        rate: Risk-free interest rate
        volatility: Volatility (sigma)
        dividend: Continuous dividend yield (default: 0.0)
    """

    def __init__(
        self,
        option_type: str,
        strike: float,
        maturity: float,
        spot: float,
        rate: float,
        volatility: float,
        dividend: float = 0.0
    ):
        """Initialize option parameters.

        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            maturity: Time to maturity in years
            spot: Current spot price
            rate: Risk-free interest rate
            volatility: Volatility (sigma)
            dividend: Continuous dividend yield

        Raises:
            ValueError: If parameters are invalid
        """
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        if strike <= 0:
            raise ValueError("strike must be positive")
        if maturity <= 0:
            raise ValueError("maturity must be positive")
        if spot <= 0:
            raise ValueError("spot must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

        self.option_type = option_type
        self.strike = strike
        self.maturity = maturity
        self.spot = spot
        self.rate = rate
        self.volatility = volatility
        self.dividend = dividend

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculate option payoff for given price paths.

        Args:
            paths: Array of price paths, shape (n_paths, n_steps)

        Returns:
            Array of payoffs, shape (n_paths,)
        """
        pass

    def __repr__(self) -> str:
        """String representation of the option."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.option_type}, "
            f"K={self.strike}, "
            f"T={self.maturity}, "
            f"S0={self.spot}, "
            f"r={self.rate}, "
            f"sigma={self.volatility})"
        )
