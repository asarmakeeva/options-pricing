"""Path generation for Monte Carlo simulation."""

import numpy as np
from typing import Optional


class PathGenerator:
    """Generate price paths using Geometric Brownian Motion.

    The underlying asset follows the stochastic differential equation:
        dS = mu * S * dt + sigma * S * dW

    where mu is the drift, sigma is the volatility, and dW is a Wiener process.

    Under risk-neutral measure, mu = r - q, where r is the risk-free rate
    and q is the dividend yield.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize path generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def generate_paths(
        self,
        spot: float,
        maturity: float,
        rate: float,
        volatility: float,
        n_paths: int,
        n_steps: int,
        dividend: float = 0.0,
        antithetic: bool = False
    ) -> np.ndarray:
        """Generate price paths using Geometric Brownian Motion.

        Args:
            spot: Initial spot price
            maturity: Time to maturity
            rate: Risk-free rate
            volatility: Volatility
            n_paths: Number of paths to generate
            n_steps: Number of time steps per path
            dividend: Dividend yield
            antithetic: If True, use antithetic variates

        Returns:
            Array of paths with shape (n_paths, n_steps + 1)
            First column is the spot price
        """
        dt = maturity / n_steps
        drift = (rate - dividend - 0.5 * volatility**2) * dt
        diffusion = volatility * np.sqrt(dt)

        # Generate random normal variables
        if antithetic:
            # Generate half the paths, then create antithetic pairs
            half_paths = n_paths // 2
            Z = self.rng.standard_normal((half_paths, n_steps))
            Z_full = np.vstack([Z, -Z])
            if n_paths % 2 == 1:
                # If odd number of paths, add one more
                Z_extra = self.rng.standard_normal((1, n_steps))
                Z_full = np.vstack([Z_full, Z_extra])
        else:
            Z = self.rng.standard_normal((n_paths, n_steps))
            Z_full = Z

        # Calculate returns
        returns = drift + diffusion * Z_full

        # Initialize paths array (includes initial spot price)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot

        # Generate paths using cumulative sum of log returns
        log_paths = np.cumsum(returns, axis=1)
        paths[:, 1:] = spot * np.exp(log_paths)

        return paths

    def generate_terminal_prices(
        self,
        spot: float,
        maturity: float,
        rate: float,
        volatility: float,
        n_paths: int,
        dividend: float = 0.0,
        antithetic: bool = False
    ) -> np.ndarray:
        """Generate only terminal prices (more efficient for European options).

        Args:
            spot: Initial spot price
            maturity: Time to maturity
            rate: Risk-free rate
            volatility: Volatility
            n_paths: Number of paths to generate
            dividend: Dividend yield
            antithetic: If True, use antithetic variates

        Returns:
            Array of terminal prices with shape (n_paths,)
        """
        drift = (rate - dividend - 0.5 * volatility**2) * maturity
        diffusion = volatility * np.sqrt(maturity)

        # Generate random normal variables
        if antithetic:
            half_paths = n_paths // 2
            Z = self.rng.standard_normal(half_paths)
            Z_full = np.concatenate([Z, -Z])
            if n_paths % 2 == 1:
                Z_extra = self.rng.standard_normal(1)
                Z_full = np.concatenate([Z_full, Z_extra])
        else:
            Z_full = self.rng.standard_normal(n_paths)

        # Calculate terminal prices
        terminal_prices = spot * np.exp(drift + diffusion * Z_full)

        return terminal_prices
