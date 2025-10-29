"""Importance sampling variance reduction technique."""

import numpy as np
from typing import Tuple, Optional
from ..core.base_option import BaseOption
from ..core.european_option import EuropeanOption


class ImportanceSampling:
    """Importance sampling variance reduction technique.

    Instead of sampling from the original distribution, sample from a
    distribution that emphasizes important regions (e.g., in-the-money).

    For options, we shift the drift to increase probability of payoff:
    - Sample from a shifted distribution
    - Weight samples by likelihood ratio to correct the bias

    The shifted measure uses drift parameter theta:
        dS = (mu + theta*sigma) * S * dt + sigma * S * dW

    The likelihood ratio is:
        L = exp(-theta * W_T - 0.5 * theta^2 * T)

    where W_T is the terminal value of the Brownian motion.
    """

    def __init__(self, theta: float = 0.5):
        """Initialize importance sampling.

        Args:
            theta: Drift shift parameter. Positive values shift distribution
                   toward higher prices (good for calls), negative for lower
                   prices (good for puts).
        """
        self.theta = theta

    def apply(
        self,
        option: BaseOption,
        payoffs: Optional[np.ndarray],
        paths: Optional[np.ndarray],
        path_generator
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply importance sampling to option pricing.

        Args:
            option: Option to price
            payoffs: Existing payoffs (ignored, will regenerate)
            paths: Existing paths (ignored, will regenerate)
            path_generator: PathGenerator instance

        Returns:
            Tuple of (weighted_payoffs, new_paths)
        """
        n_paths = len(payoffs) if payoffs is not None else 100000

        # Auto-adjust theta based on option type and moneyness
        theta = self._get_optimal_theta(option)

        # Check if we can use terminal prices only
        use_terminal_only = isinstance(option, EuropeanOption)

        if use_terminal_only:
            # Generate terminal prices with shifted drift
            terminal_prices = self._generate_shifted_terminal(
                option, n_paths, theta, path_generator
            )
            new_payoffs = option.payoff(terminal_prices)
            new_paths = None

            # Calculate likelihood ratios
            # W_T = (log(S_T/S_0) - (r - q - 0.5*sigma^2)*T) / (sigma*sqrt(T))
            drift = (option.rate - option.dividend - 0.5 * option.volatility**2) * option.maturity
            W_T = (np.log(terminal_prices / option.spot) - drift) / (
                option.volatility * np.sqrt(option.maturity)
            )
        else:
            # Generate full paths with shifted drift
            new_paths = self._generate_shifted_paths(
                option, n_paths, theta, path_generator
            )
            new_payoffs = option.payoff(new_paths)

            # Calculate likelihood ratios from terminal prices
            drift = (option.rate - option.dividend - 0.5 * option.volatility**2) * option.maturity
            terminal_prices = new_paths[:, -1]
            W_T = (np.log(terminal_prices / option.spot) - drift) / (
                option.volatility * np.sqrt(option.maturity)
            )

        # Likelihood ratio
        likelihood_ratio = np.exp(-theta * W_T - 0.5 * theta**2 * option.maturity)

        # Apply weights to payoffs
        weighted_payoffs = new_payoffs * likelihood_ratio

        return weighted_payoffs, new_paths

    def _get_optimal_theta(self, option: BaseOption) -> float:
        """Determine optimal theta based on option characteristics.

        Args:
            option: Option to price

        Returns:
            Optimal theta value
        """
        # If theta was explicitly set, use it
        if self.theta != 0.5:
            return self.theta

        # Auto-select based on option type and moneyness
        moneyness = option.spot / option.strike

        if option.option_type == 'call':
            # For OTM calls, shift toward higher prices
            if moneyness < 0.95:
                return 1.0
            elif moneyness < 1.05:
                return 0.5
            else:  # ITM calls
                return 0.2
        else:  # put
            # For OTM puts, shift toward lower prices
            if moneyness > 1.05:
                return -1.0
            elif moneyness > 0.95:
                return -0.5
            else:  # ITM puts
                return -0.2

    def _generate_shifted_terminal(
        self,
        option: BaseOption,
        n_paths: int,
        theta: float,
        path_generator
    ) -> np.ndarray:
        """Generate terminal prices with shifted drift.

        Args:
            option: Option to price
            n_paths: Number of paths
            theta: Drift shift
            path_generator: PathGenerator instance

        Returns:
            Array of terminal prices
        """
        # Generate standard normal random variables
        Z = path_generator.rng.standard_normal(n_paths)

        # Apply shifted drift
        # Original: S_T = S_0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        # Shifted: S_T = S_0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*(Z + theta*sqrt(T)))
        drift = (option.rate - option.dividend - 0.5 * option.volatility**2) * option.maturity
        diffusion = option.volatility * np.sqrt(option.maturity)

        # Add theta*sqrt(T) to the random variable
        shifted_Z = Z + theta * np.sqrt(option.maturity)

        terminal_prices = option.spot * np.exp(drift + diffusion * shifted_Z)

        return terminal_prices

    def _generate_shifted_paths(
        self,
        option: BaseOption,
        n_paths: int,
        theta: float,
        path_generator
    ) -> np.ndarray:
        """Generate full paths with shifted drift.

        Args:
            option: Option to price
            n_paths: Number of paths
            theta: Drift shift
            path_generator: PathGenerator instance

        Returns:
            Array of price paths
        """
        n_steps = 252
        dt = option.maturity / n_steps

        # Generate standard normal random variables
        Z = path_generator.rng.standard_normal((n_paths, n_steps))

        # Apply shift
        shifted_Z = Z + theta * np.sqrt(dt)

        # Generate paths
        drift = (option.rate - option.dividend - 0.5 * option.volatility**2) * dt
        diffusion = option.volatility * np.sqrt(dt)

        returns = drift + diffusion * shifted_Z

        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = option.spot

        # Generate using cumulative sum
        log_paths = np.cumsum(returns, axis=1)
        paths[:, 1:] = option.spot * np.exp(log_paths)

        return paths

    def __repr__(self) -> str:
        """String representation."""
        return f"ImportanceSampling(theta={self.theta})"
