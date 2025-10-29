"""Main Monte Carlo simulator for option pricing."""

import numpy as np
from typing import Tuple, Optional, Any
from ..core.base_option import BaseOption
from ..core.european_option import EuropeanOption
from .path_generator import PathGenerator


class MonteCarloSimulator:
    """Monte Carlo simulator for option pricing.

    This class provides the main interface for pricing options using
    Monte Carlo simulation with various variance reduction techniques.
    """

    def __init__(
        self,
        n_simulations: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ):
        """Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            n_steps: Number of time steps per path (default: 252 for daily steps)
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.path_generator = PathGenerator(seed=seed)

    def price(
        self,
        option: BaseOption,
        variance_reduction: Optional[Any] = None,
        return_paths: bool = False
    ) -> Tuple[float, float]:
        """Price an option using Monte Carlo simulation.

        Args:
            option: Option to price
            variance_reduction: Variance reduction technique to apply
            return_paths: If True, return paths as third element

        Returns:
            Tuple of (price, standard_error) or (price, standard_error, paths)
        """
        # Check if we can use terminal prices only (European options)
        use_terminal_only = isinstance(option, EuropeanOption)

        if use_terminal_only:
            # More efficient: only generate terminal prices
            terminal_prices = self.path_generator.generate_terminal_prices(
                spot=option.spot,
                maturity=option.maturity,
                rate=option.rate,
                volatility=option.volatility,
                n_paths=self.n_simulations,
                dividend=option.dividend,
                antithetic=False
            )
            payoffs = option.payoff(terminal_prices)
            paths = None
        else:
            # Generate full paths for path-dependent options
            paths = self.path_generator.generate_paths(
                spot=option.spot,
                maturity=option.maturity,
                rate=option.rate,
                volatility=option.volatility,
                n_paths=self.n_simulations,
                n_steps=self.n_steps,
                dividend=option.dividend,
                antithetic=False
            )
            payoffs = option.payoff(paths)

        # Apply variance reduction if specified
        if variance_reduction is not None:
            payoffs, paths = variance_reduction.apply(
                option, payoffs, paths, self.path_generator
            )

        # Discount payoffs to present value
        discounted_payoffs = np.exp(-option.rate * option.maturity) * payoffs

        # Calculate price and standard error
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))

        if return_paths:
            return price, std_error, paths
        else:
            return price, std_error

    def price_with_confidence_interval(
        self,
        option: BaseOption,
        confidence: float = 0.95,
        variance_reduction: Optional[Any] = None
    ) -> Tuple[float, float, float]:
        """Price option and return confidence interval.

        Args:
            option: Option to price
            confidence: Confidence level (default: 0.95 for 95% CI)
            variance_reduction: Variance reduction technique

        Returns:
            Tuple of (price, lower_bound, upper_bound)
        """
        from scipy.stats import norm

        price, std_error = self.price(option, variance_reduction)

        # Calculate confidence interval
        z_score = norm.ppf((1 + confidence) / 2)
        margin = z_score * std_error

        return price, price - margin, price + margin

    def convergence_test(
        self,
        option: BaseOption,
        n_trials: np.ndarray,
        variance_reduction: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Test convergence of MC estimator for different sample sizes.

        Args:
            option: Option to price
            n_trials: Array of different sample sizes to test
            variance_reduction: Variance reduction technique

        Returns:
            Tuple of (prices, std_errors) for each sample size
        """
        prices = []
        errors = []

        original_n = self.n_simulations

        for n in n_trials:
            self.n_simulations = n
            price, std_error = self.price(option, variance_reduction)
            prices.append(price)
            errors.append(std_error)

        # Restore original setting
        self.n_simulations = original_n

        return np.array(prices), np.array(errors)
