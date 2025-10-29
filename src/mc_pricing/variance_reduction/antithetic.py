"""Antithetic variates variance reduction technique."""

import numpy as np
from typing import Tuple, Optional
from ..core.base_option import BaseOption
from ..core.european_option import EuropeanOption


class AntitheticVariates:
    """Antithetic variates variance reduction technique.

    For each random sample Z, also use -Z. This creates negative correlation
    between pairs of samples, reducing variance.

    For a standard normal variable Z:
    - If Z leads to high payoff, -Z typically leads to low payoff
    - The average of the two is more stable than either alone

    Variance reduction is most effective when payoff is monotonic in Z.
    """

    def __init__(self):
        """Initialize antithetic variates technique."""
        pass

    def apply(
        self,
        option: BaseOption,
        payoffs: Optional[np.ndarray],
        paths: Optional[np.ndarray],
        path_generator
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply antithetic variates to option pricing.

        Args:
            option: Option to price
            payoffs: Existing payoffs (ignored, will regenerate)
            paths: Existing paths (ignored, will regenerate)
            path_generator: PathGenerator instance

        Returns:
            Tuple of (new_payoffs, new_paths)
        """
        # Determine number of paths to generate
        n_paths = len(payoffs) if payoffs is not None else 100000

        # Check if we can use terminal prices only
        use_terminal_only = isinstance(option, EuropeanOption)

        if use_terminal_only:
            # Generate terminal prices with antithetic variates
            terminal_prices = path_generator.generate_terminal_prices(
                spot=option.spot,
                maturity=option.maturity,
                rate=option.rate,
                volatility=option.volatility,
                n_paths=n_paths,
                dividend=option.dividend,
                antithetic=True
            )
            new_payoffs = option.payoff(terminal_prices)
            new_paths = None
        else:
            # Generate full paths with antithetic variates
            new_paths = path_generator.generate_paths(
                spot=option.spot,
                maturity=option.maturity,
                rate=option.rate,
                volatility=option.volatility,
                n_paths=n_paths,
                n_steps=252,  # Default step count
                dividend=option.dividend,
                antithetic=True
            )
            new_payoffs = option.payoff(new_paths)

        return new_payoffs, new_paths

    def __repr__(self) -> str:
        """String representation."""
        return "AntitheticVariates()"
