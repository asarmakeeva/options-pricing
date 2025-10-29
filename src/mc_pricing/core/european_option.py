"""European option implementation."""

import numpy as np
from .base_option import BaseOption


class EuropeanOption(BaseOption):
    """European option that can only be exercised at maturity.

    The payoff is calculated based on the final spot price:
    - Call: max(S_T - K, 0)
    - Put: max(K - S_T, 0)
    """

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculate European option payoff.

        For European options, only the final price matters.

        Args:
            paths: Array of price paths, shape (n_paths, n_steps)
                   or (n_paths,) for final prices only

        Returns:
            Array of payoffs, shape (n_paths,)
        """
        # Get final prices
        if paths.ndim == 1:
            final_prices = paths
        else:
            final_prices = paths[:, -1]

        # Calculate payoff
        if self.option_type == 'call':
            payoff = np.maximum(final_prices - self.strike, 0.0)
        else:  # put
            payoff = np.maximum(self.strike - final_prices, 0.0)

        return payoff
