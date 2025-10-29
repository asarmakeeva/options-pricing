"""Control variates variance reduction technique."""

import numpy as np
from typing import Tuple, Optional
from ..core.base_option import BaseOption
from ..core.european_option import EuropeanOption
from ..core.black_scholes import BlackScholes


class ControlVariates:
    """Control variates variance reduction technique.

    Uses a correlated variable with known expectation to reduce variance.

    For option pricing, we use a European option as the control variate:
    - Price the target option and a European option on the same paths
    - European option has closed-form solution (Black-Scholes)
    - Adjust the estimate using the difference between MC and BS prices

    The adjusted estimator is:
        Y_adjusted = Y + c * (X - E[X])

    where Y is our target, X is the control, E[X] is known expectation,
    and c is optimally chosen to minimize variance.
    """

    def __init__(self, control_option: Optional[EuropeanOption] = None):
        """Initialize control variates technique.

        Args:
            control_option: European option to use as control variate.
                           If None, will create one matching the target option.
        """
        self.control_option = control_option

    def apply(
        self,
        option: BaseOption,
        payoffs: np.ndarray,
        paths: Optional[np.ndarray],
        path_generator
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply control variates to reduce variance.

        Args:
            option: Option to price
            payoffs: Original payoffs
            paths: Price paths (or terminal prices)
            path_generator: PathGenerator instance

        Returns:
            Tuple of (adjusted_payoffs, paths)
        """
        # If no control option specified, create a European option with same parameters
        if self.control_option is None:
            control = EuropeanOption(
                option_type=option.option_type,
                strike=option.strike,
                maturity=option.maturity,
                spot=option.spot,
                rate=option.rate,
                volatility=option.volatility,
                dividend=option.dividend
            )
        else:
            control = self.control_option

        # Calculate control payoffs from the same paths
        if paths is None:
            # Need to regenerate paths
            n_paths = len(payoffs)
            if isinstance(option, EuropeanOption):
                terminal_prices = path_generator.generate_terminal_prices(
                    spot=option.spot,
                    maturity=option.maturity,
                    rate=option.rate,
                    volatility=option.volatility,
                    n_paths=n_paths,
                    dividend=option.dividend
                )
                control_payoffs = control.payoff(terminal_prices)
            else:
                paths = path_generator.generate_paths(
                    spot=option.spot,
                    maturity=option.maturity,
                    rate=option.rate,
                    volatility=option.volatility,
                    n_paths=n_paths,
                    n_steps=252,
                    dividend=option.dividend
                )
                control_payoffs = control.payoff(paths[:, -1])  # European uses terminal only
        else:
            # Use existing paths
            if paths.ndim == 1:
                # Terminal prices only
                control_payoffs = control.payoff(paths)
            else:
                # Full paths, extract terminal prices
                control_payoffs = control.payoff(paths[:, -1])

        # Get Black-Scholes price for control (known expectation)
        bs_price = BlackScholes.price(
            option_type=control.option_type,
            spot=control.spot,
            strike=control.strike,
            maturity=control.maturity,
            rate=control.rate,
            volatility=control.volatility,
            dividend=control.dividend
        )

        # Expected discounted payoff
        expected_control = bs_price

        # Discount control payoffs
        discounted_control = np.exp(-control.rate * control.maturity) * control_payoffs

        # Calculate optimal coefficient c
        # c = -Cov(Y, X) / Var(X)
        # Minimizes variance of Y + c(X - E[X])
        discounted_payoffs = np.exp(-option.rate * option.maturity) * payoffs

        covariance = np.cov(discounted_payoffs, discounted_control)[0, 1]
        variance_control = np.var(discounted_control)

        if variance_control > 1e-10:
            c = -covariance / variance_control
        else:
            c = 0.0

        # Apply control variate adjustment
        adjustment = c * (discounted_control - expected_control)
        adjusted_discounted = discounted_payoffs + adjustment

        # Convert back to undiscounted payoffs
        adjusted_payoffs = adjusted_discounted * np.exp(option.rate * option.maturity)

        return adjusted_payoffs, paths

    def __repr__(self) -> str:
        """String representation."""
        if self.control_option is not None:
            return f"ControlVariates(control={self.control_option})"
        return "ControlVariates()"
