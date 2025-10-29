"""Pathwise derivative method for calculating Greeks."""

import numpy as np
from typing import Optional
from ..core.base_option import BaseOption
from ..core.european_option import EuropeanOption


class PathwiseGreeks:
    """Calculate Greeks using pathwise derivative method.

    The pathwise method computes Greeks by differentiating the payoff
    with respect to parameters, then taking expectations.

    For a European call:
        Delta = E[1_{S_T > K} * ∂S_T/∂S_0 * exp(-rT)]
              = E[1_{S_T > K} * S_T/S_0 * exp(-rT)]

    This method is more efficient than finite differences when applicable,
    but requires the payoff to be differentiable (excludes digital options).

    Note: This implementation focuses on European options. Path-dependent
    options require more complex implementations.
    """

    def __init__(self, n_simulations: int = 100000, seed: Optional[int] = None):
        """Initialize pathwise Greeks calculator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def delta(self, option: BaseOption) -> float:
        """Calculate Delta using pathwise derivative.

        For European options:
        Delta = E[1_{payoff > 0} * (∂payoff/∂S_T) * (∂S_T/∂S_0) * exp(-rT)]

        Args:
            option: Option to calculate Delta for

        Returns:
            Delta value
        """
        if not isinstance(option, EuropeanOption):
            raise NotImplementedError(
                "Pathwise Delta currently only implemented for European options"
            )

        # Generate terminal prices
        Z = self.rng.standard_normal(self.n_simulations)
        drift = (option.rate - option.dividend - 0.5 * option.volatility**2) * option.maturity
        diffusion = option.volatility * np.sqrt(option.maturity)
        S_T = option.spot * np.exp(drift + diffusion * Z)

        # ∂S_T/∂S_0 = S_T / S_0
        dS_dS0 = S_T / option.spot

        # ∂payoff/∂S_T
        if option.option_type == 'call':
            # payoff = max(S_T - K, 0)
            # ∂payoff/∂S_T = 1 if S_T > K, else 0
            dpayoff_dST = (S_T > option.strike).astype(float)
        else:  # put
            # payoff = max(K - S_T, 0)
            # ∂payoff/∂S_T = -1 if S_T < K, else 0
            dpayoff_dST = -(S_T < option.strike).astype(float)

        # Chain rule: ∂payoff/∂S_0 = (∂payoff/∂S_T) * (∂S_T/∂S_0)
        dpayoff_dS0 = dpayoff_dST * dS_dS0

        # Expected value (already in present value terms)
        delta = np.mean(dpayoff_dS0) * np.exp(-option.rate * option.maturity)

        return delta

    def vega(self, option: BaseOption) -> float:
        """Calculate Vega using pathwise derivative.

        For European options:
        Vega = E[1_{payoff > 0} * (∂payoff/∂S_T) * (∂S_T/∂σ) * exp(-rT)]

        Args:
            option: Option to calculate Vega for

        Returns:
            Vega value (per 1% change in volatility)
        """
        if not isinstance(option, EuropeanOption):
            raise NotImplementedError(
                "Pathwise Vega currently only implemented for European options"
            )

        # Generate terminal prices and keep Z for derivative
        Z = self.rng.standard_normal(self.n_simulations)
        drift = (option.rate - option.dividend - 0.5 * option.volatility**2) * option.maturity
        diffusion = option.volatility * np.sqrt(option.maturity)
        S_T = option.spot * np.exp(drift + diffusion * Z)

        # ∂S_T/∂σ
        # S_T = S_0 * exp((r - q - 0.5σ²)T + σ√T * Z)
        # ∂S_T/∂σ = S_T * (-σT + √T * Z) = S_T * √T * (Z - σ√T)
        sqrt_T = np.sqrt(option.maturity)
        dS_dsigma = S_T * sqrt_T * (Z - option.volatility * sqrt_T)

        # ∂payoff/∂S_T
        if option.option_type == 'call':
            dpayoff_dST = (S_T > option.strike).astype(float)
        else:  # put
            dpayoff_dST = -(S_T < option.strike).astype(float)

        # Chain rule
        dpayoff_dsigma = dpayoff_dST * dS_dsigma

        # Expected value (per 1% change in vol)
        vega = np.mean(dpayoff_dsigma) * np.exp(-option.rate * option.maturity) / 100

        return vega

    def gamma_pathwise(self, option: BaseOption) -> float:
        """Calculate Gamma using pathwise derivative.

        Gamma is the second derivative, computed as:
        Gamma = E[∂²payoff/∂S₀²]

        This requires differentiating twice, which involves the
        derivative of the indicator function (Dirac delta).

        Args:
            option: Option to calculate Gamma for

        Returns:
            Gamma value
        """
        if not isinstance(option, EuropeanOption):
            raise NotImplementedError(
                "Pathwise Gamma currently only implemented for European options"
            )

        # Generate terminal prices
        Z = self.rng.standard_normal(self.n_simulations)
        drift = (option.rate - option.dividend - 0.5 * option.volatility**2) * option.maturity
        diffusion = option.volatility * np.sqrt(option.maturity)
        S_T = option.spot * np.exp(drift + diffusion * Z)

        # For Gamma, we need the density at the strike
        # This is more complex and typically done using likelihood ratio method
        # or finite differences on Delta

        # Using likelihood ratio method:
        # Gamma ≈ E[payoff * W(Z)]
        # where W(Z) is a weight function

        # Simplified approximation using finite differences on pathwise Delta
        # This is a hybrid approach
        raise NotImplementedError(
            "Pathwise Gamma is complex; use finite difference method instead"
        )
