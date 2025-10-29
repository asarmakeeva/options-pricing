"""Finite difference method for calculating Greeks."""

import numpy as np
from typing import Dict, Optional
from ..core.base_option import BaseOption
from ..monte_carlo.simulator import MonteCarloSimulator
import copy


class FiniteDifferenceGreeks:
    """Calculate option Greeks using finite difference approximations.

    Greeks are computed by bumping parameters and repricing:
    - Delta: ∂V/∂S ≈ (V(S+h) - V(S-h)) / (2h)
    - Gamma: ∂²V/∂S² ≈ (V(S+h) - 2V(S) + V(S-h)) / h²
    - Vega: ∂V/∂σ ≈ (V(σ+h) - V(σ-h)) / (2h)
    - Theta: ∂V/∂t ≈ (V(t) - V(t-h)) / h
    - Rho: ∂V/∂r ≈ (V(r+h) - V(r-h)) / (2h)

    Attributes:
        simulator: MonteCarloSimulator instance
    """

    def __init__(
        self,
        simulator: MonteCarloSimulator,
        spot_bump: float = 0.01,
        vol_bump: float = 0.01,
        rate_bump: float = 0.0001,
        time_bump: float = 1.0 / 365.0
    ):
        """Initialize finite difference calculator.

        Args:
            simulator: MonteCarloSimulator to use for pricing
            spot_bump: Relative bump size for spot (1% default)
            vol_bump: Absolute bump size for volatility (1% default)
            rate_bump: Absolute bump size for rate (1bp default)
            time_bump: Absolute bump size for time (1 day default)
        """
        self.simulator = simulator
        self.spot_bump = spot_bump
        self.vol_bump = vol_bump
        self.rate_bump = rate_bump
        self.time_bump = time_bump

    def delta(
        self,
        option: BaseOption,
        variance_reduction: Optional[object] = None
    ) -> float:
        """Calculate Delta: sensitivity to spot price.

        Delta = ∂V/∂S

        Args:
            option: Option to calculate Delta for
            variance_reduction: Variance reduction technique

        Returns:
            Delta value
        """
        # Bump spot up
        option_up = copy.deepcopy(option)
        option_up.spot *= (1 + self.spot_bump)
        price_up, _ = self.simulator.price(option_up, variance_reduction)

        # Bump spot down
        option_down = copy.deepcopy(option)
        option_down.spot *= (1 - self.spot_bump)
        price_down, _ = self.simulator.price(option_down, variance_reduction)

        # Central difference
        delta = (price_up - price_down) / (2 * option.spot * self.spot_bump)

        return delta

    def gamma(
        self,
        option: BaseOption,
        variance_reduction: Optional[object] = None
    ) -> float:
        """Calculate Gamma: second derivative with respect to spot.

        Gamma = ∂²V/∂S²

        Args:
            option: Option to calculate Gamma for
            variance_reduction: Variance reduction technique

        Returns:
            Gamma value
        """
        # Price at current spot
        price_center, _ = self.simulator.price(option, variance_reduction)

        # Bump spot up
        option_up = copy.deepcopy(option)
        option_up.spot *= (1 + self.spot_bump)
        price_up, _ = self.simulator.price(option_up, variance_reduction)

        # Bump spot down
        option_down = copy.deepcopy(option)
        option_down.spot *= (1 - self.spot_bump)
        price_down, _ = self.simulator.price(option_down, variance_reduction)

        # Second derivative
        h = option.spot * self.spot_bump
        gamma = (price_up - 2 * price_center + price_down) / (h ** 2)

        return gamma

    def vega(
        self,
        option: BaseOption,
        variance_reduction: Optional[object] = None
    ) -> float:
        """Calculate Vega: sensitivity to volatility.

        Vega = ∂V/∂σ (reported per 1% change in vol)

        Args:
            option: Option to calculate Vega for
            variance_reduction: Variance reduction technique

        Returns:
            Vega value (per 1% change)
        """
        # Bump volatility up
        option_up = copy.deepcopy(option)
        option_up.volatility += self.vol_bump
        price_up, _ = self.simulator.price(option_up, variance_reduction)

        # Bump volatility down
        option_down = copy.deepcopy(option)
        option_down.volatility -= self.vol_bump
        price_down, _ = self.simulator.price(option_down, variance_reduction)

        # Central difference (per 1% change in volatility)
        vega = (price_up - price_down) / (2 * self.vol_bump * 100)

        return vega

    def theta(
        self,
        option: BaseOption,
        variance_reduction: Optional[object] = None
    ) -> float:
        """Calculate Theta: time decay.

        Theta = ∂V/∂t (reported per day)

        Args:
            option: Option to calculate Theta for
            variance_reduction: Variance reduction technique

        Returns:
            Theta value (per day)
        """
        # Price at current time
        price_now, _ = self.simulator.price(option, variance_reduction)

        # Price one day earlier (maturity reduced)
        option_earlier = copy.deepcopy(option)
        option_earlier.maturity -= self.time_bump
        if option_earlier.maturity <= 0:
            # Option has expired
            return -price_now / self.time_bump

        price_earlier, _ = self.simulator.price(option_earlier, variance_reduction)

        # Theta (negative because we go backward in time)
        theta = (price_now - price_earlier) / self.time_bump

        return theta

    def rho(
        self,
        option: BaseOption,
        variance_reduction: Optional[object] = None
    ) -> float:
        """Calculate Rho: sensitivity to interest rate.

        Rho = ∂V/∂r (reported per 1% change in rate)

        Args:
            option: Option to calculate Rho for
            variance_reduction: Variance reduction technique

        Returns:
            Rho value (per 1% change)
        """
        # Bump rate up
        option_up = copy.deepcopy(option)
        option_up.rate += self.rate_bump
        price_up, _ = self.simulator.price(option_up, variance_reduction)

        # Bump rate down
        option_down = copy.deepcopy(option)
        option_down.rate -= self.rate_bump
        price_down, _ = self.simulator.price(option_down, variance_reduction)

        # Central difference (per 1% change in rate)
        rho = (price_up - price_down) / (2 * self.rate_bump * 100)

        return rho

    def all_greeks(
        self,
        option: BaseOption,
        variance_reduction: Optional[object] = None
    ) -> Dict[str, float]:
        """Calculate all Greeks at once.

        Args:
            option: Option to calculate Greeks for
            variance_reduction: Variance reduction technique

        Returns:
            Dictionary with all Greeks
        """
        return {
            'delta': self.delta(option, variance_reduction),
            'gamma': self.gamma(option, variance_reduction),
            'vega': self.vega(option, variance_reduction),
            'theta': self.theta(option, variance_reduction),
            'rho': self.rho(option, variance_reduction)
        }
