"""Volatility surface representation and interpolation."""

import numpy as np
from scipy.interpolate import RectBivariateSpline, griddata
from typing import Tuple, Optional
import pandas as pd


class VolatilitySurface:
    """Representation of implied volatility surface.

    The volatility surface σ(K, T) gives implied volatility as a function
    of strike K and maturity T.

    Supports:
    - 2D interpolation across strikes and maturities
    - Extrapolation with sensible boundaries
    - Surface visualization
    """

    def __init__(self):
        """Initialize volatility surface."""
        self.strikes = None
        self.maturities = None
        self.volatilities = None
        self.interpolator = None
        self.spot = None

    def fit(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        volatilities: np.ndarray,
        spot: float
    ):
        """Fit volatility surface to market data.

        Args:
            strikes: Array of strike prices (or 2D grid)
            maturities: Array of maturities (or 2D grid)
            volatilities: Array of implied volatilities (2D grid)
            spot: Current spot price

        Raises:
            ValueError: If dimensions don't match
        """
        self.spot = spot

        # If 1D arrays provided, assume they're grid axes
        if strikes.ndim == 1 and maturities.ndim == 1:
            if volatilities.shape != (len(maturities), len(strikes)):
                raise ValueError(
                    f"Volatilities shape {volatilities.shape} doesn't match "
                    f"expected ({len(maturities)}, {len(strikes)})"
                )

            self.strikes = strikes
            self.maturities = maturities
            self.volatilities = volatilities

            # Create interpolator
            self.interpolator = RectBivariateSpline(
                self.maturities,
                self.strikes,
                self.volatilities,
                kx=min(3, len(self.maturities) - 1),
                ky=min(3, len(self.strikes) - 1)
            )

        else:
            # Scattered data - use griddata
            # Flatten arrays
            self.strikes_flat = strikes.flatten()
            self.maturities_flat = maturities.flatten()
            self.volatilities_flat = volatilities.flatten()

            # Store for interpolation
            self.scattered = True

    def volatility(self, strike: float, maturity: float) -> float:
        """Get implied volatility for given strike and maturity.

        Args:
            strike: Strike price
            maturity: Time to maturity

        Returns:
            Implied volatility

        Raises:
            ValueError: If surface hasn't been fitted
        """
        if self.interpolator is None:
            raise ValueError("Surface must be fitted before querying")

        # Interpolate
        vol = float(self.interpolator(maturity, strike))

        # Ensure positive volatility
        return max(vol, 0.01)

    def slice_by_maturity(self, maturity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get volatility smile for a specific maturity.

        Args:
            maturity: Time to maturity

        Returns:
            Tuple of (strikes, volatilities)
        """
        if self.interpolator is None:
            raise ValueError("Surface must be fitted before querying")

        vols = np.array([self.interpolator(maturity, k)[0, 0] for k in self.strikes])

        return self.strikes, vols

    def slice_by_strike(self, strike: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get term structure for a specific strike.

        Args:
            strike: Strike price

        Returns:
            Tuple of (maturities, volatilities)
        """
        if self.interpolator is None:
            raise ValueError("Surface must be fitted before querying")

        vols = np.array([self.interpolator(t, strike)[0, 0] for t in self.maturities])

        return self.maturities, vols

    def to_dataframe(self) -> pd.DataFrame:
        """Convert surface to pandas DataFrame.

        Returns:
            DataFrame with strikes as columns and maturities as rows
        """
        if self.volatilities is None:
            raise ValueError("Surface must be fitted before conversion")

        df = pd.DataFrame(
            self.volatilities,
            index=self.maturities,
            columns=self.strikes
        )
        df.index.name = 'Maturity'
        df.columns.name = 'Strike'

        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, spot: float) -> 'VolatilitySurface':
        """Create surface from pandas DataFrame.

        Args:
            df: DataFrame with strikes as columns and maturities as rows
            spot: Current spot price

        Returns:
            VolatilitySurface instance
        """
        surface = cls()

        strikes = df.columns.values.astype(float)
        maturities = df.index.values.astype(float)
        volatilities = df.values.astype(float)

        surface.fit(strikes, maturities, volatilities, spot)

        return surface

    def atm_volatility(self, maturity: float) -> float:
        """Get at-the-money volatility for given maturity.

        Args:
            maturity: Time to maturity

        Returns:
            ATM implied volatility
        """
        return self.volatility(self.spot, maturity)

    def moneyness_slice(
        self,
        maturity: float,
        moneyness_range: Tuple[float, float] = (0.8, 1.2),
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get volatility smile in terms of moneyness.

        Args:
            maturity: Time to maturity
            moneyness_range: Range of moneyness (K/S) to evaluate
            n_points: Number of points

        Returns:
            Tuple of (moneyness, volatilities)
        """
        moneyness = np.linspace(moneyness_range[0], moneyness_range[1], n_points)
        strikes = moneyness * self.spot

        vols = np.array([self.volatility(k, maturity) for k in strikes])

        return moneyness, vols
