"""Volatility smile calibration."""

import numpy as np
from scipy.optimize import minimize, least_squares
from typing import List, Tuple, Optional, Callable
from ..core.black_scholes import BlackScholes


class SmileCalibration:
    """Calibrate implied volatility smile to market data.

    Supports multiple parametric forms:
    - Quadratic: σ(K) = a + b*(K-K₀) + c*(K-K₀)²
    - SVI (Stochastic Volatility Inspired): w(k) = a + b*(ρ*(k-m) + √((k-m)² + σ²))
    - Polynomial: σ(K) = Σ aᵢ*(K-K₀)ⁱ

    where K is the strike price and K₀ is typically ATM strike.
    """

    def __init__(self, model: str = 'quadratic'):
        """Initialize smile calibration.

        Args:
            model: Parametric model to use ('quadratic', 'svi', 'polynomial')

        Raises:
            ValueError: If model is not supported
        """
        if model not in ['quadratic', 'svi', 'polynomial']:
            raise ValueError(f"Unsupported model: {model}")

        self.model = model
        self.params = None
        self.atm_strike = None

    def fit(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        spot: float,
        maturity: float,
        rate: float,
        option_type: str = 'call',
        dividend: float = 0.0
    ) -> dict:
        """Fit smile model to market data.

        Args:
            strikes: Array of strike prices
            market_prices: Array of observed market prices
            spot: Current spot price
            maturity: Time to maturity
            rate: Risk-free rate
            option_type: 'call' or 'put'
            dividend: Dividend yield

        Returns:
            Dictionary with calibrated parameters and fit statistics
        """
        # First, calculate implied volatilities from market prices
        implied_vols = self._calculate_implied_vols(
            strikes, market_prices, spot, maturity, rate, option_type, dividend
        )

        # Filter out any None values (failed calibrations)
        valid_idx = [i for i, vol in enumerate(implied_vols) if vol is not None]
        strikes_valid = strikes[valid_idx]
        implied_vols_valid = np.array([implied_vols[i] for i in valid_idx])

        if len(strikes_valid) < 2:
            raise ValueError("Not enough valid market prices to calibrate")

        # Use ATM strike as reference
        self.atm_strike = spot  # Could also use closest strike to spot

        # Fit the model
        if self.model == 'quadratic':
            self.params = self._fit_quadratic(strikes_valid, implied_vols_valid)
        elif self.model == 'svi':
            self.params = self._fit_svi(strikes_valid, implied_vols_valid)
        elif self.model == 'polynomial':
            self.params = self._fit_polynomial(strikes_valid, implied_vols_valid)

        # Calculate fit statistics
        fitted_vols = self.volatility(strikes_valid)
        rmse = np.sqrt(np.mean((fitted_vols - implied_vols_valid) ** 2))
        max_error = np.max(np.abs(fitted_vols - implied_vols_valid))

        return {
            'params': self.params,
            'rmse': rmse,
            'max_error': max_error,
            'n_points': len(strikes_valid)
        }

    def volatility(self, strikes: np.ndarray) -> np.ndarray:
        """Get implied volatility for given strikes using fitted model.

        Args:
            strikes: Array of strike prices

        Returns:
            Array of implied volatilities

        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")

        if self.model == 'quadratic':
            return self._quadratic_vol(strikes, self.params)
        elif self.model == 'svi':
            return self._svi_vol(strikes, self.params)
        elif self.model == 'polynomial':
            return self._polynomial_vol(strikes, self.params)

    def _calculate_implied_vols(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        spot: float,
        maturity: float,
        rate: float,
        option_type: str,
        dividend: float
    ) -> List[Optional[float]]:
        """Calculate implied volatilities from market prices.

        Args:
            strikes: Strike prices
            market_prices: Market prices
            spot: Spot price
            maturity: Maturity
            rate: Risk-free rate
            option_type: Option type
            dividend: Dividend yield

        Returns:
            List of implied volatilities (None for failed calibrations)
        """
        implied_vols = []

        for strike, price in zip(strikes, market_prices):
            iv = BlackScholes.implied_volatility(
                option_type=option_type,
                market_price=price,
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                dividend=dividend
            )
            implied_vols.append(iv)

        return implied_vols

    def _fit_quadratic(self, strikes: np.ndarray, vols: np.ndarray) -> dict:
        """Fit quadratic model: σ(K) = a + b*(K-K₀) + c*(K-K₀)².

        Args:
            strikes: Strike prices
            vols: Implied volatilities

        Returns:
            Dictionary with parameters a, b, c
        """
        # Transform strikes to moneyness
        moneyness = (strikes - self.atm_strike) / self.atm_strike

        # Fit quadratic polynomial
        coeffs = np.polyfit(moneyness, vols, deg=2)

        return {'a': coeffs[2], 'b': coeffs[1], 'c': coeffs[0]}

    def _quadratic_vol(self, strikes: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate quadratic model.

        Args:
            strikes: Strike prices
            params: Model parameters

        Returns:
            Implied volatilities
        """
        moneyness = (strikes - self.atm_strike) / self.atm_strike
        vols = params['a'] + params['b'] * moneyness + params['c'] * moneyness**2
        return np.maximum(vols, 0.01)  # Ensure positive volatility

    def _fit_svi(self, strikes: np.ndarray, vols: np.ndarray) -> dict:
        """Fit SVI model.

        SVI formula: w(k) = a + b*(ρ*(k-m) + √((k-m)² + σ²))
        where w = σ²*T (total variance) and k = log(K/F) (log-moneyness)

        Args:
            strikes: Strike prices
            vols: Implied volatilities

        Returns:
            Dictionary with SVI parameters
        """
        # Convert to log-moneyness (assuming forward = spot for simplicity)
        forward = self.atm_strike
        k = np.log(strikes / forward)

        # Total variance
        # For simplicity, assume T=1, so w ≈ σ²
        w = vols ** 2

        # Initial guess for SVI parameters
        # [a, b, rho, m, sigma]
        x0 = [
            np.mean(w),  # a: level
            0.1,  # b: slope
            0.0,  # rho: correlation
            0.0,  # m: ATM position
            0.1   # sigma: width
        ]

        # Constraints for SVI to ensure no arbitrage
        def svi_objective(x):
            a, b, rho, m, sigma = x
            w_pred = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
            return np.sum((w_pred - w) ** 2)

        # Bounds to ensure valid SVI parameters
        bounds = [
            (0, None),      # a >= 0
            (0, None),      # b >= 0
            (-0.999, 0.999),  # -1 < rho < 1
            (None, None),   # m unbounded
            (0.01, None)    # sigma > 0
        ]

        result = minimize(svi_objective, x0, bounds=bounds, method='L-BFGS-B')

        a, b, rho, m, sigma = result.x

        return {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma}

    def _svi_vol(self, strikes: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate SVI model.

        Args:
            strikes: Strike prices
            params: SVI parameters

        Returns:
            Implied volatilities
        """
        forward = self.atm_strike
        k = np.log(strikes / forward)

        a = params['a']
        b = params['b']
        rho = params['rho']
        m = params['m']
        sigma = params['sigma']

        w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        # Convert total variance back to volatility (assuming T=1)
        vols = np.sqrt(np.maximum(w, 0.0001))

        return vols

    def _fit_polynomial(
        self,
        strikes: np.ndarray,
        vols: np.ndarray,
        degree: int = 4
    ) -> dict:
        """Fit polynomial model.

        Args:
            strikes: Strike prices
            vols: Implied volatilities
            degree: Polynomial degree

        Returns:
            Dictionary with polynomial coefficients
        """
        moneyness = (strikes - self.atm_strike) / self.atm_strike
        coeffs = np.polyfit(moneyness, vols, deg=degree)

        return {'coeffs': coeffs, 'degree': degree}

    def _polynomial_vol(self, strikes: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate polynomial model.

        Args:
            strikes: Strike prices
            params: Polynomial parameters

        Returns:
            Implied volatilities
        """
        moneyness = (strikes - self.atm_strike) / self.atm_strike
        vols = np.polyval(params['coeffs'], moneyness)
        return np.maximum(vols, 0.01)
