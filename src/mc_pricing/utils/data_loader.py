"""Data loading utilities for market data."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path


class DataLoader:
    """Load and process market data for options pricing.

    Handles:
    - Market option prices
    - Volatility surfaces
    - Historical price data
    - Interest rate curves
    """

    @staticmethod
    def load_volatility_surface(
        filepath: str,
        delimiter: str = ','
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load volatility surface from CSV file.

        Expected format:
        - First row: strike prices
        - First column: maturities
        - Data: implied volatilities

        Args:
            filepath: Path to CSV file
            delimiter: CSV delimiter

        Returns:
            Tuple of (strikes, maturities, volatilities)
        """
        df = pd.read_csv(filepath, index_col=0, delimiter=delimiter)

        strikes = df.columns.values.astype(float)
        maturities = df.index.values.astype(float)
        volatilities = df.values.astype(float)

        return strikes, maturities, volatilities

    @staticmethod
    def load_option_chain(
        filepath: str,
        delimiter: str = ','
    ) -> pd.DataFrame:
        """Load option chain data.

        Expected columns:
        - strike: Strike price
        - maturity: Time to maturity
        - option_type: 'call' or 'put'
        - price: Market price
        - bid: Bid price (optional)
        - ask: Ask price (optional)
        - volume: Trading volume (optional)
        - open_interest: Open interest (optional)

        Args:
            filepath: Path to CSV file
            delimiter: CSV delimiter

        Returns:
            DataFrame with option chain data
        """
        df = pd.read_csv(filepath, delimiter=delimiter)

        # Validate required columns
        required = ['strike', 'maturity', 'option_type', 'price']
        missing = set(required) - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    @staticmethod
    def generate_sample_volatility_surface(
        spot: float = 100.0,
        strikes_pct: np.ndarray = np.linspace(0.8, 1.2, 11),
        maturities: np.ndarray = np.array([0.25, 0.5, 1.0, 2.0]),
        atm_vol: float = 0.2,
        smile_amplitude: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate sample volatility surface with smile.

        Args:
            spot: Current spot price
            strikes_pct: Strike prices as percentage of spot
            maturities: Array of maturities
            atm_vol: ATM volatility
            smile_amplitude: Amplitude of volatility smile

        Returns:
            Tuple of (strikes, maturities, volatilities)
        """
        strikes = strikes_pct * spot
        n_strikes = len(strikes)
        n_maturities = len(maturities)

        volatilities = np.zeros((n_maturities, n_strikes))

        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                # Simple quadratic smile
                moneyness = K / spot
                smile = smile_amplitude * (moneyness - 1.0) ** 2

                # Term structure: vol increases slightly with maturity
                term_adj = 0.02 * np.sqrt(T)

                volatilities[i, j] = atm_vol + smile + term_adj

        return strikes, maturities, volatilities

    @staticmethod
    def save_volatility_surface(
        filepath: str,
        strikes: np.ndarray,
        maturities: np.ndarray,
        volatilities: np.ndarray
    ):
        """Save volatility surface to CSV file.

        Args:
            filepath: Output file path
            strikes: Array of strikes
            maturities: Array of maturities
            volatilities: 2D array of volatilities
        """
        df = pd.DataFrame(
            volatilities,
            index=maturities,
            columns=strikes
        )
        df.index.name = 'Maturity'
        df.columns.name = 'Strike'

        df.to_csv(filepath)

    @staticmethod
    def generate_benchmark_dataset(
        output_dir: str,
        n_options: int = 100
    ) -> Dict[str, str]:
        """Generate benchmark dataset for testing.

        Args:
            output_dir: Directory to save files
            n_options: Number of options to generate

        Returns:
            Dictionary with paths to generated files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate sample option chain
        np.random.seed(42)

        spot = 100.0
        strikes = np.random.uniform(80, 120, n_options)
        maturities = np.random.uniform(0.1, 2.0, n_options)
        option_types = np.random.choice(['call', 'put'], n_options)

        # Simulate prices (simplified Black-Scholes approximation)
        from ..core.black_scholes import BlackScholes

        prices = []
        for strike, maturity, opt_type in zip(strikes, maturities, option_types):
            price = BlackScholes.price(
                option_type=opt_type,
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=0.05,
                volatility=0.2
            )
            # Add some noise
            price += np.random.normal(0, 0.1)
            prices.append(max(price, 0.01))

        option_chain = pd.DataFrame({
            'strike': strikes,
            'maturity': maturities,
            'option_type': option_types,
            'price': prices
        })

        chain_path = output_path / 'option_chain.csv'
        option_chain.to_csv(chain_path, index=False)

        # Generate sample volatility surface
        strikes, maturities, vols = DataLoader.generate_sample_volatility_surface()
        surface_path = output_path / 'vol_surface.csv'
        DataLoader.save_volatility_surface(surface_path, strikes, maturities, vols)

        return {
            'option_chain': str(chain_path),
            'volatility_surface': str(surface_path)
        }

    @staticmethod
    def load_historical_prices(
        filepath: str,
        date_column: str = 'date',
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """Load historical price data.

        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            price_column: Name of price column

        Returns:
            DataFrame with datetime index and prices
        """
        df = pd.read_csv(filepath)

        # Parse dates
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

        # Sort by date
        df = df.sort_index()

        return df

    @staticmethod
    def calculate_historical_volatility(
        prices: pd.Series,
        window: int = 252,
        annualization_factor: int = 252
    ) -> pd.Series:
        """Calculate rolling historical volatility.

        Args:
            prices: Series of prices
            window: Rolling window size
            annualization_factor: Factor to annualize volatility (252 for daily)

        Returns:
            Series of annualized volatilities
        """
        # Calculate returns
        returns = np.log(prices / prices.shift(1))

        # Rolling standard deviation
        vol = returns.rolling(window=window).std() * np.sqrt(annualization_factor)

        return vol
