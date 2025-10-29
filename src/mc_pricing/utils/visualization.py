"""Visualization utilities for options pricing analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
from ..core.base_option import BaseOption


class Visualizer:
    """Visualization tools for option pricing analysis.

    Provides plotting functions for:
    - Price convergence
    - Volatility smiles and surfaces
    - Greeks surfaces
    - Monte Carlo path visualization
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style not available
        sns.set_palette("husl")

    def plot_convergence(
        self,
        n_simulations: np.ndarray,
        prices: np.ndarray,
        std_errors: np.ndarray,
        true_price: Optional[float] = None,
        title: str = "Monte Carlo Convergence",
        save_path: Optional[str] = None
    ):
        """Plot price convergence as function of number of simulations.

        Args:
            n_simulations: Array of simulation counts
            prices: Array of estimated prices
            std_errors: Array of standard errors
            true_price: True price for comparison (e.g., Black-Scholes)
            title: Plot title
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Price convergence
        ax1.plot(n_simulations, prices, 'b-', linewidth=2, label='MC Estimate')
        ax1.fill_between(
            n_simulations,
            prices - 2 * std_errors,
            prices + 2 * std_errors,
            alpha=0.3,
            label='95% CI'
        )

        if true_price is not None:
            ax1.axhline(y=true_price, color='r', linestyle='--',
                       linewidth=2, label='True Price')

        ax1.set_xlabel('Number of Simulations')
        ax1.set_ylabel('Option Price')
        ax1.set_xscale('log')
        ax1.set_title('Price Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Standard error convergence (log-log plot)
        ax2.loglog(n_simulations, std_errors, 'b-', linewidth=2,
                  marker='o', label='Empirical')

        # Theoretical O(1/sqrt(n)) line
        theoretical = std_errors[0] * np.sqrt(n_simulations[0] / n_simulations)
        ax2.loglog(n_simulations, theoretical, 'r--', linewidth=2,
                  label='Theoretical O(1/√n)')

        ax2.set_xlabel('Number of Simulations')
        ax2.set_ylabel('Standard Error')
        ax2.set_title('Error Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_volatility_smile(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        spot: float,
        title: str = "Implied Volatility Smile",
        fitted_vols: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """Plot implied volatility smile.

        Args:
            strikes: Array of strike prices
            implied_vols: Array of implied volatilities
            spot: Current spot price
            title: Plot title
            fitted_vols: Optional fitted volatilities for comparison
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Convert to moneyness
        moneyness = strikes / spot

        # Plot market implied vols
        ax.plot(moneyness, implied_vols * 100, 'o', markersize=8,
               label='Market', color='blue')

        # Plot fitted curve if provided
        if fitted_vols is not None:
            ax.plot(moneyness, fitted_vols * 100, '-', linewidth=2,
                   label='Fitted', color='red')

        # Mark ATM
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5,
                  label='ATM')

        ax.set_xlabel('Moneyness (K/S)', fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_volatility_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        volatilities: np.ndarray,
        spot: float,
        title: str = "Implied Volatility Surface",
        save_path: Optional[str] = None
    ):
        """Plot 3D volatility surface.

        Args:
            strikes: Array of strike prices (1D)
            maturities: Array of maturities (1D)
            volatilities: 2D array of implied volatilities
            spot: Current spot price
            title: Plot title
            save_path: Path to save figure
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create mesh grid
        K, T = np.meshgrid(strikes, maturities)

        # Plot surface
        surf = ax.plot_surface(
            K / spot,  # Moneyness
            T,
            volatilities * 100,
            cmap='viridis',
            alpha=0.8,
            edgecolor='none'
        )

        ax.set_xlabel('Moneyness (K/S)', fontsize=10)
        ax.set_ylabel('Maturity (years)', fontsize=10)
        ax.set_zlabel('Implied Volatility (%)', fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_greeks_surface(
        self,
        spots: np.ndarray,
        maturities: np.ndarray,
        greek_values: np.ndarray,
        greek_name: str,
        strike: float,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """Plot Greek surface as function of spot and time.

        Args:
            spots: Array of spot prices
            maturities: Array of maturities
            greek_values: 2D array of Greek values
            greek_name: Name of the Greek (e.g., 'Delta', 'Gamma')
            strike: Strike price
            title: Plot title
            save_path: Path to save figure
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create mesh grid
        S, T = np.meshgrid(spots, maturities)

        # Plot surface
        surf = ax.plot_surface(
            S,
            T,
            greek_values,
            cmap='coolwarm',
            alpha=0.8,
            edgecolor='none'
        )

        ax.set_xlabel('Spot Price', fontsize=10)
        ax.set_ylabel('Time to Maturity', fontsize=10)
        ax.set_zlabel(greek_name, fontsize=10)

        if title is None:
            title = f"{greek_name} Surface (K={strike})"

        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_price_paths(
        self,
        paths: np.ndarray,
        n_paths_to_plot: int = 100,
        strike: Optional[float] = None,
        title: str = "Monte Carlo Price Paths",
        save_path: Optional[str] = None
    ):
        """Plot sample of Monte Carlo price paths.

        Args:
            paths: Array of price paths (n_paths, n_steps)
            n_paths_to_plot: Number of paths to display
            strike: Optional strike price to mark
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Select subset of paths
        n_paths = min(n_paths_to_plot, paths.shape[0])
        indices = np.random.choice(paths.shape[0], n_paths, replace=False)

        # Time axis
        time = np.linspace(0, 1, paths.shape[1])

        # Plot paths
        for idx in indices:
            ax.plot(time, paths[idx, :], alpha=0.3, linewidth=0.5)

        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        ax.plot(time, mean_path, 'r-', linewidth=2, label='Mean Path')

        # Mark strike if provided
        if strike is not None:
            ax.axhline(y=strike, color='green', linestyle='--',
                      linewidth=2, label=f'Strike ({strike})')

        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Asset Price', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def compare_variance_reduction(
        self,
        methods: List[str],
        prices: List[float],
        std_errors: List[float],
        true_price: Optional[float] = None,
        title: str = "Variance Reduction Comparison",
        save_path: Optional[str] = None
    ):
        """Compare different variance reduction techniques.

        Args:
            methods: List of method names
            prices: List of estimated prices
            std_errors: List of standard errors
            true_price: True price for reference
            title: Plot title
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(methods))

        # Price comparison
        bars1 = ax1.bar(x, prices, color='steelblue', alpha=0.8)
        ax1.errorbar(x, prices, yerr=2 * np.array(std_errors),
                    fmt='none', color='black', capsize=5)

        if true_price is not None:
            ax1.axhline(y=true_price, color='r', linestyle='--',
                       linewidth=2, label='True Price')
            ax1.legend()

        ax1.set_xlabel('Method')
        ax1.set_ylabel('Price Estimate')
        ax1.set_title('Price Estimates with 95% CI')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        # Standard error comparison
        bars2 = ax2.bar(x, std_errors, color='coral', alpha=0.8)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Standard Error')
        ax2.set_title('Standard Errors')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Show variance reduction percentage
        if len(std_errors) > 0:
            baseline_var = std_errors[0] ** 2
            for i, (method, se) in enumerate(zip(methods, std_errors)):
                var_reduction = (1 - (se**2 / baseline_var)) * 100
                if var_reduction > 0:
                    ax2.text(i, se, f'{var_reduction:.1f}%',
                           ha='center', va='bottom', fontsize=8)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
