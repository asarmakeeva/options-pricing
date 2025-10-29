"""Convergence analysis tools for Monte Carlo simulation."""

import numpy as np
from typing import List, Tuple, Optional
from ..core.base_option import BaseOption
from .simulator import MonteCarloSimulator


class ConvergenceAnalyzer:
    """Analyze convergence properties of Monte Carlo estimators.

    This class provides tools to study the convergence rate of MC simulations
    and compare variance reduction techniques.
    """

    @staticmethod
    def analyze_convergence(
        option: BaseOption,
        n_trials: np.ndarray,
        n_runs: int = 10,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analyze convergence for different sample sizes.

        Args:
            option: Option to price
            n_trials: Array of sample sizes to test
            n_runs: Number of independent runs for each sample size
            seed: Random seed for reproducibility

        Returns:
            Tuple of (mean_prices, std_prices, std_errors)
            Each is an array of length len(n_trials)
        """
        mean_prices = []
        std_prices = []
        std_errors = []

        for n in n_trials:
            prices_for_n = []

            for run in range(n_runs):
                sim_seed = None if seed is None else seed + run
                simulator = MonteCarloSimulator(n_simulations=n, seed=sim_seed)
                price, std_error = simulator.price(option)
                prices_for_n.append(price)

            mean_prices.append(np.mean(prices_for_n))
            std_prices.append(np.std(prices_for_n))
            # Average standard error across runs
            std_errors.append(std_error)

        return (
            np.array(mean_prices),
            np.array(std_prices),
            np.array(std_errors)
        )

    @staticmethod
    def compare_variance_reduction(
        option: BaseOption,
        variance_reduction_methods: List,
        method_names: List[str],
        n_simulations: int = 100000,
        n_runs: int = 20,
        seed: Optional[int] = None
    ) -> dict:
        """Compare different variance reduction techniques.

        Args:
            option: Option to price
            variance_reduction_methods: List of variance reduction objects
            method_names: Names of the methods
            n_simulations: Number of simulations per run
            n_runs: Number of independent runs
            seed: Random seed

        Returns:
            Dictionary with comparison metrics
        """
        results = {}

        for method, name in zip(variance_reduction_methods, method_names):
            prices = []
            errors = []

            for run in range(n_runs):
                sim_seed = None if seed is None else seed + run
                simulator = MonteCarloSimulator(n_simulations=n_simulations, seed=sim_seed)
                price, std_error = simulator.price(option, variance_reduction=method)
                prices.append(price)
                errors.append(std_error)

            results[name] = {
                'mean_price': np.mean(prices),
                'std_price': np.std(prices),
                'mean_error': np.mean(errors),
                'variance': np.var(prices),
            }

        # Calculate variance reduction ratios relative to baseline
        if 'Standard MC' in results:
            baseline_var = results['Standard MC']['variance']
            for name in results:
                if name != 'Standard MC':
                    var_reduction = 1 - (results[name]['variance'] / baseline_var)
                    results[name]['variance_reduction_pct'] = var_reduction * 100

        return results

    @staticmethod
    def estimate_required_simulations(
        target_error: float,
        estimated_std: float,
        confidence: float = 0.95
    ) -> int:
        """Estimate number of simulations needed for target standard error.

        Uses the relationship: SE = std / sqrt(n)

        Args:
            target_error: Desired standard error
            estimated_std: Estimated standard deviation of payoffs
            confidence: Confidence level

        Returns:
            Estimated number of simulations needed
        """
        from scipy.stats import norm

        z_score = norm.ppf((1 + confidence) / 2)
        # SE = std / sqrt(n), so n = (std / SE)^2
        # For confidence interval: margin = z * SE, so SE = margin / z
        target_se = target_error / z_score
        n_required = int(np.ceil((estimated_std / target_se) ** 2))

        return n_required

    @staticmethod
    def theoretical_convergence_rate(n_trials: np.ndarray) -> np.ndarray:
        """Calculate theoretical O(1/sqrt(n)) convergence rate.

        Args:
            n_trials: Array of sample sizes

        Returns:
            Array of theoretical error scaling (proportional to 1/sqrt(n))
        """
        # Normalize to first element
        return 1.0 / np.sqrt(n_trials / n_trials[0])
