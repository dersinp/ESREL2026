"""
Integrated Preventive Maintenance Scheduler using Time Transformation
======================================================================
Combines:
1. Health Index (HI) trajectory simulation
2. Threshold crossing detection
3. Time transformation-based RUL prediction
4. Optimal next maintenance time calculation

Based on the paper: "Analysis of RUL dynamics and uncertainty via time transformation"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import gamma, weibull_min, lognorm
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.stats import gaussian_kde
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import sys
from pathlib import Path

# Import existing modules
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class MaintenanceResult:
    """Container for maintenance scheduling results for a single trajectory."""
    trajectory_id: int
    hi_trajectory: np.ndarray
    time_grid: np.ndarray
    threshold_crossing_time: float  # t*
    next_maintenance_time: float    # t* + s*
    inspection_interval: float      # s*
    rul_at_crossing: float          # Expected RUL at t*
    crossed_threshold: bool


class PreventiveMaintenanceScheduler:
    """
    Scheduler that determines optimal preventive maintenance times based on
    HI threshold crossings and time transformation-based RUL prediction.

    Attributes:
    -----------
    tau : float
        Health index threshold for triggering maintenance planning
    alpha : float
        Probability constraint: P(RUL(t*) < s*) ≤ alpha
    mu : float
        Mean time to failure
    k : float
        Degradation slope parameter
    """

    def __init__(self, tau: float, alpha: float = 0.05):
        """
        Initialize the preventive maintenance scheduler.

        Parameters:
        -----------
        tau : float
            HI threshold value (between 0 and 1)
        alpha : float, optional
            Probability level for maintenance scheduling (default: 0.05)
        """
        if not 0 < tau < 1:
            raise ValueError("Threshold tau must be between 0 and 1")
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")

        self.tau = tau
        self.alpha = alpha
        self.mu = None
        self.k = None
        self.cv = None
        self.ttf_data = None
        self._reliability = None
        self._kde = None
        self.t_grid = None
        self.g_vals = None
        self.g_fun = None
        self.g_inv = None

    def fit_from_ttf_data(self, ttf_data: np.ndarray):
        """
        Fit the time transformation model from time-to-failure data.

        Parameters:
        -----------
        ttf_data : np.ndarray
            Array of time-to-failure observations
        """
        # Clean data
        ttf_data = np.asarray(ttf_data)
        ttf_data = ttf_data[(ttf_data > 0) & (~np.isnan(ttf_data))]
        self.ttf_data = ttf_data

        # Compute statistics
        self.mu = np.mean(ttf_data)
        std = np.std(ttf_data)
        self.cv = std / self.mu
        self.k = np.clip((1 - self.cv ** 2) / (1 + self.cv ** 2), 1e-6, 1 - 1e-6)

        # Build reliability function using KDE
        self._kde = gaussian_kde(ttf_data)
        self.t_grid = np.linspace(0, np.max(ttf_data), 5000)
        pdf_on_grid = self._kde(self.t_grid)
        pdf_on_grid[pdf_on_grid < 1e-6] = 0
        kde_cdf = cumtrapz(pdf_on_grid, self.t_grid, initial=0)
        self._reliability = 1 - kde_cdf

        # Compute time transformation g(t) and its inverse
        exponent = self.k / (1 - self.k)
        self.g_vals = (self.mu / self.k) * (1 - np.power(self._reliability, exponent))

        # Create interpolation functions for g(t) and g_inv
        self.g_fun = interp1d(self.t_grid, self.g_vals,
                             bounds_error=False, fill_value="extrapolate")
        self.g_inv = interp1d(self.g_vals, self.t_grid,
                             bounds_error=False, fill_value="extrapolate")

        print(f"Model fitted: μ={self.mu:.4f}, k={self.k:.4f}, CV={self.cv:.4f}")

    def compute_inspection_time(self, t_star: float) -> float:
        """
        Compute next inspection time s* at threshold crossing time t*.

        Correct formula using time transformation:
        g(t* + s*) = g(t*) + (μ/k - g(t*)) × [1 - (1-α)^(k/(1-k))]
        Then: s* = g_inv(g(t* + s*)) - t*

        Parameters:
        -----------
        t_star : float
            Time of threshold crossing (t*)

        Returns:
        --------
        s_star : float
            Optimal inspection interval
        """
        if self.mu is None or self.k is None:
            raise ValueError("Model must be fitted first using fit_from_ttf_data()")

        # Transform t* to transformed time
        g_t_star = float(self.g_fun(t_star))

        # Compute inspection time in transformed space
        factor = self.mu / self.k - g_t_star
        exponent = self.k / (1 - self.k)
        g_inspection = g_t_star + factor * (1 - (1 - self.alpha) ** exponent)

        # Map back to physical time and get interval
        t_inspection = float(self.g_inv(g_inspection))
        s_star = t_inspection - t_star

        return np.maximum(s_star, 0.0)

    def find_threshold_crossing(self, hi_trajectory: np.ndarray,
                               time_grid: np.ndarray) -> Tuple[bool, float]:
        """
        Find the first time when HI crosses the threshold tau.

        Parameters:
        -----------
        hi_trajectory : np.ndarray
            Health index trajectory
        time_grid : np.ndarray
            Time points corresponding to HI values

        Returns:
        --------
        crossed : bool
            Whether threshold was crossed
        t_star : float
            Time of first crossing (NaN if not crossed)
        """
        # Find where HI drops below threshold
        below_threshold = hi_trajectory <= self.tau

        if not np.any(below_threshold):
            return False, np.nan

        # Find first crossing
        crossing_idx = np.where(below_threshold)[0][0]

        # Interpolate for more precise crossing time if not at exact grid point
        if crossing_idx > 0:
            t1, t2 = time_grid[crossing_idx-1], time_grid[crossing_idx]
            hi1, hi2 = hi_trajectory[crossing_idx-1], hi_trajectory[crossing_idx]

            # Linear interpolation to find exact crossing point
            if hi1 != hi2:
                t_star = t1 + (self.tau - hi1) * (t2 - t1) / (hi2 - hi1)
            else:
                t_star = t1
        else:
            t_star = time_grid[crossing_idx]

        return True, t_star

    def process_single_trajectory(self, hi_trajectory: np.ndarray,
                                  time_grid: np.ndarray,
                                  trajectory_id: int = 0) -> MaintenanceResult:
        """
        Process a single HI trajectory and determine maintenance time.

        Parameters:
        -----------
        hi_trajectory : np.ndarray
            Health index values over time
        time_grid : np.ndarray
            Time points for HI trajectory
        trajectory_id : int
            Identifier for this trajectory

        Returns:
        --------
        MaintenanceResult
            Complete results for this trajectory
        """
        # Find threshold crossing
        crossed, t_star = self.find_threshold_crossing(hi_trajectory, time_grid)

        if not crossed:
            return MaintenanceResult(
                trajectory_id=trajectory_id,
                hi_trajectory=hi_trajectory,
                time_grid=time_grid,
                threshold_crossing_time=np.nan,
                next_maintenance_time=np.nan,
                inspection_interval=np.nan,
                rul_at_crossing=np.nan,
                crossed_threshold=False
            )

        # Compute optimal inspection interval
        s_star = self.compute_inspection_time(t_star)
        next_maintenance = t_star + s_star

        # Compute expected RUL at crossing time
        rul_mean = self._compute_mean_rul(t_star)

        return MaintenanceResult(
            trajectory_id=trajectory_id,
            hi_trajectory=hi_trajectory,
            time_grid=time_grid,
            threshold_crossing_time=t_star,
            next_maintenance_time=next_maintenance,
            inspection_interval=s_star,
            rul_at_crossing=rul_mean,
            crossed_threshold=True
        )

    def _compute_mean_rul(self, t: float) -> float:
        """Compute mean residual life at time t."""
        if t >= self.t_grid[-1]:
            return 0.0

        # Interpolate reliability at time t
        R_t = np.interp(t, self.t_grid, self._reliability)

        if R_t < 1e-6:
            return 0.0

        # Compute integral of R from t to infinity
        idx = np.searchsorted(self.t_grid, t)
        if idx >= len(self.t_grid):
            return 0.0

        integral = np.trapz(self._reliability[idx:], self.t_grid[idx:])
        mrl = integral / R_t

        return mrl

    def process_multiple_trajectories(self,
                                     hi_trajectories: List[np.ndarray],
                                     time_grids: List[np.ndarray]) -> List[MaintenanceResult]:
        """
        Process multiple HI trajectories.

        Parameters:
        -----------
        hi_trajectories : List[np.ndarray]
            List of HI trajectories
        time_grids : List[np.ndarray]
            List of corresponding time grids

        Returns:
        --------
        List[MaintenanceResult]
            Results for all trajectories
        """
        if len(hi_trajectories) != len(time_grids):
            raise ValueError("Number of trajectories must match number of time grids")

        results = []
        for i, (hi_traj, time_grid) in enumerate(zip(hi_trajectories, time_grids)):
            result = self.process_single_trajectory(hi_traj, time_grid, trajectory_id=i)
            results.append(result)

        return results

    def summarize_results(self, results: List[MaintenanceResult]) -> pd.DataFrame:
        """
        Create a summary DataFrame of all results.

        Parameters:
        -----------
        results : List[MaintenanceResult]
            List of maintenance results

        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        data = {
            'Trajectory_ID': [],
            'Crossed_Threshold': [],
            'Threshold_Crossing_Time_t_star': [],
            'Inspection_Interval_s_star': [],
            'Next_Maintenance_Time': [],
            'Expected_RUL_at_t_star': []
        }

        for result in results:
            data['Trajectory_ID'].append(result.trajectory_id)
            data['Crossed_Threshold'].append(result.crossed_threshold)
            data['Threshold_Crossing_Time_t_star'].append(
                result.threshold_crossing_time if result.crossed_threshold else np.nan
            )
            data['Inspection_Interval_s_star'].append(
                result.inspection_interval if result.crossed_threshold else np.nan
            )
            data['Next_Maintenance_Time'].append(
                result.next_maintenance_time if result.crossed_threshold else np.nan
            )
            data['Expected_RUL_at_t_star'].append(
                result.rul_at_crossing if result.crossed_threshold else np.nan
            )

        df = pd.DataFrame(data)
        return df


# ============================================================================
# HI TRAJECTORY GENERATION FUNCTIONS
# ============================================================================

def generate_frechet(shape: float, scale: float, size: int) -> np.ndarray:
    """Generate random samples from Fréchet distribution."""
    U = np.random.uniform(0, 1, size)
    samples = scale * ((-np.log(U)) ** (-1/shape))
    return samples


def generate_hi_trajectories_parametric(n_trajectories: int,
                                       dist_type: str,
                                       params: dict,
                                       p: float,
                                       time_points: int = 200,
                                       t_max: Optional[float] = None) -> Tuple[List[np.ndarray],
                                                                                 List[np.ndarray],
                                                                                 np.ndarray]:
    """
    Generate HI trajectories using parametric degradation model.

    HI(t) = 1 - b*t^p  (for most distributions)
    or
    HI(t) = 1 - (b*t)^p (for Gamma)

    Parameters:
    -----------
    n_trajectories : int
        Number of trajectories to generate
    dist_type : str
        Distribution type: 'frechet', 'gamma', 'weibull', 'lognormal'
    params : dict
        Distribution parameters
    p : float
        Power parameter
    time_points : int
        Number of time points in each trajectory
    t_max : float, optional
        Maximum time. If None, computed from distribution

    Returns:
    --------
    hi_trajectories : List[np.ndarray]
        List of HI trajectories
    time_grids : List[np.ndarray]
        List of time grids
    ttf_samples : np.ndarray
        Time-to-failure samples
    """
    # Generate b samples
    if dist_type.lower() == 'frechet':
        b_samples = generate_frechet(params['shape'], params['scale'], n_trajectories)
    elif dist_type.lower() == 'gamma':
        b_samples = gamma.rvs(a=params['shape'], scale=params['scale'], size=n_trajectories)
    elif dist_type.lower() == 'weibull':
        b_samples = weibull_min.rvs(c=params['shape'], scale=params['scale'], size=n_trajectories)
    elif dist_type.lower() == 'lognormal':
        b_samples = lognorm.rvs(s=params['std'], scale=np.exp(params['mean']), size=n_trajectories)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")

    # Compute TTF samples
    if dist_type.lower() == 'gamma':
        ttf_samples = 1 / b_samples  # For Gamma: HI(t) = 1 - (b*t)^p
    else:
        ttf_samples = (1 / b_samples) ** (1/p)  # For others: HI(t) = 1 - b*t^p

    # Determine time grid
    if t_max is None:
        t_max = np.percentile(ttf_samples, 95) * 1.2

    hi_trajectories = []
    time_grids = []

    for b in b_samples:
        time_grid = np.linspace(0, t_max, time_points)

        if dist_type.lower() == 'gamma':
            hi_trajectory = 1 - (b * time_grid) ** p
        else:
            hi_trajectory = 1 - b * (time_grid ** p)

        # Ensure HI stays in [0, 1]
        hi_trajectory = np.clip(hi_trajectory, 0, 1)

        hi_trajectories.append(hi_trajectory)
        time_grids.append(time_grid)

    return hi_trajectories, time_grids, ttf_samples


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_maintenance_results(results: List[MaintenanceResult],
                                  tau: float,
                                  max_trajectories: int = 20,
                                  save_path: str = 'maintenance_results.png'):
    """
    Visualize maintenance scheduling results.

    Parameters:
    -----------
    results : List[MaintenanceResult]
        List of maintenance results
    tau : float
        Threshold value
    max_trajectories : int
        Maximum number of trajectories to plot
    save_path : str
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Select results to plot
    results_to_plot = results[:max_trajectories]

    # 1. HI Trajectories with threshold and maintenance times
    ax = axes[0, 0]
    for result in results_to_plot:
        ax.plot(result.time_grid, result.hi_trajectory, alpha=0.5, linewidth=1)

        if result.crossed_threshold:
            # Mark threshold crossing
            ax.plot(result.threshold_crossing_time, tau, 'ro', markersize=6)
            # Mark next maintenance time
            ax.axvline(result.next_maintenance_time, color='red',
                      linestyle='--', alpha=0.3, linewidth=1)

    ax.axhline(tau, color='red', linestyle='--', linewidth=2,
              label=f'Threshold τ={tau}')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Health Index', fontsize=12)
    ax.set_title('HI Trajectories with Maintenance Planning', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Distribution of threshold crossing times
    ax = axes[0, 1]
    crossing_times = [r.threshold_crossing_time for r in results if r.crossed_threshold]
    if crossing_times:
        ax.hist(crossing_times, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(crossing_times), color='red', linestyle='--',
                  linewidth=2, label=f'Mean = {np.mean(crossing_times):.2f}')
        ax.set_xlabel('Threshold Crossing Time (t*)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Threshold Crossing Times', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Distribution of inspection intervals
    ax = axes[1, 0]
    inspection_intervals = [r.inspection_interval for r in results if r.crossed_threshold]
    if inspection_intervals:
        ax.hist(inspection_intervals, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(inspection_intervals), color='red', linestyle='--',
                  linewidth=2, label=f'Mean = {np.mean(inspection_intervals):.2f}')
        ax.set_xlabel('Inspection Interval (s*)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Optimal Inspection Intervals', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Relationship: t* vs s*
    ax = axes[1, 1]
    if crossing_times and inspection_intervals:
        ax.scatter(crossing_times, inspection_intervals, alpha=0.6, s=50)
        ax.set_xlabel('Threshold Crossing Time (t*)', fontsize=12)
        ax.set_ylabel('Inspection Interval (s*)', fontsize=12)
        ax.set_title('Relationship: Crossing Time vs Inspection Interval',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(crossing_times) > 1:
            z = np.polyfit(crossing_times, inspection_intervals, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(crossing_times), max(crossing_times), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax.legend()

    plt.suptitle('Preventive Maintenance Scheduling Results',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to '{save_path}'")
    except Exception as e:
        print(f"⚠ Could not save figure: {e}")

    return fig


# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def main_demo():
    """
    Demonstration of the preventive maintenance scheduler.
    """
    print("\n" + "="*70)
    print(" "*10 + "PREVENTIVE MAINTENANCE SCHEDULER")
    print(" "*15 + "Using Time Transformation")
    print("="*70)

    # Configuration
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)

    n_trajectories = int(input("Number of HI trajectories to generate [50]: ") or "50")
    tau = float(input("Health index threshold τ [0.7]: ") or "0.7")
    alpha = float(input("Probability level α [0.05]: ") or "0.05")

    print(f"\n  • Generating {n_trajectories} trajectories")
    print(f"  • Threshold τ = {tau}")
    print(f"  • Probability level α = {alpha}")
    print(f"  • Constraint: P(RUL(t*) < s*) ≤ {alpha}")

    # Distribution selection
    print("\n" + "="*70)
    print("DISTRIBUTION SELECTION")
    print("="*70)
    print("\nSelect TTF distribution for b parameter:")
    print("  1. Fréchet [default]")
    print("  2. Weibull")
    print("  3. Gamma")
    print("  4. Lognormal")

    choice = input("\nChoice [1]: ") or "1"

    if choice == '2':
        dist_type = 'weibull'
        shape = float(input("Weibull shape parameter [2.5]: ") or "2.5")
        scale = float(input("Weibull scale parameter [0.015]: ") or "0.015")
        params = {'shape': shape, 'scale': scale}
    elif choice == '3':
        dist_type = 'gamma'
        shape = float(input("Gamma shape parameter [5.0]: ") or "5.0")
        scale = float(input("Gamma scale parameter [0.003]: ") or "0.003")
        params = {'shape': shape, 'scale': scale}
    elif choice == '4':
        dist_type = 'lognormal'
        mean = float(input("Lognormal mean (log-scale) [-4.0]: ") or "-4.0")
        std = float(input("Lognormal std (log-scale) [0.5]: ") or "0.5")
        params = {'mean': mean, 'std': std}
    else:
        dist_type = 'frechet'
        shape = float(input("Fréchet shape parameter [3.0]: ") or "3.0")
        scale = float(input("Fréchet scale parameter [0.01]: ") or "0.01")
        params = {'shape': shape, 'scale': scale}

    p = float(input("\nPower parameter p for HI(t)=1-b*t^p [3.0]: ") or "3.0")

    # Generate HI trajectories
    print("\n" + "="*70)
    print("GENERATING HI TRAJECTORIES")
    print("="*70)
    print(f"Using {dist_type.upper()} distribution for b parameter...")
    print(f"Parameters: {params}")
    print(f"Power: p = {p}")

    hi_trajectories, time_grids, ttf_samples = generate_hi_trajectories_parametric(
        n_trajectories=n_trajectories,
        dist_type=dist_type,
        params=params,
        p=p,
        time_points=300
    )

    print(f"✓ Generated {len(hi_trajectories)} HI trajectories")
    print(f"  • TTF mean: {np.mean(ttf_samples):.2f}")
    print(f"  • TTF std: {np.std(ttf_samples):.2f}")

    # Initialize scheduler and fit model
    print("\n" + "="*70)
    print("FITTING TIME TRANSFORMATION MODEL")
    print("="*70)

    scheduler = PreventiveMaintenanceScheduler(tau=tau, alpha=alpha)
    scheduler.fit_from_ttf_data(ttf_samples)

    # Process all trajectories
    print("\n" + "="*70)
    print("PROCESSING TRAJECTORIES")
    print("="*70)

    results = scheduler.process_multiple_trajectories(hi_trajectories, time_grids)

    # Summarize results
    n_crossed = sum(1 for r in results if r.crossed_threshold)
    print(f"\n✓ Processed {len(results)} trajectories")
    print(f"  • {n_crossed} crossed threshold ({n_crossed/len(results)*100:.1f}%)")

    if n_crossed > 0:
        crossing_times = [r.threshold_crossing_time for r in results if r.crossed_threshold]
        intervals = [r.inspection_interval for r in results if r.crossed_threshold]

        print(f"\n  Threshold Crossing Times (t*):")
        print(f"    Mean: {np.mean(crossing_times):.4f}")
        print(f"    Std:  {np.std(crossing_times):.4f}")
        print(f"    Range: [{np.min(crossing_times):.4f}, {np.max(crossing_times):.4f}]")

        print(f"\n  Inspection Intervals (s*):")
        print(f"    Mean: {np.mean(intervals):.4f}")
        print(f"    Std:  {np.std(intervals):.4f}")
        print(f"    Range: [{np.min(intervals):.4f}, {np.max(intervals):.4f}]")

    # Create summary DataFrame and save
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    summary_df = scheduler.summarize_results(results)
    summary_df.to_csv('maintenance_schedule.csv', index=False)
    print("✓ Saved summary to 'maintenance_schedule.csv'")

    # Display first few rows
    print("\nFirst 10 results:")
    print(summary_df.head(10).to_string(index=False))

    # Visualize
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    fig = visualize_maintenance_results(results, tau, max_trajectories=30)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • maintenance_schedule.csv - Detailed results")
    print("  • maintenance_results.png - Visualizations")
    print("="*70 + "\n")

    plt.show()

    return scheduler, results, summary_df


if __name__ == "__main__":
    try:
        scheduler, results, summary = main_demo()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n⚠ Error: {str(e)}")
        import traceback
        traceback.print_exc()
