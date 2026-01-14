"""
Extended HI Simulation with Preventive Maintenance Scheduling
==============================================================
Extends the generalized_hi_simulation.py to include:
- Threshold-based maintenance triggering
- Optimal inspection interval calculation using time transformation
- Integration with existing parametric and non-parametric HI generation

Usage:
    python hi_simulation_with_maintenance.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import gamma, weibull_min, lognorm, gaussian_kde
from scipy.integrate import cumulative_trapezoid as cumtrapz
from typing import Tuple, List, Optional
from datetime import datetime
import sys
from pathlib import Path


# ============================================================================
# TIME TRANSFORMATION & MAINTENANCE SCHEDULER
# ============================================================================

class MaintenanceScheduler:
    """
    Computes optimal preventive maintenance times using time transformation.

    Given:
    - Threshold τ: when HI drops below this, start planning maintenance
    - Probability α: constraint P(RUL(t*) < s*) ≤ α

    At threshold crossing time t*, computes next inspection time s* such that
    the probability of failure before inspection is at most α.
    """

    def __init__(self, ttf_data: np.ndarray, tau: float, alpha: float):
        """
        Parameters:
        -----------
        ttf_data : np.ndarray
            Time-to-failure observations for model fitting
        tau : float
            HI threshold for triggering maintenance (0 < tau < 1)
        alpha : float
            Probability level (0 < alpha < 1)
        """
        self.tau = tau
        self.alpha = alpha

        # Fit time transformation model
        self._fit_model(ttf_data)

    def _fit_model(self, ttf_data: np.ndarray):
        """Fit time transformation model from TTF data."""
        # Clean data
        ttf_data = np.asarray(ttf_data)
        ttf_data = ttf_data[(ttf_data > 0) & (~np.isnan(ttf_data))]

        # Basic statistics
        self.mu = np.mean(ttf_data)
        std = np.std(ttf_data)
        self.cv = std / self.mu
        self.k = np.clip((1 - self.cv ** 2) / (1 + self.cv ** 2), 1e-6, 1 - 1e-6)

        # Build reliability function
        kde = gaussian_kde(ttf_data)
        t_grid = np.linspace(0, np.max(ttf_data), 5000)
        pdf = kde(t_grid)
        pdf[pdf < 1e-6] = 0
        cdf = cumtrapz(pdf, t_grid, initial=0)
        reliability = 1 - cdf

        # Compute g(t) transformation and its inverse
        exponent = self.k / (1 - self.k)
        g_vals = (self.mu / self.k) * (1 - np.power(reliability, exponent))

        # Store for computations
        self.t_grid = t_grid
        self.reliability = reliability
        self.g_vals = g_vals

        # Create interpolation functions for g(t) and g_inv
        self.g_fun = interp1d(t_grid, g_vals, bounds_error=False, fill_value="extrapolate")
        self.g_inv = interp1d(g_vals, t_grid, bounds_error=False, fill_value="extrapolate")

        print(f"  Model: μ={self.mu:.4f}, k={self.k:.4f}, CV={self.cv:.4f}")

    def compute_next_inspection_time(self, t_star: float) -> Tuple[float, float]:
        """
        Compute optimal inspection interval s* at time t*.

        Correct formula using time transformation:
        g(t* + s*) = g(t*) + (μ/k - g(t*)) × [1 - (1-α)^(k/(1-k))]
        Then: s* = g_inv(g(t* + s*)) - t*

        Parameters:
        -----------
        t_star : float
            Threshold crossing time

        Returns:
        --------
        s_star : float
            Optimal inspection interval
        rul_mean : float
            Expected RUL at t*
        """
        # Transform t* to transformed time
        g_t_star = float(self.g_fun(t_star))

        # Compute inspection time in transformed space
        factor = self.mu / self.k - g_t_star
        exponent = self.k / (1 - self.k)
        g_inspection = g_t_star + factor * (1 - (1 - self.alpha) ** exponent)

        # Map back to physical time and get interval
        t_inspection = float(self.g_inv(g_inspection))
        s_star = t_inspection - t_star
        s_star = np.maximum(s_star, 0.0)

        # Compute mean RUL at t*
        rul_mean = self._compute_mean_rul(t_star)

        return s_star, rul_mean

    def _compute_mean_rul(self, t: float) -> float:
        """Compute mean residual life at time t."""
        if t >= self.t_grid[-1]:
            return 0.0

        R_t = np.interp(t, self.t_grid, self.reliability)
        if R_t < 1e-6:
            return 0.0

        idx = np.searchsorted(self.t_grid, t)
        if idx >= len(self.t_grid):
            return 0.0

        integral = np.trapz(self.reliability[idx:], self.t_grid[idx:])
        return integral / R_t

    def analyze_trajectory(self, hi_traj: np.ndarray, time_grid: np.ndarray) -> dict:
        """
        Analyze single HI trajectory for maintenance scheduling.

        Returns dictionary with:
        - crossed: bool
        - t_star: threshold crossing time
        - s_star: inspection interval
        - next_maintenance: t* + s*
        - rul_at_crossing: expected RUL at t*
        """
        # Find threshold crossing
        below_threshold = hi_traj <= self.tau

        if not np.any(below_threshold):
            return {
                'crossed': False,
                't_star': np.nan,
                's_star': np.nan,
                'next_maintenance': np.nan,
                'rul_at_crossing': np.nan
            }

        # Find first crossing with interpolation
        crossing_idx = np.where(below_threshold)[0][0]

        if crossing_idx > 0:
            t1, t2 = time_grid[crossing_idx-1], time_grid[crossing_idx]
            hi1, hi2 = hi_traj[crossing_idx-1], hi_traj[crossing_idx]

            if hi1 != hi2:
                t_star = t1 + (self.tau - hi1) * (t2 - t1) / (hi2 - hi1)
            else:
                t_star = t1
        else:
            t_star = time_grid[crossing_idx]

        # Compute maintenance schedule
        s_star, rul_mean = self.compute_next_inspection_time(t_star)

        return {
            'crossed': True,
            't_star': t_star,
            's_star': s_star,
            'next_maintenance': t_star + s_star,
            'rul_at_crossing': rul_mean
        }


# ============================================================================
# HI TRAJECTORY GENERATION (compatible with existing code)
# ============================================================================

def generate_frechet(shape: float, scale: float, size: int) -> np.ndarray:
    """Generate random samples from Fréchet distribution."""
    U = np.random.uniform(0, 1, size)
    return scale * ((-np.log(U)) ** (-1/shape))


def generate_b_from_distribution(dist_type: str, params: dict, size: int) -> np.ndarray:
    """Generate b samples from various distributions."""
    if dist_type.lower() == 'frechet':
        return generate_frechet(params['shape'], params['scale'], size)
    elif dist_type.lower() == 'gamma':
        return gamma.rvs(a=params['shape'], scale=params['scale'], size=size)
    elif dist_type.lower() == 'weibull':
        return weibull_min.rvs(c=params['shape'], scale=params['scale'], size=size)
    elif dist_type.lower() == 'lognormal':
        return lognorm.rvs(s=params['std'], scale=np.exp(params['mean']), size=size)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")


def simulate_hi_with_maintenance(n_assets: int,
                                dist_type: str,
                                params: dict,
                                p: float,
                                tau: float,
                                alpha: float,
                                time_points: int = 300,
                                t_max: Optional[float] = None) -> dict:
    """
    Simulate HI trajectories and compute maintenance schedules.

    Parameters:
    -----------
    n_assets : int
        Number of assets to simulate
    dist_type : str
        Distribution for b parameter
    params : dict
        Distribution parameters
    p : float
        Power parameter for HI model
    tau : float
        HI threshold for maintenance
    alpha : float
        Probability constraint
    time_points : int
        Number of time points per trajectory
    t_max : float, optional
        Maximum simulation time

    Returns:
    --------
    dict with:
        - hi_trajectories: List of HI trajectories
        - time_grids: List of time grids
        - ttf_samples: TTF values
        - maintenance_results: List of maintenance analysis results
        - scheduler: MaintenanceScheduler object
    """
    # Generate b samples
    b_samples = generate_b_from_distribution(dist_type, params, n_assets)

    # Compute TTF
    if dist_type.lower() == 'gamma':
        ttf_samples = 1 / b_samples
    else:
        ttf_samples = (1 / b_samples) ** (1/p)

    # Determine time grid
    if t_max is None:
        t_max = np.percentile(ttf_samples, 95) * 1.2

    # Generate HI trajectories
    hi_trajectories = []
    time_grids = []

    for b in b_samples:
        time_grid = np.linspace(0, t_max, time_points)

        if dist_type.lower() == 'gamma':
            hi_traj = 1 - (b * time_grid) ** p
        else:
            hi_traj = 1 - b * (time_grid ** p)

        hi_traj = np.clip(hi_traj, 0, 1)
        hi_trajectories.append(hi_traj)
        time_grids.append(time_grid)

    # Initialize maintenance scheduler
    scheduler = MaintenanceScheduler(ttf_samples, tau, alpha)

    # Analyze all trajectories
    maintenance_results = []
    for hi_traj, time_grid in zip(hi_trajectories, time_grids):
        result = scheduler.analyze_trajectory(hi_traj, time_grid)
        maintenance_results.append(result)

    return {
        'hi_trajectories': hi_trajectories,
        'time_grids': time_grids,
        'ttf_samples': ttf_samples,
        'b_samples': b_samples,
        'maintenance_results': maintenance_results,
        'scheduler': scheduler
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_visualization(results: dict,
                                      dist_type: str,
                                      save_prefix: str = 'maintenance_analysis'):
    """Create comprehensive visualization of maintenance analysis."""

    hi_trajs = results['hi_trajectories']
    time_grids = results['time_grids']
    maint_results = results['maintenance_results']
    ttf = results['ttf_samples']
    tau = results['scheduler'].tau
    alpha = results['scheduler'].alpha

    fig = plt.figure(figsize=(18, 12))

    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. HI Trajectories (large plot, top left 2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    n_plot = min(30, len(hi_trajs))

    for i in range(n_plot):
        ax1.plot(time_grids[i], hi_trajs[i], alpha=0.4, linewidth=1, color='steelblue')

        if maint_results[i]['crossed']:
            t_star = maint_results[i]['t_star']
            t_maint = maint_results[i]['next_maintenance']

            # Mark threshold crossing
            ax1.plot(t_star, tau, 'ro', markersize=5, alpha=0.6)
            # Mark maintenance time
            ax1.axvline(t_maint, color='red', linestyle='--', alpha=0.2, linewidth=1)

    ax1.axhline(tau, color='red', linestyle='--', linewidth=2.5,
               label=f'Threshold τ={tau:.2f}')
    ax1.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Health Index', fontsize=13, fontweight='bold')
    ax1.set_title(f'HI Trajectories with Maintenance Planning (n={len(hi_trajs)})',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # 2. TTF Distribution (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(ttf, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(ttf), color='red', linestyle='--', linewidth=2,
               label=f'μ={np.mean(ttf):.2f}')
    ax2.set_xlabel('Time to Failure', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('TTF Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Threshold Crossing Times (middle right)
    ax3 = fig.add_subplot(gs[1, 2])
    t_stars = [r['t_star'] for r in maint_results if r['crossed']]

    if t_stars:
        ax3.hist(t_stars, bins=25, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(np.mean(t_stars), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(t_stars):.2f}')
        ax3.set_xlabel('Crossing Time t*', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title(f'Threshold Crossings ({len(t_stars)} assets)',
                     fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    # 4. Inspection Intervals (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    s_stars = [r['s_star'] for r in maint_results if r['crossed']]

    if s_stars:
        ax4.hist(s_stars, bins=25, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(np.mean(s_stars), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(s_stars):.2f}')
        ax4.set_xlabel('Inspection Interval s*', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Optimal Inspection Intervals', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    # 5. t* vs s* relationship (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])

    if t_stars and s_stars:
        ax5.scatter(t_stars, s_stars, alpha=0.6, s=60, c='darkblue', edgecolors='black')

        # Trend line
        if len(t_stars) > 1:
            z = np.polyfit(t_stars, s_stars, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(t_stars), max(t_stars), 100)
            ax5.plot(x_trend, p(x_trend), "r--", linewidth=2.5,
                    label=f'Trend: s*={z[0]:.3f}t*{z[1]:+.3f}')

        ax5.set_xlabel('Crossing Time t*', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Inspection Interval s*', fontsize=11, fontweight='bold')
        ax5.set_title('Maintenance Scheduling Pattern', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

    # 6. Summary Statistics (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    n_crossed = len(t_stars)
    crossing_rate = n_crossed / len(hi_trajs) * 100

    summary_text = f"""
    SUMMARY STATISTICS
    {'='*35}

    Assets Simulated: {len(hi_trajs)}
    Distribution: {dist_type.upper()}

    TTF Statistics:
    • Mean: {np.mean(ttf):.3f}
    • Std: {np.std(ttf):.3f}
    • CV: {results['scheduler'].cv:.3f}

    Model Parameters:
    • μ (MTTF): {results['scheduler'].mu:.3f}
    • k (slope): {results['scheduler'].k:.3f}
    • τ (threshold): {tau:.3f}
    • α (prob level): {alpha:.3f}

    Crossing Analysis:
    • Crossed threshold: {n_crossed} ({crossing_rate:.1f}%)
    • Mean t*: {np.mean(t_stars) if t_stars else 'N/A':.3f}
    • Mean s*: {np.mean(s_stars) if s_stars else 'N/A':.3f}
    • Mean next maint: {np.mean([r['next_maintenance'] for r in maint_results if r['crossed']]) if t_stars else 'N/A':.3f}
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Overall title
    fig.suptitle(f'Preventive Maintenance Analysis using Time Transformation (α={alpha})',
                fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{save_prefix}_{timestamp}.png'

    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to '{filename}'")
    except Exception as e:
        print(f"⚠ Could not save figure: {e}")

    return fig


def save_results_to_csv(results: dict, filename: str = 'maintenance_results.csv'):
    """Save detailed results to CSV file."""

    data = {
        'Asset_ID': [],
        'TTF': [],
        'b_parameter': [],
        'Crossed_Threshold': [],
        'Crossing_Time_t_star': [],
        'Inspection_Interval_s_star': [],
        'Next_Maintenance': [],
        'Expected_RUL_at_t_star': []
    }

    ttf = results['ttf_samples']
    b_vals = results['b_samples']
    maint = results['maintenance_results']

    for i, (ttf_val, b_val, m) in enumerate(zip(ttf, b_vals, maint)):
        data['Asset_ID'].append(i)
        data['TTF'].append(ttf_val)
        data['b_parameter'].append(b_val)
        data['Crossed_Threshold'].append(m['crossed'])
        data['Crossing_Time_t_star'].append(m['t_star'] if m['crossed'] else np.nan)
        data['Inspection_Interval_s_star'].append(m['s_star'] if m['crossed'] else np.nan)
        data['Next_Maintenance'].append(m['next_maintenance'] if m['crossed'] else np.nan)
        data['Expected_RUL_at_t_star'].append(m['rul_at_crossing'] if m['crossed'] else np.nan)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✓ Results saved to '{filename}'")

    return df


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main interface for HI simulation with maintenance scheduling."""

    print("\n" + "="*75)
    print(" "*15 + "HI SIMULATION WITH MAINTENANCE SCHEDULING")
    print(" "*20 + "Using Time Transformation Method")
    print("="*75)

    print("\n📋 This tool will:")
    print("  1. Generate HI trajectories from parametric distributions")
    print("  2. Detect when HI crosses threshold τ")
    print("  3. Compute optimal next inspection time using time transformation")
    print("  4. Provide comprehensive analysis and visualization")
    print("\n" + "="*75)

    # Get basic parameters
    try:
        print("\n" + "="*75)
        print("STEP 1: BASIC CONFIGURATION")
        print("="*75)

        n_assets = int(input("\nNumber of assets to simulate [50]: ") or "50")
        tau = float(input("HI threshold τ for maintenance trigger [0.7]: ") or "0.7")
        alpha = float(input("Probability level α (e.g., 0.05 means 95% confidence) [0.05]: ") or "0.05")

        print(f"\n  ✓ Will simulate {n_assets} assets")
        print(f"  ✓ Threshold: τ = {tau}")
        print(f"  ✓ Constraint: P(RUL(t*) < s*) ≤ {alpha}")

        # Get distribution parameters
        print("\n" + "="*75)
        print("STEP 2: SELECT DEGRADATION MODEL")
        print("="*75)
        print("\nAvailable distributions for parameter b:")
        print("  1. Fréchet [default]")
        print("  2. Weibull")
        print("  3. Gamma")
        print("  4. Lognormal")

        choice = input("\nSelect distribution [1]: ") or "1"

        if choice == '1':
            dist_type = 'frechet'
            shape = float(input("Fréchet shape parameter [3.0]: ") or "3.0")
            scale = float(input("Fréchet scale parameter [0.01]: ") or "0.01")
            params = {'shape': shape, 'scale': scale}
        elif choice == '2':
            dist_type = 'weibull'
            shape = float(input("Weibull shape parameter [2.5]: ") or "2.5")
            scale = float(input("Weibull scale parameter [0.015]: ") or "0.015")
            params = {'shape': shape, 'scale': scale}
        elif choice == '3':
            dist_type = 'gamma'
            shape = float(input("Gamma shape parameter [5.0]: ") or "5.0")
            scale = float(input("Gamma scale parameter [0.003]: ") or "0.003")
            params = {'shape': shape, 'scale': scale}
        else:
            dist_type = 'lognormal'
            mean = float(input("Lognormal mean (log-scale) [-4.0]: ") or "-4.0")
            std = float(input("Lognormal std (log-scale) [0.5]: ") or "0.5")
            params = {'mean': mean, 'std': std}

        p = float(input("\nPower parameter p for HI(t)=1-b*t^p [3.0]: ") or "3.0")

        print(f"\n  ✓ Distribution: {dist_type.upper()}")
        print(f"  ✓ Parameters: {params}")
        print(f"  ✓ Power: p = {p}")

        # Run simulation
        print("\n" + "="*75)
        print("STEP 3: RUNNING SIMULATION")
        print("="*75)
        print("\n⏳ Generating HI trajectories...")

        results = simulate_hi_with_maintenance(
            n_assets=n_assets,
            dist_type=dist_type,
            params=params,
            p=p,
            tau=tau,
            alpha=alpha,
            time_points=300
        )

        print("✓ HI trajectories generated")
        print("⏳ Analyzing threshold crossings and computing maintenance schedules...")

        # Print summary
        n_crossed = sum(1 for r in results['maintenance_results'] if r['crossed'])
        crossing_rate = n_crossed / n_assets * 100

        print("\n" + "="*75)
        print("STEP 4: RESULTS SUMMARY")
        print("="*75)

        print(f"\n📊 Simulation Results:")
        print(f"  • Total assets simulated: {n_assets}")
        print(f"  • Assets crossing threshold: {n_crossed} ({crossing_rate:.1f}%)")

        if n_crossed > 0:
            t_stars = [r['t_star'] for r in results['maintenance_results'] if r['crossed']]
            s_stars = [r['s_star'] for r in results['maintenance_results'] if r['crossed']]
            ruls = [r['rul_at_crossing'] for r in results['maintenance_results'] if r['crossed']]

            print(f"\n📈 Threshold Crossing Times (t*):")
            print(f"  • Mean: {np.mean(t_stars):.4f}")
            print(f"  • Std:  {np.std(t_stars):.4f}")
            print(f"  • Range: [{np.min(t_stars):.4f}, {np.max(t_stars):.4f}]")

            print(f"\n⏱️  Optimal Inspection Intervals (s*):")
            print(f"  • Mean: {np.mean(s_stars):.4f}")
            print(f"  • Std:  {np.std(s_stars):.4f}")
            print(f"  • Range: [{np.min(s_stars):.4f}, {np.max(s_stars):.4f}]")

            print(f"\n🔧 Next Maintenance Times (t* + s*):")
            next_maint = [r['next_maintenance'] for r in results['maintenance_results'] if r['crossed']]
            print(f"  • Mean: {np.mean(next_maint):.4f}")
            print(f"  • Range: [{np.min(next_maint):.4f}, {np.max(next_maint):.4f}]")

            print(f"\n💡 Expected RUL at Threshold Crossing:")
            print(f"  • Mean: {np.mean(ruls):.4f}")
            print(f"  • Compare with mean s*: {np.mean(s_stars):.4f}")
        else:
            print("\n  ⚠️  No assets crossed the threshold in the simulation period.")
            print("     Consider:")
            print("       - Increasing threshold τ")
            print("       - Extending simulation time")
            print("       - Adjusting distribution parameters")

        # Save results
        print("\n" + "="*75)
        print("STEP 5: SAVING RESULTS")
        print("="*75)

        df = save_results_to_csv(results)

        print(f"\n📄 Saved detailed results:")
        print(f"  • CSV file: maintenance_results.csv")
        print(f"  • Rows: {len(df)}")
        print(f"  • Columns: {', '.join(df.columns)}")

        # Create visualization
        print("\n⏳ Creating visualizations...")
        fig = create_comprehensive_visualization(results, dist_type)

        print("\n" + "="*75)
        print("✅ ANALYSIS COMPLETE!")
        print("="*75)
        print("\n📁 Generated files:")
        print("  • maintenance_results.csv - Detailed data")
        print("  • maintenance_analysis_[timestamp].png - Comprehensive plots")
        print("\n💡 Interpretation Guide:")
        print("  • t* = Time when HI first drops below τ")
        print("  • s* = Optimal time to next inspection")
        print(f"  • Constraint: P(RUL(t*) < s*) ≤ {alpha}")
        print("  • Next maintenance should occur at t* + s*")
        print("="*75 + "\n")

        # Show plots
        plt.show()

        return results, df

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
        return None, None
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    results, df = main()
