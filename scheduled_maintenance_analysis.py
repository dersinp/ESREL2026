"""
Scheduled Preventive Maintenance Analysis and Optimization
===========================================================

Implements fixed-interval preventive maintenance policy where maintenance
occurs at predetermined times T, 2T, 3T, etc., regardless of asset health.

Features:
- Scheduled preventive maintenance at fixed intervals T
- Corrective maintenance when failure occurs (HI ≤ 0)
- Perfect maintenance: HI returns to 1 after intervention
- Cost tracking and optimization
- Comparison with predictive maintenance strategy

Key Difference from Predictive Maintenance:
- Predictive: Maintenance based on HI threshold τ and probability α
- Scheduled: Maintenance at fixed intervals T (time-based)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma, weibull_min, lognorm, gaussian_kde
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MaintenanceEvent:
    """Record of a single maintenance event."""
    time: float
    event_type: str  # 'corrective' or 'preventive'
    cost: float
    hi_before: float
    asset_id: int


@dataclass
class AssetMaintenanceHistory:
    """Complete maintenance history for a single asset over horizon H."""
    asset_id: int
    b_parameter: float
    ttf: float  # Original TTF (first passage time)

    # Events lists
    corrective_events: List[MaintenanceEvent] = field(default_factory=list)
    preventive_events: List[MaintenanceEvent] = field(default_factory=list)

    # Cost totals
    total_corrective_cost: float = 0.0
    total_preventive_cost: float = 0.0
    total_cost: float = 0.0

    # Counts
    n_corrective: int = 0
    n_preventive: int = 0
    n_total: int = 0

    # Time series data
    hi_trajectory: Optional[np.ndarray] = None
    time_grid: Optional[np.ndarray] = None
    maintenance_times: List[float] = field(default_factory=list)
    maintenance_types: List[str] = field(default_factory=list)


class ScheduledMaintenanceScheduler:
    """
    Scheduler for fixed-interval preventive maintenance.
    
    Maintenance occurs at times: T, 2T, 3T, ..., until horizon H.
    """

    def __init__(self, T: float, Cc: float, Cp: float, H: float):
        """
        Parameters:
        -----------
        T : float
            Preventive maintenance interval (time between scheduled maintenances)
        Cc : float
            Corrective maintenance cost (failure cost)
        Cp : float
            Preventive maintenance cost (scheduled cost)
        H : float
            Time horizon for analysis
        """
        self.T = T
        self.Cc = Cc
        self.Cp = Cp
        self.H = H

        print(f"Scheduled Maintenance Scheduler Initialized:")
        print(f"  • Maintenance interval T = {T}")
        print(f"  • Corrective cost Cc = ${Cc:,.2f}")
        print(f"  • Preventive cost Cp = ${Cp:,.2f}")
        print(f"  • Time horizon H = {H}")
        print(f"  • Expected maintenances per asset: {int(H/T)}")

    def simulate_single_asset(self, b: float, p: float,
                             dist_type: str = 'weibull',
                             asset_id: int = 0,
                             time_resolution: int = 500) -> AssetMaintenanceHistory:
        """
        Simulate a single asset over horizon H with scheduled maintenance.

        Parameters:
        -----------
        b : float
            Degradation parameter
        p : float
            Power parameter
        dist_type : str
            Type of degradation model
        asset_id : int
            Asset identifier
        time_resolution : int
            Number of time points per cycle for trajectory

        Returns:
        --------
        AssetMaintenanceHistory
            Complete maintenance history with costs
        """
        # Initialize history
        if dist_type.lower() == 'gamma':
            ttf = 1 / b
        else:
            ttf = (1 / b) ** (1/p)

        history = AssetMaintenanceHistory(
            asset_id=asset_id,
            b_parameter=b,
            ttf=ttf
        )

        # Track full trajectory over horizon
        full_time = []
        full_hi = []

        current_time = 0.0
        next_scheduled_maintenance = self.T
        cycle_number = 0

        while current_time < self.H:
            cycle_number += 1

            # Generate HI trajectory for this cycle starting from HI=1
            cycle_duration = min(self.H - current_time, next_scheduled_maintenance - current_time + ttf)
            t_cycle = np.linspace(0, cycle_duration, time_resolution)

            if dist_type.lower() == 'gamma':
                hi_cycle = 1 - (b * t_cycle) ** p
            else:
                hi_cycle = 1 - b * (t_cycle ** p)

            hi_cycle = np.clip(hi_cycle, 0, 1)

            # Find failure time in this cycle (HI ≤ 0)
            failure_idx = np.where(hi_cycle <= 0)[0]

            if len(failure_idx) > 0:
                t_failure_local = t_cycle[failure_idx[0]]
                t_failure_global = current_time + t_failure_local
            else:
                t_failure_global = np.inf

            # Determine what happens first: scheduled maintenance or failure
            if t_failure_global < next_scheduled_maintenance and t_failure_global < self.H:
                # CASE 1: FAILURE BEFORE SCHEDULED MAINTENANCE
                maintenance_time = t_failure_global
                maintenance_type = 'corrective'
                cost = self.Cc

                # Add trajectory up to failure
                failure_mask = (current_time + t_cycle) <= maintenance_time
                full_time.extend((current_time + t_cycle)[failure_mask])
                full_hi.extend(hi_cycle[failure_mask])

                # Record event
                event = MaintenanceEvent(
                    time=maintenance_time,
                    event_type='corrective',
                    cost=cost,
                    hi_before=0.0,
                    asset_id=asset_id
                )
                history.corrective_events.append(event)
                history.n_corrective += 1
                history.total_corrective_cost += cost

                # Next scheduled maintenance stays the same
                # (we don't reset the schedule after a failure)

            elif next_scheduled_maintenance < self.H:
                # CASE 2: SCHEDULED PREVENTIVE MAINTENANCE
                maintenance_time = next_scheduled_maintenance
                maintenance_type = 'preventive'
                cost = self.Cp

                # Add trajectory up to scheduled maintenance
                maint_mask = (current_time + t_cycle) <= maintenance_time
                full_time.extend((current_time + t_cycle)[maint_mask])
                full_hi.extend(hi_cycle[maint_mask])

                # Get HI at scheduled maintenance time
                hi_at_maint = np.interp(maintenance_time - current_time,
                                       t_cycle, hi_cycle)

                # Record event
                event = MaintenanceEvent(
                    time=maintenance_time,
                    event_type='preventive',
                    cost=cost,
                    hi_before=hi_at_maint,
                    asset_id=asset_id
                )
                history.preventive_events.append(event)
                history.n_preventive += 1
                history.total_preventive_cost += cost

                # Schedule next preventive maintenance
                next_scheduled_maintenance += self.T

            else:
                # Beyond horizon - add remaining trajectory
                horizon_mask = (current_time + t_cycle) <= self.H
                full_time.extend((current_time + t_cycle)[horizon_mask])
                full_hi.extend(hi_cycle[horizon_mask])
                break

            # Record maintenance
            history.maintenance_times.append(maintenance_time)
            history.maintenance_types.append(maintenance_type)

            # Perfect maintenance: HI returns to 1, continue from maintenance time
            current_time = maintenance_time

            # Add maintenance event (HI jumps back to 1)
            full_time.append(maintenance_time)
            full_hi.append(1.0)

        # Finalize history
        history.total_cost = history.total_corrective_cost + history.total_preventive_cost
        history.n_total = history.n_corrective + history.n_preventive
        history.time_grid = np.array(full_time)
        history.hi_trajectory = np.array(full_hi)

        return history

    def simulate_fleet(self, n_assets: int, dist_type: str,
                      params: dict, p: float) -> List[AssetMaintenanceHistory]:
        """
        Simulate entire fleet with scheduled maintenance.

        Parameters:
        -----------
        n_assets : int
            Number of assets
        dist_type : str
            Distribution type
        params : dict
            Distribution parameters
        p : float
            Power parameter

        Returns:
        --------
        List[AssetMaintenanceHistory]
            Complete histories for all assets
        """
        # Generate b parameters
        b_samples = self._generate_b_samples(dist_type, params, n_assets)

        print(f"\n⏳ Simulating {n_assets} assets over horizon H={self.H}...")

        histories = []
        for i, b in enumerate(b_samples):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{n_assets} assets...")

            history = self.simulate_single_asset(
                b=b, p=p, dist_type=dist_type, asset_id=i
            )
            histories.append(history)

        print(f"✓ Simulation complete!")

        return histories

    def _generate_b_samples(self, dist_type: str, params: dict, size: int) -> np.ndarray:
        """Generate b parameter samples from distribution."""
        if dist_type.lower() == 'weibull':
            return weibull_min.rvs(c=params['shape'], scale=params['scale'], size=size)
        elif dist_type.lower() == 'gamma':
            return gamma.rvs(a=params['shape'], scale=params['scale'], size=size)
        elif dist_type.lower() == 'frechet':
            U = np.random.uniform(0, 1, size)
            return params['scale'] * ((-np.log(U)) ** (-1/params['shape']))
        elif dist_type.lower() == 'lognormal':
            return lognorm.rvs(s=params['std'], scale=np.exp(params['mean']), size=size)
        else:
            raise ValueError(f"Unknown distribution: {dist_type}")

    def compute_fleet_statistics(self, histories: List[AssetMaintenanceHistory]) -> dict:
        """
        Compute aggregate statistics for entire fleet.

        Returns:
        --------
        dict with costs, counts, and rates
        """
        n_assets = len(histories)

        total_corrective = sum(h.total_corrective_cost for h in histories)
        total_preventive = sum(h.total_preventive_cost for h in histories)
        total_cost = sum(h.total_cost for h in histories)

        total_corrective_events = sum(h.n_corrective for h in histories)
        total_preventive_events = sum(h.n_preventive for h in histories)
        total_events = sum(h.n_total for h in histories)

        # Per asset averages
        avg_cost_per_asset = total_cost / n_assets
        avg_corrective_per_asset = total_corrective / n_assets
        avg_preventive_per_asset = total_preventive / n_assets

        # Cost per unit time
        cost_per_unit_time = total_cost / (self.H * n_assets)

        # Failure rate
        failure_rate = total_corrective_events / (self.H * n_assets)

        # Maintenance rate
        maintenance_rate = total_events / (self.H * n_assets)

        return {
            'n_assets': n_assets,
            'horizon': self.H,
            'T': self.T,
            'Cc': self.Cc,
            'Cp': self.Cp,

            # Total costs
            'total_corrective_cost': total_corrective,
            'total_preventive_cost': total_preventive,
            'total_cost': total_cost,

            # Average costs
            'avg_cost_per_asset': avg_cost_per_asset,
            'avg_corrective_per_asset': avg_corrective_per_asset,
            'avg_preventive_per_asset': avg_preventive_per_asset,

            # Counts
            'total_corrective_events': total_corrective_events,
            'total_preventive_events': total_preventive_events,
            'total_events': total_events,

            # Rates
            'cost_per_unit_time': cost_per_unit_time,
            'failure_rate': failure_rate,
            'maintenance_rate': maintenance_rate,

            # Ratios
            'corrective_cost_ratio': total_corrective / total_cost if total_cost > 0 else 0,
            'preventive_cost_ratio': total_preventive / total_cost if total_cost > 0 else 0,
            'corrective_event_ratio': total_corrective_events / total_events if total_events > 0 else 0,
            'preventive_event_ratio': total_preventive_events / total_events if total_events > 0 else 0
        }


# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def optimize_maintenance_interval(n_assets: int, H: float, Cc: float, Cp: float,
                                  dist_type: str, params: dict, p: float,
                                  T_range: List[float] = None) -> dict:
    """
    Find optimal maintenance interval T that minimizes total cost.

    Parameters:
    -----------
    n_assets : int
        Number of assets to simulate
    H : float
        Time horizon
    Cc : float
        Corrective cost
    Cp : float
        Preventive cost
    dist_type : str
        Distribution type
    params : dict
        Distribution parameters
    p : float
        Power parameter
    T_range : List[float], optional
        List of T values to test

    Returns:
    --------
    dict with optimization results
    """
    if T_range is None:
        T_range = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]

    print("\n" + "="*70)
    print("OPTIMIZING MAINTENANCE INTERVAL T")
    print("="*70)
    print(f"\nTesting T values: {T_range}")
    print(f"Configuration: n={n_assets}, H={H}, Cc=${Cc:,.0f}, Cp=${Cp:,.0f}")
    print("\n⏳ Running simulations...")

    results = []

    for i, T in enumerate(T_range):
        print(f"\n[{i+1}/{len(T_range)}] Testing T = {T:.1f}...")

        # Create scheduler
        scheduler = ScheduledMaintenanceScheduler(
            T=T,
            Cc=Cc,
            Cp=Cp,
            H=H
        )

        # Simulate fleet
        histories = scheduler.simulate_fleet(
            n_assets=n_assets,
            dist_type=dist_type,
            params=params,
            p=p
        )

        # Compute stats
        stats = scheduler.compute_fleet_statistics(histories)

        results.append({
            'T': T,
            'total_cost': stats['total_cost'],
            'avg_cost_per_asset': stats['avg_cost_per_asset'],
            'corrective_cost': stats['total_corrective_cost'],
            'preventive_cost': stats['total_preventive_cost'],
            'n_corrective': stats['total_corrective_events'],
            'n_preventive': stats['total_preventive_events'],
            'failure_rate': stats['failure_rate'],
            'cost_per_unit_time': stats['cost_per_unit_time']
        })

        print(f"  Total cost: ${stats['total_cost']:,.2f}")
        print(f"  Failures: {stats['total_corrective_events']}, Preventive: {stats['total_preventive_events']}")

    # Find optimal
    costs = [r['total_cost'] for r in results]
    min_idx = np.argmin(costs)
    optimal_T = results[min_idx]['T']
    optimal_cost = results[min_idx]['total_cost']

    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\n✓ Optimal T = {optimal_T:.1f}")
    print(f"✓ Minimum total cost = ${optimal_cost:,.2f}")
    print(f"✓ Expected maintenances per asset: ~{int(H/optimal_T)}")

    return {
        'optimal_T': optimal_T,
        'optimal_cost': optimal_cost,
        'results': results,
        'T_range': T_range
    }


def visualize_scheduled_optimization(opt_results: dict,
                                    save_path: str = 'scheduled_optimization.png'):
    """Visualize scheduled maintenance optimization results."""

    results = opt_results['results']
    T_values = [r['T'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Total cost vs T
    ax = axes[0, 0]
    total_costs = [r['total_cost'] for r in results]
    ax.plot(T_values, total_costs, 'o-', linewidth=2.5, markersize=10, color='darkblue')
    ax.axvline(opt_results['optimal_T'], color='red', linestyle='--', linewidth=2,
              label=f'Optimal T={opt_results["optimal_T"]:.1f}')
    ax.set_xlabel('Maintenance Interval T', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Total Cost vs Maintenance Interval', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Cost breakdown vs T
    ax = axes[0, 1]
    corrective_costs = [r['corrective_cost'] for r in results]
    preventive_costs = [r['preventive_cost'] for r in results]

    ax.plot(T_values, corrective_costs, 's-', linewidth=2, markersize=8,
           label='Corrective', color='red')
    ax.plot(T_values, preventive_costs, '^-', linewidth=2, markersize=8,
           label='Preventive', color='green')
    ax.axvline(opt_results['optimal_T'], color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Maintenance Interval T', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Components vs T', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Maintenance counts vs T
    ax = axes[1, 0]
    n_corrective = [r['n_corrective'] for r in results]
    n_preventive = [r['n_preventive'] for r in results]

    ax.plot(T_values, n_corrective, 's-', linewidth=2, markersize=8,
           label='Corrective', color='red')
    ax.plot(T_values, n_preventive, '^-', linewidth=2, markersize=8,
           label='Preventive', color='green')
    ax.axvline(opt_results['optimal_T'], color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Maintenance Interval T', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax.set_title('Maintenance Events vs T', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Failure rate vs T
    ax = axes[1, 1]
    failure_rates = [r['failure_rate'] for r in results]

    ax.plot(T_values, failure_rates, 'o-', linewidth=2.5, markersize=10, color='purple')
    ax.axvline(opt_results['optimal_T'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Maintenance Interval T', fontsize=12, fontweight='bold')
    ax.set_ylabel('Failure Rate (per asset per time)', fontsize=12, fontweight='bold')
    ax.set_title('Failure Rate vs T', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Optimization of Scheduled Maintenance Interval T',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"✓ Optimization visualization saved to '{save_path}'")

    return fig


# ============================================================================
# COMPARISON WITH PREDICTIVE MAINTENANCE
# ============================================================================

def compare_strategies(scheduled_results: dict, predictive_results: dict,
                      save_path: str = 'strategy_comparison.png'):
    """
    Compare scheduled vs. predictive maintenance strategies.
    
    Parameters:
    -----------
    scheduled_results : dict
        Results from optimize_maintenance_interval()
    predictive_results : dict
        Results from predictive maintenance optimization (with optimal τ, α)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Cost comparison
    ax = axes[0, 0]
    strategies = ['Scheduled\nMaintenance', 'Predictive\nMaintenance']
    costs = [scheduled_results['optimal_cost'], predictive_results['optimal_cost']]
    colors = ['steelblue', 'orange']
    
    bars = ax.bar(strategies, costs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Total Cost Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${cost:,.0f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Calculate and show savings
    savings = abs(costs[0] - costs[1])
    better_idx = np.argmin(costs)
    savings_pct = savings / costs[1-better_idx] * 100
    
    ax.text(0.5, 0.95, f'Savings: ${savings:,.0f} ({savings_pct:.1f}%)\nBest: {strategies[better_idx].replace(chr(10), " ")}',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
           fontsize=10, fontweight='bold')

    # 2. Cost breakdown comparison
    ax = axes[0, 1]
    sched_res = [r for r in scheduled_results['results'] if r['T'] == scheduled_results['optimal_T']][0]
    
    x = np.arange(2)
    width = 0.35
    
    corrective = [sched_res['corrective_cost'], predictive_results.get('corrective_cost', 0)]
    preventive = [sched_res['preventive_cost'], predictive_results.get('preventive_cost', 0)]
    
    ax.bar(x - width/2, corrective, width, label='Corrective', color='red', alpha=0.7)
    ax.bar(x + width/2, preventive, width, label='Preventive', color='green', alpha=0.7)
    
    ax.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Breakdown by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Event counts comparison
    ax = axes[1, 0]
    
    n_corrective = [sched_res['n_corrective'], predictive_results.get('n_corrective', 0)]
    n_preventive = [sched_res['n_preventive'], predictive_results.get('n_preventive', 0)]
    
    ax.bar(x - width/2, n_corrective, width, label='Failures', color='red', alpha=0.7)
    ax.bar(x + width/2, n_preventive, width, label='Preventive', color='green', alpha=0.7)
    
    ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax.set_title('Maintenance Events by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Key metrics comparison
    ax = axes[1, 1]
    ax.axis('off')
    
    comparison_text = f"""
    STRATEGY COMPARISON SUMMARY
    {'='*50}
    
    SCHEDULED MAINTENANCE:
    • Optimal interval: T = {scheduled_results['optimal_T']:.1f}
    • Total cost: ${scheduled_results['optimal_cost']:,.0f}
    • Corrective events: {sched_res['n_corrective']:.0f}
    • Preventive events: {sched_res['n_preventive']:.0f}
    • Failure rate: {sched_res['failure_rate']:.4f}
    
    PREDICTIVE MAINTENANCE:
    • Optimal threshold: τ = {predictive_results.get('tau', 'N/A')}
    • Optimal probability: α = {predictive_results.get('alpha', 'N/A')}
    • Total cost: ${predictive_results['optimal_cost']:,.0f}
    • Corrective events: {predictive_results.get('n_corrective', 0):.0f}
    • Preventive events: {predictive_results.get('n_preventive', 0):.0f}
    • Failure rate: {predictive_results.get('failure_rate', 0):.4f}
    
    WINNER: {strategies[better_idx].replace(chr(10), " ")}
    Savings: ${savings:,.0f} ({savings_pct:.1f}%)
    
    Key Insight:
    {'Predictive maintenance provides better' if better_idx == 1 else 'Scheduled maintenance provides better'}
    cost efficiency by {'adapting to asset health' if better_idx == 1 else 'regular interventions'}.
    """
    
    ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Scheduled vs. Predictive Maintenance Comparison',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"✓ Strategy comparison saved to '{save_path}'")
    
    return fig


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Interactive scheduled maintenance analysis."""

    print("\n" + "="*75)
    print(" "*8 + "SCHEDULED PREVENTIVE MAINTENANCE ANALYSIS")
    print(" "*15 + "Fixed-Interval Maintenance Policy")
    print("="*75)

    print("\n📋 This system implements:")
    print("  • Scheduled preventive maintenance at fixed intervals T")
    print("  • Corrective maintenance when failures occur (HI ≤ 0)")
    print("  • Perfect maintenance (HI → 1) after each intervention")
    print("  • Cost optimization to find optimal interval T")
    print("  • Comparison with predictive maintenance strategy")
    print("\n" + "="*75)

    try:
        # Configuration
        print("\n" + "="*75)
        print("STEP 1: BASIC CONFIGURATION")
        print("="*75)

        n_assets = int(input("\nNumber of assets [50]: ") or "50")
        H = float(input("Time horizon H [200]: ") or "200")

        print("\n" + "="*75)
        print("STEP 2: COST PARAMETERS")
        print("="*75)

        Cc = float(input("\nCorrective maintenance cost Cc [1000]: ") or "1000")
        Cp = float(input("Preventive maintenance cost Cp [100]: ") or "100")

        print(f"\n  ✓ Cost ratio Cc/Cp = {Cc/Cp:.2f}:1")

        # Distribution
        print("\n" + "="*75)
        print("STEP 3: DEGRADATION MODEL")
        print("="*75)

        print("\nSelect distribution:")
        print("  1. Fréchet [default]")
        print("  2. Weibull")
        print("  3. Gamma")
        print("  4. Lognormal")

        choice = input("\nChoice [1]: ") or "1"

        p = float(input("\nPower parameter p [3.0]: ") or "3.0")

        if choice == '1':
            dist_type = 'frechet'
            print("\n" + "-"*50)
            print("Fréchet Distribution (TTF Parameters)")
            print("-"*50)
            print("For HI(t) = 1 - b*t^p with b ~ Fréchet")
            print("Enter TTF distribution parameters:")
            print("  TTF = time until HI reaches 0")
            print("")
            
            beta = float(input("TTF shape parameter β [3.0]: ") or "3.0")
            eta = float(input("TTF scale parameter η [100.0]: ") or "100.0")
            
            # Convert TTF parameters to Fréchet b parameters
            beta_b = beta / p
            eta_b = 1.0 / (eta ** p)
            
            params = {'shape': beta_b, 'scale': eta_b}
            
            print(f"\n  ✓ TTF parameters: β={beta:.2f}, η={eta:.2f}")
            print(f"  ✓ Converted to Fréchet b parameters: shape={beta_b:.4f}, scale={eta_b:.6e}")
            
        elif choice == '2':
            dist_type = 'weibull'
            shape = float(input("Weibull shape [2.5]: ") or "2.5")
            scale = float(input("Weibull scale [0.015]: ") or "0.015")
            params = {'shape': shape, 'scale': scale}
        elif choice == '3':
            dist_type = 'gamma'
            shape = float(input("Gamma shape [5.0]: ") or "5.0")
            scale = float(input("Gamma scale [0.003]: ") or "0.003")
            params = {'shape': shape, 'scale': scale}
        else:
            dist_type = 'lognormal'
            mean = float(input("Lognormal mean (log-scale) [-4.0]: ") or "-4.0")
            std = float(input("Lognormal std (log-scale) [0.5]: ") or "0.5")
            params = {'mean': mean, 'std': std}

        # Optimization
        print("\n" + "="*75)
        print("STEP 4: RUNNING OPTIMIZATION")
        print("="*75)

        # Optimize T
        opt_results = optimize_maintenance_interval(
            n_assets=n_assets,
            H=H,
            Cc=Cc,
            Cp=Cp,
            dist_type=dist_type,
            params=params,
            p=p,
            T_range=[10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
        )

        # Visualize
        print("\n⏳ Creating visualizations...")
        fig = visualize_scheduled_optimization(opt_results)

        # Save results
        results_df = pd.DataFrame(opt_results['results'])
        results_df.to_csv('scheduled_optimization_results.csv', index=False)

        summary_data = {
            'Optimal_T': [opt_results['optimal_T']],
            'Optimal_Cost': [opt_results['optimal_cost']],
            'N_Assets': [n_assets],
            'Horizon': [H],
            'Cc': [Cc],
            'Cp': [Cp],
            'Cost_Ratio': [Cc/Cp],
            'Distribution': [dist_type]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('scheduled_optimization_summary.csv', index=False)

        print("\n" + "="*75)
        print("✅ SCHEDULED MAINTENANCE ANALYSIS COMPLETE!")
        print("="*75)
        print("\n📁 Generated files:")
        print("  • scheduled_optimization.png - Optimization results")
        print("  • scheduled_optimization_results.csv - Detailed results")
        print("  • scheduled_optimization_summary.csv - Summary statistics")
        
        print("\n💡 Key findings:")
        print(f"  • Optimal maintenance interval: T* = {opt_results['optimal_T']:.1f}")
        print(f"  • Minimum total cost: ${opt_results['optimal_cost']:,.0f}")
        print(f"  • Expected maintenances per asset: ~{int(H/opt_results['optimal_T'])}")
        
        print("\n📊 Next steps:")
        print("  • Compare with predictive maintenance using same configuration")
        print("  • Test sensitivity to cost ratio (Cc/Cp)")
        print("  • Validate on real asset data")
        print("="*75 + "\n")

        plt.show()

        return opt_results

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
        return None
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
