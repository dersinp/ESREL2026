"""
Cost-Based Preventive Maintenance Analysis
===========================================

Extends the maintenance scheduler to include:
- Corrective maintenance cost (Cc) when failure occurs
- Preventive maintenance cost (Cp) when scheduled inspection performed
- Perfect maintenance: HI returns to 1 after intervention
- Trajectory continuation over finite horizon H
- Total cost optimization analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import gamma, weibull_min, lognorm, gaussian_kde
from scipy.integrate import cumulative_trapezoid as cumtrapz
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import sys
from pathlib import Path


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


class CostBasedMaintenanceScheduler:
    """
    Enhanced scheduler that tracks costs and simulates maintenance cycles
    over a finite horizon with perfect maintenance.
    """

    def __init__(self, ttf_data: np.ndarray, tau: float, alpha: float,
                 Cc: float, Cp: float, H: float):
        """
        Parameters:
        -----------
        ttf_data : np.ndarray
            Time-to-failure data for model fitting
        tau : float
            HI threshold (0 < tau < 1)
        alpha : float
            Probability constraint (0 < alpha < 1)
        Cc : float
            Corrective maintenance cost (failure cost)
        Cp : float
            Preventive maintenance cost (scheduled cost)
        H : float
            Time horizon for analysis
        """
        self.tau = tau
        self.alpha = alpha
        self.Cc = Cc
        self.Cp = Cp
        self.H = H

        # Fit time transformation model
        self._fit_model(ttf_data)

        print(f"Cost-Based Scheduler Initialized:")
        print(f"  • Threshold τ = {tau}")
        print(f"  • Confidence 1-α = {1-alpha}")
        print(f"  • Corrective cost Cc = ${Cc:,.2f}")
        print(f"  • Preventive cost Cp = ${Cp:,.2f}")
        print(f"  • Time horizon H = {H}")
        print(f"  • Model: μ={self.mu:.4f}, k={self.k:.4f}, CV={self.cv:.4f}")

    def _fit_model(self, ttf_data: np.ndarray):
        """Fit time transformation model from TTF data."""
        ttf_data = np.asarray(ttf_data)
        ttf_data = ttf_data[(ttf_data > 0) & (~np.isnan(ttf_data))]

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

        self.t_grid = t_grid
        self.reliability = reliability

        # Compute g(t) transformation and its inverse
        exponent = self.k / (1 - self.k)
        g_vals = (self.mu / self.k) * (1 - np.power(reliability, exponent))

        # Create interpolation functions for g(t) and g_inv
        self.g_fun = interp1d(t_grid, g_vals, bounds_error=False, fill_value="extrapolate")
        self.g_inv = interp1d(g_vals, t_grid, bounds_error=False, fill_value="extrapolate")

    def compute_next_inspection_time(self, t_star: float) -> float:
        """
        Compute optimal inspection interval s* at time t*.

        Uses the correct formula with time transformation:
        g(t* + s*) = g(t*) + (μ/k - g(t*)) × [1 - (1-α)^(k/(1-k))]
        Then: s* = g_inv(g(t* + s*)) - t*
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

        return np.maximum(s_star, 0.0)

    def simulate_single_asset_with_costs(self, b: float, p: float,
                                        dist_type: str = 'weibull',
                                        asset_id: int = 0,
                                        time_resolution: int = 500) -> AssetMaintenanceHistory:
        """
        Simulate a single asset over horizon H with maintenance cycles.

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
        cycle_number = 0

        while current_time < self.H:
            cycle_number += 1

            # Generate HI trajectory for this cycle starting from HI=1
            cycle_duration = min(self.H - current_time, ttf * 2)  # Extend a bit beyond expected TTF
            t_cycle = np.linspace(0, cycle_duration, time_resolution)

            if dist_type.lower() == 'gamma':
                hi_cycle = 1 - (b * t_cycle) ** p
            else:
                hi_cycle = 1 - b * (t_cycle ** p)

            hi_cycle = np.clip(hi_cycle, 0, 1)

            # Find threshold crossing in this cycle
            below_threshold = hi_cycle <= self.tau

            if not np.any(below_threshold):
                # No crossing in this cycle - just add trajectory and end
                full_time.extend(current_time + t_cycle)
                full_hi.extend(hi_cycle)
                break

            # Find threshold crossing time
            crossing_idx = np.where(below_threshold)[0][0]

            if crossing_idx > 0:
                # Interpolate exact crossing
                t1, t2 = t_cycle[crossing_idx-1], t_cycle[crossing_idx]
                hi1, hi2 = hi_cycle[crossing_idx-1], hi_cycle[crossing_idx]

                if hi1 != hi2:
                    t_cross_local = t1 + (self.tau - hi1) * (t2 - t1) / (hi2 - hi1)
                else:
                    t_cross_local = t1
            else:
                t_cross_local = t_cycle[crossing_idx]

            t_cross_global = current_time + t_cross_local

            # Compute next inspection time
            s_star = self.compute_next_inspection_time(t_cross_local)
            t_inspection_global = t_cross_global + s_star

            # Check if failure occurs before inspection
            failure_idx = np.where(hi_cycle <= 0)[0]

            if len(failure_idx) > 0:
                t_failure_local = t_cycle[failure_idx[0]]
                t_failure_global = current_time + t_failure_local
            else:
                t_failure_global = np.inf

            # Determine maintenance type
            if t_failure_global < t_inspection_global and t_failure_global < self.H:
                # CASE 1: CORRECTIVE MAINTENANCE (Failure occurred)
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

            elif t_inspection_global < self.H:
                # CASE 2: PREVENTIVE MAINTENANCE (Inspection performed)
                maintenance_time = t_inspection_global
                maintenance_type = 'preventive'
                cost = self.Cp

                # Add trajectory up to inspection
                inspection_mask = (current_time + t_cycle) <= maintenance_time
                full_time.extend((current_time + t_cycle)[inspection_mask])
                full_hi.extend(hi_cycle[inspection_mask])

                # Get HI at inspection time
                hi_at_inspection = np.interp(t_inspection_global - current_time,
                                            t_cycle, hi_cycle)

                # Record event
                event = MaintenanceEvent(
                    time=maintenance_time,
                    event_type='preventive',
                    cost=cost,
                    hi_before=hi_at_inspection,
                    asset_id=asset_id
                )
                history.preventive_events.append(event)
                history.n_preventive += 1
                history.total_preventive_cost += cost

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

    def simulate_fleet_with_costs(self, n_assets: int,
                                  dist_type: str,
                                  params: dict,
                                  p: float) -> List[AssetMaintenanceHistory]:
        """
        Simulate entire fleet with cost tracking.

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

            history = self.simulate_single_asset_with_costs(
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
        dict with:
            - Total costs (corrective, preventive, total)
            - Average costs per asset
            - Maintenance counts
            - Cost per unit time
            - Failure rate
            - And more...
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
            'tau': self.tau,
            'alpha': self.alpha,
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
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_cost_analysis(histories: List[AssetMaintenanceHistory],
                           stats: dict,
                           max_assets_to_plot: int = 10,
                           save_path: str = 'cost_analysis_results.png'):
    """
    Create comprehensive cost analysis visualization.
    """
    # Increased figure size and spacing for better readability
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35,
                         top=0.98, bottom=0.05, left=0.05, right=0.98)

    # Select subset for trajectory plots
    plot_histories = histories[:max_assets_to_plot]

    # 1. HI Trajectories with maintenance events (large, top-left 2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    for h in plot_histories:
        ax1.plot(h.time_grid, h.hi_trajectory, alpha=0.6, linewidth=1.5)

        # Mark corrective maintenance (failures)
        for event in h.corrective_events:
            ax1.plot(event.time, 0, 'rx', markersize=10, markeredgewidth=2)

        # Mark preventive maintenance
        for event in h.preventive_events:
            ax1.plot(event.time, event.hi_before, 'go', markersize=8, markeredgewidth=2)

    ax1.axhline(stats['tau'], color='orange', linestyle='--', linewidth=2.5,
               label=f'Threshold τ={stats["tau"]:.2f}')
    ax1.plot([], [], 'rx', markersize=10, markeredgewidth=2,
            label=f'Corrective (${stats["Cc"]:,.0f})')
    ax1.plot([], [], 'go', markersize=8, markeredgewidth=2,
            label=f'Preventive (${stats["Cp"]:,.0f})')

    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Health Index', fontsize=12, fontweight='bold')
    title_text = f'Asset Trajectories - Maintenance Events\n(Showing {len(plot_histories)} of {stats["n_assets"]} assets; see pie charts for full fleet counts)'
    ax1.set_title(title_text, fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, stats['horizon']])
    ax1.set_ylim([0, 1.05])

    # 2. Cost breakdown pie chart (top right)
    ax2 = fig.add_subplot(gs[0, 2])

    costs = [stats['total_corrective_cost'], stats['total_preventive_cost']]
    labels = [f'Corrective\n${stats["total_corrective_cost"]:,.0f}\n({stats["corrective_cost_ratio"]*100:.1f}%)',
              f'Preventive\n${stats["total_preventive_cost"]:,.0f}\n({stats["preventive_cost_ratio"]*100:.1f}%)']
    colors = ['#ff6b6b', '#51cf66']

    ax2.pie(costs, labels=labels, colors=colors, autopct='', startangle=90,
           textprops={'fontsize': 9, 'fontweight': 'bold'})
    ax2.set_title(f'Cost Breakdown - Full Fleet\nTotal: ${stats["total_cost"]:,.0f}',
                 fontsize=11, fontweight='bold', pad=10)

    # 3. Maintenance events pie chart (middle right)
    ax3 = fig.add_subplot(gs[1, 2])

    events = [stats['total_corrective_events'], stats['total_preventive_events']]
    labels_events = [f'Corrective\n{stats["total_corrective_events"]}\n({stats["corrective_event_ratio"]*100:.1f}%)',
                    f'Preventive\n{stats["total_preventive_events"]}\n({stats["preventive_event_ratio"]*100:.1f}%)']

    ax3.pie(events, labels=labels_events, colors=colors, autopct='', startangle=90,
           textprops={'fontsize': 9, 'fontweight': 'bold'})
    ax3.set_title(f'Maintenance Events - Full Fleet\nTotal: {stats["total_events"]}',
                 fontsize=11, fontweight='bold', pad=10)

    # 4. Cost distribution histogram (middle left)
    ax4 = fig.add_subplot(gs[2, 0])

    total_costs = [h.total_cost for h in histories]
    ax4.hist(total_costs, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(np.mean(total_costs), color='red', linestyle='--', linewidth=2.5,
               label=f'Mean=${np.mean(total_costs):,.0f}')
    ax4.set_xlabel('Total Cost per Asset', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax4.set_title(f'Distribution of Total Costs\n(All {stats["n_assets"]} Assets)',
                 fontsize=11, fontweight='bold', pad=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Maintenance count distribution (middle center)
    ax5 = fig.add_subplot(gs[2, 1])

    n_maintenances = [h.n_total for h in histories]
    ax5.hist(n_maintenances, bins=range(0, max(n_maintenances)+2),
            alpha=0.7, color='teal', edgecolor='black', align='left')
    ax5.axvline(np.mean(n_maintenances), color='red', linestyle='--', linewidth=2.5,
               label=f'Mean={np.mean(n_maintenances):.1f}')
    ax5.set_xlabel('Number of Maintenances', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax5.set_title(f'Distribution of Maintenance Events\n(All {stats["n_assets"]} Assets)',
                 fontsize=11, fontweight='bold', pad=8)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Corrective vs Preventive costs scatter (middle right)
    ax6 = fig.add_subplot(gs[2, 2])

    corrective_costs = [h.total_corrective_cost for h in histories]
    preventive_costs = [h.total_preventive_cost for h in histories]

    ax6.scatter(corrective_costs, preventive_costs, alpha=0.6, s=50,
               c='darkblue', edgecolors='black')
    ax6.set_xlabel('Corrective Cost ($)', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Preventive Cost ($)', fontsize=10, fontweight='bold')
    ax6.set_title('Corrective vs Preventive Costs\nPer Asset',
                 fontsize=11, fontweight='bold', pad=8)
    ax6.grid(True, alpha=0.3)

    # 7. Cumulative cost over time (bottom, spans 2 columns)
    ax7 = fig.add_subplot(gs[3, 0:2])

    # Collect all events across fleet
    all_events = []
    for h in histories:
        all_events.extend(h.corrective_events)
        all_events.extend(h.preventive_events)

    all_events.sort(key=lambda e: e.time)

    if all_events:
        times = [0] + [e.time for e in all_events]
        cumulative_costs = [0]
        cumulative_corrective = [0]
        cumulative_preventive = [0]

        for event in all_events:
            cumulative_costs.append(cumulative_costs[-1] + event.cost)
            if event.event_type == 'corrective':
                cumulative_corrective.append(cumulative_corrective[-1] + event.cost)
                cumulative_preventive.append(cumulative_preventive[-1])
            else:
                cumulative_preventive.append(cumulative_preventive[-1] + event.cost)
                cumulative_corrective.append(cumulative_corrective[-1])

        # Plot cumulative costs by type
        ax7.fill_between(times, 0, cumulative_corrective, alpha=0.5, color='#ff6b6b', label='Corrective')
        ax7.fill_between(times, cumulative_corrective, cumulative_costs, alpha=0.5, color='#51cf66', label='Preventive')
        ax7.plot(times, cumulative_costs, linewidth=2.5, color='darkgreen', label='Total')

        ax7.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Cumulative Cost ($)', fontsize=11, fontweight='bold')
        ax7.set_title('Fleet Cumulative Cost Over Time', fontsize=12, fontweight='bold', pad=8)
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)
        ax7.set_xlim([0, stats['horizon']])

    # 8. Summary statistics table (bottom right)
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.axis('off')

    summary_text = f"""
    COST ANALYSIS SUMMARY
    {'='*40}

    Configuration:
    • Assets: {stats['n_assets']}
    • Horizon: {stats['horizon']:.1f}
    • Threshold τ: {stats['tau']:.2f}
    • Confidence: {100*(1-stats['alpha']):.0f}%

    Costs:
    • Corrective (Cc): ${stats['Cc']:,.2f}
    • Preventive (Cp): ${stats['Cp']:,.2f}
    • Cost Ratio: {stats['Cc']/stats['Cp']:.2f}:1

    Total Fleet Costs:
    • Corrective: ${stats['total_corrective_cost']:,.0f}
    • Preventive: ${stats['total_preventive_cost']:,.0f}
    • TOTAL: ${stats['total_cost']:,.0f}

    Per Asset Averages:
    • Total cost: ${stats['avg_cost_per_asset']:,.2f}
    • Corrective: ${stats['avg_corrective_per_asset']:,.2f}
    • Preventive: ${stats['avg_preventive_per_asset']:,.2f}

    Events:
    • Failures: {stats['total_corrective_events']}
    • Preventive: {stats['total_preventive_events']}
    • Total: {stats['total_events']}

    Rates (per asset per time unit):
    • Failure rate: {stats['failure_rate']:.4f}
    • Maintenance rate: {stats['maintenance_rate']:.4f}
    • Cost rate: ${stats['cost_per_unit_time']:.2f}
    """

    ax8.text(0.05, 0.98, summary_text, transform=ax8.transAxes,
            fontsize=8.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    # No main suptitle to avoid overlap with subplot titles
    # Configuration info is already shown in the summary box

    # Save with high DPI
    try:
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
        print(f"✓ Visualization saved to '{save_path}'")
    except Exception as e:
        print(f"⚠ Could not save figure: {e}")

    return fig


def save_cost_results(histories: List[AssetMaintenanceHistory],
                     stats: dict,
                     filename: str = 'cost_analysis_results.csv'):
    """Save detailed cost results to CSV."""

    data = {
        'Asset_ID': [],
        'TTF': [],
        'b_parameter': [],
        'Total_Cost': [],
        'Corrective_Cost': [],
        'Preventive_Cost': [],
        'N_Corrective': [],
        'N_Preventive': [],
        'N_Total': []
    }

    for h in histories:
        data['Asset_ID'].append(h.asset_id)
        data['TTF'].append(h.ttf)
        data['b_parameter'].append(h.b_parameter)
        data['Total_Cost'].append(h.total_cost)
        data['Corrective_Cost'].append(h.total_corrective_cost)
        data['Preventive_Cost'].append(h.total_preventive_cost)
        data['N_Corrective'].append(h.n_corrective)
        data['N_Preventive'].append(h.n_preventive)
        data['N_Total'].append(h.n_total)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    # Also save summary stats
    summary_df = pd.DataFrame([stats])
    summary_filename = filename.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_filename, index=False)

    print(f"✓ Results saved to '{filename}'")
    print(f"✓ Summary saved to '{summary_filename}'")

    return df


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Interactive cost-based maintenance analysis."""

    print("\n" + "="*75)
    print(" "*12 + "COST-BASED PREVENTIVE MAINTENANCE ANALYSIS")
    print(" "*20 + "Economic Optimization System")
    print("="*75)

    print("\n📋 This system will:")
    print("  1. Simulate asset degradation over finite horizon H")
    print("  2. Track corrective maintenance (failures) at cost Cc")
    print("  3. Track preventive maintenance (inspections) at cost Cp")
    print("  4. Apply perfect maintenance (HI → 1) after each intervention")
    print("  5. Calculate total costs and optimize maintenance strategy")
    print("\n" + "="*75)

    try:
        # Configuration
        print("\n" + "="*75)
        print("STEP 1: BASIC CONFIGURATION")
        print("="*75)

        n_assets = int(input("\nNumber of assets [50]: ") or "50")
        H = float(input("Time horizon H [200]: ") or "200")
        tau = float(input("HI threshold τ [0.7]: ") or "0.7")
        alpha = float(input("Probability level α [0.05]: ") or "0.05")

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
            # beta_b = beta / p
            # eta_b = 1 / (eta^p)
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

        # Generate initial TTF for model fitting
        print("\n" + "="*75)
        print("STEP 4: RUNNING COST-BASED SIMULATION")
        print("="*75)

        print("\n⏳ Generating initial TTF data for model fitting...")
        b_samples = CostBasedMaintenanceScheduler._generate_b_samples(
            None, dist_type, params, n_assets
        )

        if dist_type == 'gamma':
            ttf_samples = 1 / b_samples
        else:
            ttf_samples = (1 / b_samples) ** (1/p)

        print(f"✓ TTF mean: {np.mean(ttf_samples):.2f}, std: {np.std(ttf_samples):.2f}")

        # Initialize scheduler
        scheduler = CostBasedMaintenanceScheduler(
            ttf_data=ttf_samples,
            tau=tau,
            alpha=alpha,
            Cc=Cc,
            Cp=Cp,
            H=H
        )

        # Simulate fleet
        print("\n⏳ Simulating fleet with maintenance cycles...")
        histories = scheduler.simulate_fleet_with_costs(
            n_assets=n_assets,
            dist_type=dist_type,
            params=params,
            p=p
        )

        # Compute statistics
        print("\n⏳ Computing fleet statistics...")
        stats = scheduler.compute_fleet_statistics(histories)

        # Print results
        print("\n" + "="*75)
        print("STEP 5: RESULTS SUMMARY")
        print("="*75)

        print(f"\n💰 COST ANALYSIS:")
        print(f"  • Total fleet cost: ${stats['total_cost']:,.2f}")
        print(f"  • Corrective cost: ${stats['total_corrective_cost']:,.2f} ({stats['corrective_cost_ratio']*100:.1f}%)")
        print(f"  • Preventive cost: ${stats['total_preventive_cost']:,.2f} ({stats['preventive_cost_ratio']*100:.1f}%)")
        print(f"  • Average cost per asset: ${stats['avg_cost_per_asset']:,.2f}")
        print(f"  • Cost per unit time: ${stats['cost_per_unit_time']:.2f}")

        print(f"\n📊 MAINTENANCE EVENTS:")
        print(f"  • Total failures (corrective): {stats['total_corrective_events']}")
        print(f"  • Total preventive maintenances: {stats['total_preventive_events']}")
        print(f"  • Total events: {stats['total_events']}")
        print(f"  • Failure rate: {stats['failure_rate']:.4f} per asset per time unit")

        print(f"\n💡 EFFICIENCY METRICS:")
        print(f"  • Preventive success rate: {stats['preventive_cost_ratio']*100:.1f}%")
        print(f"  • Average maintenances per asset: {stats['total_events']/n_assets:.2f}")

        # Save results
        print("\n" + "="*75)
        print("STEP 6: SAVING RESULTS")
        print("="*75)

        df = save_cost_results(histories, stats)

        # Visualize
        print("\n⏳ Creating visualizations...")
        fig = visualize_cost_analysis(histories, stats)

        print("\n" + "="*75)
        print("✅ COST ANALYSIS COMPLETE!")
        print("="*75)
        print("\n📁 Generated files:")
        print("  • cost_analysis_results.csv - Detailed asset data")
        print("  • cost_analysis_results_summary.csv - Fleet statistics")
        print("  • cost_analysis_results.png - Comprehensive visualization")
        print("\n💡 Next steps:")
        print("  • Try different α values to find optimal cost")
        print("  • Vary Cc/Cp ratio to see sensitivity")
        print("  • Compare with different threshold τ values")
        print("="*75 + "\n")

        plt.show()

        return scheduler, histories, stats, df

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
        return None, None, None, None
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    scheduler, histories, stats, df = main()
