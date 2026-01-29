"""
Maintenance Strategy Comparison Tool (Enhanced)
================================================

Compares scheduled (time-based) vs. predictive (condition-based) maintenance
to determine which strategy minimizes costs for your specific scenario.

Strategies Compared:
1. Scheduled Maintenance: Fixed intervals T (optimized)
2. Predictive Maintenance: Threshold τ + Probability α (both optimized via 2D search)

This script runs both optimizations and provides comprehensive comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
from pathlib import Path

# Try to import required modules
try:
    from scheduled_maintenance_analysis import (
        ScheduledMaintenanceScheduler,
        optimize_maintenance_interval,
    )
    SCHEDULED_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: scheduled_maintenance_analysis.py not found in current directory")
    print("   Please ensure scheduled_maintenance_analysis.py is in the same folder")
    SCHEDULED_AVAILABLE = False

try:
    from cost_based_maintenance_analysis import CostBasedMaintenanceScheduler
    PREDICTIVE_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: cost_based_maintenance_analysis.py not found in current directory")
    print("   Please ensure cost_based_maintenance_analysis.py is in the same folder")
    PREDICTIVE_AVAILABLE = False


def optimize_predictive_maintenance_2d(n_assets: int, H: float, Cc: float, Cp: float,
                                       dist_type: str, params: dict, p: float,
                                       tau_range: list = None,
                                       alpha_range: list = None) -> dict:
    """
    Optimize predictive maintenance strategy via 2D search over (τ, α).
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
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
    tau_range : list, optional
        List of τ values to test
    alpha_range : list, optional
        List of α values to test
    
    Returns:
    --------
    dict with optimal (τ, α) and associated costs
    """
    if not PREDICTIVE_AVAILABLE:
        raise ImportError("cost_based_maintenance_analysis module not available")
    
    if tau_range is None:
        tau_range = [0.60, 0.65, 0.70, 0.75, 0.80]
    
    if alpha_range is None:
        alpha_range = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    
    print("\n" + "="*80)
    print("OPTIMIZING PREDICTIVE MAINTENANCE (2D: τ and α)")
    print("="*80)
    print(f"\nTesting τ values: {tau_range}")
    print(f"Testing α values: {alpha_range}")
    print(f"Total combinations: {len(tau_range)} × {len(alpha_range)} = {len(tau_range)*len(alpha_range)}")
    print(f"Configuration: n={n_assets}, H={H}, Cc=${Cc:,.0f}, Cp=${Cp:,.0f}")
    print("\n⏳ Running 2D optimization (this may take several minutes)...")

    # Generate common b samples for consistency
    scheduler_temp = CostBasedMaintenanceScheduler.__new__(CostBasedMaintenanceScheduler)
    b_samples_base = scheduler_temp._generate_b_samples(dist_type, params, n_assets)

    if dist_type == 'gamma':
        ttf_samples_base = 1 / b_samples_base
    else:
        ttf_samples_base = (1 / b_samples_base) ** (1/p)

    results_grid = []
    total_iterations = len(tau_range) * len(alpha_range)
    current_iteration = 0

    for i, tau in enumerate(tau_range):
        for j, alpha in enumerate(alpha_range):
            current_iteration += 1
            print(f"\n[{current_iteration}/{total_iterations}] Testing τ={tau:.2f}, α={alpha:.3f}...", end=" ")

            try:
                # Create scheduler
                scheduler = CostBasedMaintenanceScheduler(
                    ttf_data=ttf_samples_base,
                    tau=tau,
                    alpha=alpha,
                    Cc=Cc,
                    Cp=Cp,
                    H=H
                )

                # Simulate fleet
                histories = scheduler.simulate_fleet_with_costs(
                    n_assets=n_assets,
                    dist_type=dist_type,
                    params=params,
                    p=p
                )

                # Compute stats
                stats = scheduler.compute_fleet_statistics(histories)

                results_grid.append({
                    'tau': tau,
                    'alpha': alpha,
                    'total_cost': stats['total_cost'],
                    'corrective_cost': stats['total_corrective_cost'],
                    'preventive_cost': stats['total_preventive_cost'],
                    'n_corrective': stats['total_corrective_events'],
                    'n_preventive': stats['total_preventive_events'],
                    'failure_rate': stats['failure_rate'],
                    'cost_per_unit_time': stats['cost_per_unit_time']
                })

                print(f"Cost = ${stats['total_cost']:,.0f}")

            except Exception as e:
                print(f"Error: {str(e)}")
                results_grid.append({
                    'tau': tau,
                    'alpha': alpha,
                    'total_cost': np.inf,
                    'corrective_cost': 0,
                    'preventive_cost': 0,
                    'n_corrective': 0,
                    'n_preventive': 0,
                    'failure_rate': 0,
                    'cost_per_unit_time': 0
                })

    # Find optimal (τ, α) pair
    valid_results = [r for r in results_grid if r['total_cost'] != np.inf]
    
    if not valid_results:
        raise RuntimeError("No valid results obtained from optimization")
    
    min_cost = min(r['total_cost'] for r in valid_results)
    optimal_result = [r for r in valid_results if r['total_cost'] == min_cost][0]

    optimal_tau = optimal_result['tau']
    optimal_alpha = optimal_result['alpha']

    print("\n" + "="*80)
    print("PREDICTIVE OPTIMIZATION RESULTS (2D)")
    print("="*80)
    print(f"\n🎯 OPTIMAL POLICY:")
    print(f"  • Optimal threshold: τ* = {optimal_tau:.2f}")
    print(f"  • Optimal probability: α* = {optimal_alpha:.3f}")
    print(f"  • Minimum total cost: ${min_cost:,.2f}")
    print(f"\n📊 AT OPTIMAL POINT:")
    print(f"  • Corrective cost: ${optimal_result['corrective_cost']:,.2f}")
    print(f"  • Preventive cost: ${optimal_result['preventive_cost']:,.2f}")
    print(f"  • Failures: {optimal_result['n_corrective']:.0f}")
    print(f"  • Preventive: {optimal_result['n_preventive']:.0f}")
    print(f"  • Failure rate: {optimal_result['failure_rate']:.4f}")

    return {
        'optimal_tau': optimal_tau,
        'optimal_alpha': optimal_alpha,
        'optimal_cost': min_cost,
        'tau': optimal_tau,  # For compatibility
        'alpha': optimal_alpha,  # For compatibility
        'corrective_cost': optimal_result['corrective_cost'],
        'preventive_cost': optimal_result['preventive_cost'],
        'n_corrective': optimal_result['n_corrective'],
        'n_preventive': optimal_result['n_preventive'],
        'failure_rate': optimal_result['failure_rate'],
        'results_grid': results_grid,
        'tau_range': tau_range,
        'alpha_range': alpha_range
    }


def comprehensive_comparison(n_assets: int, H: float, Cc: float, Cp: float,
                            dist_type: str, params: dict, p: float,
                            tau_range: list = None,
                            alpha_range: list = None) -> Dict:
    """
    Run complete comparison between scheduled and predictive strategies.
    
    Returns:
    --------
    dict with:
        - scheduled_results: Optimization results for scheduled maintenance
        - predictive_results: Optimization results for predictive maintenance (2D)
        - comparison_summary: Key comparison metrics
    """
    if not SCHEDULED_AVAILABLE or not PREDICTIVE_AVAILABLE:
        raise ImportError("Required modules not available. Please ensure both "
                         "scheduled_maintenance_analysis.py and "
                         "cost_based_maintenance_analysis.py are in the current directory")
    
    print("\n" + "="*80)
    print(" "*15 + "COMPREHENSIVE STRATEGY COMPARISON")
    print(" "*10 + "Scheduled vs. Predictive Maintenance Analysis")
    print("="*80)

    # 1. Optimize Scheduled Maintenance
    print("\n" + "🔧 "*25)
    print("PART 1: OPTIMIZING SCHEDULED MAINTENANCE")
    print("🔧 "*25)
    
    scheduled_results = optimize_maintenance_interval(
        n_assets=n_assets,
        H=H,
        Cc=Cc,
        Cp=Cp,
        dist_type=dist_type,
        params=params,
        p=p,
        T_range=[10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    )

    # 2. Optimize Predictive Maintenance (2D)
    print("\n" + "🎯 "*25)
    print("PART 2: OPTIMIZING PREDICTIVE MAINTENANCE (2D: τ and α)")
    print("🎯 "*25)
    
    predictive_results = optimize_predictive_maintenance_2d(
        n_assets=n_assets,
        H=H,
        Cc=Cc,
        Cp=Cp,
        dist_type=dist_type,
        params=params,
        p=p,
        tau_range=tau_range,
        alpha_range=alpha_range
    )

    # 3. Compute comparison metrics
    scheduled_cost = scheduled_results['optimal_cost']
    predictive_cost = predictive_results['optimal_cost']
    
    cost_difference = abs(scheduled_cost - predictive_cost)
    cost_savings_pct = cost_difference / max(scheduled_cost, predictive_cost) * 100
    
    if scheduled_cost < predictive_cost:
        winner = 'Scheduled'
        loser = 'Predictive'
    else:
        winner = 'Predictive'
        loser = 'Scheduled'

    comparison_summary = {
        'winner': winner,
        'loser': loser,
        'scheduled_cost': scheduled_cost,
        'predictive_cost': predictive_cost,
        'cost_difference': cost_difference,
        'savings_percent': cost_savings_pct,
        'scheduled_T': scheduled_results['optimal_T'],
        'predictive_tau': predictive_results['optimal_tau'],
        'predictive_alpha': predictive_results['optimal_alpha']
    }

    return {
        'scheduled': scheduled_results,
        'predictive': predictive_results,
        'summary': comparison_summary
    }


def visualize_full_comparison(comparison_results: Dict,
                              save_path: str = 'full_strategy_comparison.png'):
    """
    Create comprehensive visualization comparing both strategies.
    """
    scheduled = comparison_results['scheduled']
    predictive = comparison_results['predictive']
    summary = comparison_results['summary']

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.35,
                         top=0.96, bottom=0.05, left=0.05, right=0.98)

    # Get optimal results for each strategy
    sched_optimal = [r for r in scheduled['results'] 
                     if r['T'] == scheduled['optimal_T']][0]
    pred_optimal = predictive

    # 1. Total Cost Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    strategies = ['Scheduled\nMaintenance', 'Predictive\nMaintenance']
    costs = [summary['scheduled_cost'], summary['predictive_cost']]
    colors = ['steelblue' if summary['winner'] == 'Scheduled' else 'lightsteelblue',
              'orange' if summary['winner'] == 'Predictive' else 'wheat']
    
    bars = ax1.bar(strategies, costs, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax1.set_ylabel('Total Cost ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Total Cost Comparison', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add winner indicator
    ax1.text(0.5, 0.95, 
            f'🏆 WINNER: {summary["winner"]}\nSavings: ${summary["cost_difference"]:,.0f} ({summary["savings_percent"]:.1f}%)',
            transform=ax1.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3),
            fontsize=10, fontweight='bold')

    # 2. Cost Breakdown Comparison (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    
    x = np.arange(2)
    width = 0.35
    
    corrective = [sched_optimal['corrective_cost'], pred_optimal['corrective_cost']]
    preventive = [sched_optimal['preventive_cost'], pred_optimal['preventive_cost']]
    
    ax2.bar(x - width/2, corrective, width, label='Corrective', 
            color='#ff6b6b', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, preventive, width, label='Preventive', 
            color='#51cf66', alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Cost ($)', fontsize=11, fontweight='bold')
    ax2.set_title('Cost Breakdown', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Scheduled', 'Predictive'])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Event Counts Comparison (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    n_corrective = [sched_optimal['n_corrective'], pred_optimal['n_corrective']]
    n_preventive = [sched_optimal['n_preventive'], pred_optimal['n_preventive']]
    
    ax3.bar(x - width/2, n_corrective, width, label='Failures', 
            color='#ff6b6b', alpha=0.7, edgecolor='black')
    ax3.bar(x + width/2, n_preventive, width, label='Preventive', 
            color='#51cf66', alpha=0.7, edgecolor='black')
    
    ax3.set_ylabel('Number of Events', fontsize=11, fontweight='bold')
    ax3.set_title('Maintenance Events', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Scheduled', 'Predictive'])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Scheduled Optimization Curve (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    
    T_values = [r['T'] for r in scheduled['results']]
    sched_costs = [r['total_cost'] for r in scheduled['results']]
    
    ax4.plot(T_values, sched_costs, 'o-', linewidth=2.5, markersize=8, 
            color='steelblue')
    ax4.axvline(scheduled['optimal_T'], color='red', linestyle='--', 
               linewidth=2, label=f'Optimal T={scheduled["optimal_T"]:.0f}')
    ax4.set_xlabel('Maintenance Interval T', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Total Cost ($)', fontsize=10, fontweight='bold')
    ax4.set_title('Scheduled: Cost vs. Interval T', fontsize=12, fontweight='bold', pad=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Predictive 2D Heatmap (middle center and right - spans 2 columns)
    ax5 = fig.add_subplot(gs[1, 1:3])
    
    # Create cost matrix for heatmap
    tau_range = predictive['tau_range']
    alpha_range = predictive['alpha_range']
    cost_matrix = np.zeros((len(tau_range), len(alpha_range)))
    
    for r in predictive['results_grid']:
        if r['total_cost'] != np.inf:
            i = tau_range.index(r['tau'])
            j = alpha_range.index(r['alpha'])
            cost_matrix[i, j] = r['total_cost']
        else:
            i = tau_range.index(r['tau'])
            j = alpha_range.index(r['alpha'])
            cost_matrix[i, j] = np.nan
    
    # Plot heatmap
    im = ax5.imshow(cost_matrix, cmap='RdYlGn_r', aspect='auto',
                   extent=[alpha_range[0], alpha_range[-1],
                          tau_range[-1], tau_range[0]],
                   origin='upper')
    
    # Mark optimal point
    ax5.plot(predictive['optimal_alpha'], predictive['optimal_tau'], 
            'r*', markersize=25, markeredgecolor='white', markeredgewidth=3,
            label=f'Optimal: (τ={predictive["optimal_tau"]:.2f}, α={predictive["optimal_alpha"]:.3f})')
    
    ax5.set_xlabel('Probability Level α', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Threshold τ', fontsize=10, fontweight='bold')
    ax5.set_title('Predictive: Cost Heatmap (τ, α) - 2D Optimization', 
                 fontsize=12, fontweight='bold', pad=8)
    ax5.legend(fontsize=9, loc='upper right')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax5, label='Total Cost ($)')

    # 6. Cost Components Breakdown (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    
    categories = ['Corrective\nCost', 'Preventive\nCost']
    scheduled_breakdown = [sched_optimal['corrective_cost'], 
                          sched_optimal['preventive_cost']]
    predictive_breakdown = [pred_optimal['corrective_cost'], 
                           pred_optimal['preventive_cost']]
    
    x_cat = np.arange(len(categories))
    width = 0.35
    
    ax6.bar(x_cat - width/2, scheduled_breakdown, width, label='Scheduled',
           color='steelblue', alpha=0.7, edgecolor='black')
    ax6.bar(x_cat + width/2, predictive_breakdown, width, label='Predictive',
           color='orange', alpha=0.7, edgecolor='black')
    
    ax6.set_ylabel('Cost ($)', fontsize=10, fontweight='bold')
    ax6.set_title('Cost Components Detail', fontsize=12, fontweight='bold', pad=8)
    ax6.set_xticks(x_cat)
    ax6.set_xticklabels(categories)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Failure Rate Comparison (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    
    failure_rates = [sched_optimal['failure_rate'], pred_optimal['failure_rate']]
    
    bars = ax7.bar(strategies, failure_rates, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax7.set_ylabel('Failure Rate\n(per asset per time)', fontsize=10, fontweight='bold')
    ax7.set_title('Failure Rate Comparison', fontsize=12, fontweight='bold', pad=8)
    ax7.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, failure_rates):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 8. Summary Text Box (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = f"""
    COMPARISON SUMMARY
    {'='*45}
    
    🏆 WINNER: {summary['winner']} Maintenance
    
    SCHEDULED MAINTENANCE:
    • Optimal interval: T = {summary['scheduled_T']:.0f}
    • Total cost: ${summary['scheduled_cost']:,.0f}
    • Failures: {sched_optimal['n_corrective']:.0f}
    • Preventive: {sched_optimal['n_preventive']:.0f}
    
    PREDICTIVE MAINTENANCE (2D):
    • Optimal threshold: τ* = {summary['predictive_tau']:.2f}
    • Optimal probability: α* = {summary['predictive_alpha']:.3f}
    • Total cost: ${summary['predictive_cost']:,.0f}
    • Failures: {pred_optimal['n_corrective']:.0f}
    • Preventive: {pred_optimal['n_preventive']:.0f}
    
    COST SAVINGS:
    • Absolute: ${summary['cost_difference']:,.0f}
    • Percentage: {summary['savings_percent']:.1f}%
    
    RECOMMENDATION:
    Use {summary['winner']} maintenance
    for this configuration.
    
    Note: Predictive used full 2D
    optimization over (τ, α) space
    to find global optimum.
    """
    
    ax8.text(0.05, 0.98, summary_text, transform=ax8.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', 
                     facecolor='gold' if summary['winner'] == 'Scheduled' else 'orange', 
                     alpha=0.2))

    plt.suptitle('Comprehensive Maintenance Strategy Comparison\n(Predictive: Full 2D Optimization)',
                fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"✓ Full comparison visualization saved to '{save_path}'")
    
    return fig


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Interactive comparison between scheduled and predictive maintenance."""

    print("\n" + "="*80)
    print(" "*15 + "MAINTENANCE STRATEGY COMPARISON TOOL")
    print(" "*8 + "Scheduled (Time-Based) vs. Predictive (Condition-Based)")
    print("="*80)

    print("\n📋 This tool will:")
    print("  1. Optimize SCHEDULED maintenance (find optimal interval T)")
    print("  2. Optimize PREDICTIVE maintenance (find optimal τ AND α via 2D search)")
    print("  3. Compare total costs and recommend the best strategy")
    print("  4. Generate comprehensive visualization and reports")
    print("\n" + "="*80)

    # Check dependencies
    if not SCHEDULED_AVAILABLE or not PREDICTIVE_AVAILABLE:
        print("\n❌ ERROR: Required modules not found!")
        print("\nPlease ensure the following files are in the current directory:")
        if not SCHEDULED_AVAILABLE:
            print("  ✗ scheduled_maintenance_analysis.py")
        if not PREDICTIVE_AVAILABLE:
            print("  ✗ cost_based_maintenance_analysis.py")
        print("\nExiting...")
        return None

    try:
        # Configuration
        print("\n" + "="*80)
        print("STEP 1: BASIC CONFIGURATION")
        print("="*80)

        n_assets = int(input("\nNumber of assets [50]: ") or "50")
        H = float(input("Time horizon H [200]: ") or "200")

        print("\n" + "="*80)
        print("STEP 2: COST PARAMETERS")
        print("="*80)

        Cc = float(input("\nCorrective maintenance cost Cc [1000]: ") or "1000")
        Cp = float(input("Preventive maintenance cost Cp [100]: ") or "100")

        print(f"\n  ✓ Cost ratio Cc/Cp = {Cc/Cp:.2f}:1")

        # Distribution
        print("\n" + "="*80)
        print("STEP 3: DEGRADATION MODEL")
        print("="*80)

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

        # Predictive maintenance ranges
        print("\n" + "="*80)
        print("STEP 4: PREDICTIVE MAINTENANCE OPTIMIZATION RANGES")
        print("="*80)
        
        print("\nPredictive maintenance will optimize BOTH τ and α")
        use_default = input("Use default ranges for (τ, α)? [Y/n]: ").strip().lower()
        
        if use_default == 'n':
            tau_min = float(input("Minimum τ [0.60]: ") or "0.60")
            tau_max = float(input("Maximum τ [0.80]: ") or "0.80")
            tau_steps = int(input("Number of τ values [5]: ") or "5")
            tau_range = list(np.linspace(tau_min, tau_max, tau_steps))
            
            alpha_min = float(input("Minimum α [0.01]: ") or "0.01")
            alpha_max = float(input("Maximum α [0.20]: ") or "0.20")
            alpha_steps = int(input("Number of α values [7]: ") or "7")
            alpha_range = list(np.linspace(alpha_min, alpha_max, alpha_steps))
        else:
            tau_range = [0.60, 0.65, 0.70, 0.75, 0.80]
            alpha_range = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
        
        print(f"\n  ✓ τ range: {[f'{t:.2f}' for t in tau_range]}")
        print(f"  ✓ α range: {[f'{a:.3f}' for a in alpha_range]}")
        print(f"  ✓ Predictive combinations: {len(tau_range)} × {len(alpha_range)} = {len(tau_range)*len(alpha_range)}")

        # Run comparison
        print("\n" + "="*80)
        print("STEP 5: RUNNING COMPREHENSIVE COMPARISON")
        print("="*80)
        print("\n⏳ This will take several minutes...\n")

        comparison_results = comprehensive_comparison(
            n_assets=n_assets,
            H=H,
            Cc=Cc,
            Cp=Cp,
            dist_type=dist_type,
            params=params,
            p=p,
            tau_range=tau_range,
            alpha_range=alpha_range
        )

        # Visualize
        print("\n⏳ Creating comprehensive visualization...")
        fig = visualize_full_comparison(comparison_results)

        # Save results
        summary = comparison_results['summary']
        
        # Save detailed comparison
        comparison_data = {
            'Strategy': ['Scheduled', 'Predictive'],
            'Winner': [summary['winner'] == 'Scheduled', summary['winner'] == 'Predictive'],
            'Total_Cost': [summary['scheduled_cost'], summary['predictive_cost']],
            'Optimal_Parameter': [f"T={summary['scheduled_T']:.0f}", 
                                 f"τ={summary['predictive_tau']:.2f}, α={summary['predictive_alpha']:.3f}"],
            'Corrective_Events': [
                [r for r in comparison_results['scheduled']['results'] 
                 if r['T'] == summary['scheduled_T']][0]['n_corrective'],
                comparison_results['predictive']['n_corrective']
            ],
            'Preventive_Events': [
                [r for r in comparison_results['scheduled']['results'] 
                 if r['T'] == summary['scheduled_T']][0]['n_preventive'],
                comparison_results['predictive']['n_preventive']
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('strategy_comparison_results.csv', index=False)
        
        # Save predictive 2D results
        predictive_df = pd.DataFrame(comparison_results['predictive']['results_grid'])
        predictive_df.to_csv('predictive_2d_optimization_results.csv', index=False)

        print("\n" + "="*80)
        print("✅ STRATEGY COMPARISON COMPLETE!")
        print("="*80)
        
        print("\n🏆 RESULTS:")
        print(f"  • Winner: {summary['winner']} Maintenance")
        print(f"  • Cost Savings: ${summary['cost_difference']:,.0f} ({summary['savings_percent']:.1f}%)")
        print(f"\n  Scheduled: T* = {summary['scheduled_T']:.0f}, Cost = ${summary['scheduled_cost']:,.0f}")
        print(f"  Predictive: τ* = {summary['predictive_tau']:.2f}, α* = {summary['predictive_alpha']:.3f}, Cost = ${summary['predictive_cost']:,.0f}")
        
        print("\n📁 Generated files:")
        print("  • full_strategy_comparison.png - Comprehensive visualization with 2D heatmap")
        print("  • strategy_comparison_results.csv - Comparison summary")
        print("  • predictive_2d_optimization_results.csv - Full (τ, α) grid results")
        
        print("\n💡 Recommendation:")
        print(f"  Use {summary['winner'].upper()} MAINTENANCE for this configuration")
        print(f"  to minimize total maintenance costs.")
        
        print("\n📊 Key Insight:")
        print(f"  • Predictive maintenance was optimized over full (τ, α) space")
        print(f"  • Global optimum found at (τ*={summary['predictive_tau']:.2f}, α*={summary['predictive_alpha']:.3f})")
        if summary['winner'] == 'Scheduled':
            print("  • Despite 2D optimization, scheduled maintenance is still more cost-effective")
            print("  • This suggests low degradation variability or high inspection costs")
        else:
            print("  • Predictive maintenance's flexibility provides significant cost savings")
            print("  • Adapting to asset condition is more valuable than fixed intervals")
        
        print("="*80 + "\n")

        plt.show()

        return comparison_results

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
