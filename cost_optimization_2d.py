"""
Enhanced Cost Optimization: 2D Parameter Sweep
===============================================

Optimizes both threshold τ and probability level α simultaneously to find
the global minimum cost maintenance policy.

Features:
- 2D grid search over (τ, α) space
- 3D surface plots and contour maps
- Heatmap visualization
- Trade-off analysis
- Optimal policy identification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from cost_based_maintenance_analysis import CostBasedMaintenanceScheduler
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def optimize_tau_and_alpha(n_assets: int, H: float, Cc: float, Cp: float,
                           dist_type: str, params: dict, p: float,
                           tau_range: List[float] = None,
                           alpha_range: List[float] = None,
                           n_simulations: int = 1) -> dict:
    """
    Perform 2D optimization over both τ and α to minimize total cost.

    Parameters:
    -----------
    n_assets : int
        Number of assets to simulate
    H : float
        Time horizon
    Cc : float
        Corrective maintenance cost
    Cp : float
        Preventive maintenance cost
    dist_type : str
        Distribution type ('weibull', 'gamma', etc.)
    params : dict
        Distribution parameters
    p : float
        Power parameter for HI model
    tau_range : List[float], optional
        List of τ values to test (default: [0.60, 0.65, 0.70, 0.75, 0.80])
    alpha_range : List[float], optional
        List of α values to test (default: [0.01, 0.02, 0.05, 0.08, 0.10, 0.15])
    n_simulations : int, optional
        Number of simulation replications per (τ, α) pair for averaging (default: 1)

    Returns:
    --------
    dict with:
        - optimal_tau: Best threshold value
        - optimal_alpha: Best probability level
        - optimal_cost: Minimum total cost
        - results_grid: Complete results for all (τ, α) pairs
        - cost_matrix: 2D array of costs for heatmap
    """
    if tau_range is None:
        tau_range = [0.60, 0.65, 0.70, 0.75, 0.80]

    if alpha_range is None:
        alpha_range = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15]

    print("\n" + "="*80)
    print("2D OPTIMIZATION: THRESHOLD (τ) AND PROBABILITY LEVEL (α)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  • Assets: {n_assets}")
    print(f"  • Horizon: {H}")
    print(f"  • Corrective cost: Cc = ${Cc:,.0f}")
    print(f"  • Preventive cost: Cp = ${Cp:,.0f}")
    print(f"  • Cost ratio: {Cc/Cp:.1f}:1")
    print(f"  • Distribution: {dist_type}")
    print(f"\nParameter ranges:")
    print(f"  • τ values: {tau_range}")
    print(f"  • α values: {alpha_range}")
    print(f"  • Total combinations: {len(tau_range)} × {len(alpha_range)} = {len(tau_range)*len(alpha_range)}")
    print(f"  • Simulations per point: {n_simulations}")
    print(f"\n⏳ Starting optimization (this may take a few minutes)...\n")

    # Generate common b samples for consistency across all runs
    scheduler_temp = CostBasedMaintenanceScheduler.__new__(CostBasedMaintenanceScheduler)
    b_samples_base = scheduler_temp._generate_b_samples(dist_type, params, n_assets)

    if dist_type == 'gamma':
        ttf_samples_base = 1 / b_samples_base
    else:
        ttf_samples_base = (1 / b_samples_base) ** (1/p)

    # Initialize results storage
    results_grid = []
    cost_matrix = np.zeros((len(tau_range), len(alpha_range)))

    total_iterations = len(tau_range) * len(alpha_range)
    current_iteration = 0

    # Grid search over all (τ, α) combinations
    for i, tau in enumerate(tau_range):
        for j, alpha in enumerate(alpha_range):
            current_iteration += 1

            print(f"[{current_iteration}/{total_iterations}] Testing τ={tau:.2f}, α={alpha:.3f}...", end=" ")

            # Run multiple simulations and average if requested
            costs_replications = []
            stats_replications = []

            for rep in range(n_simulations):
                # Use same TTF samples but potentially different random degradation
                try:
                    scheduler = CostBasedMaintenanceScheduler(
                        ttf_data=ttf_samples_base,
                        tau=tau,
                        alpha=alpha,
                        Cc=Cc,
                        Cp=Cp,
                        H=H
                    )

                    histories = scheduler.simulate_fleet_with_costs(
                        n_assets=n_assets,
                        dist_type=dist_type,
                        params=params,
                        p=p
                    )

                    stats = scheduler.compute_fleet_statistics(histories)
                    costs_replications.append(stats['total_cost'])
                    stats_replications.append(stats)

                except Exception as e:
                    print(f"\n  ⚠️ Error at τ={tau}, α={alpha}: {str(e)}")
                    costs_replications.append(np.nan)
                    stats_replications.append(None)

            # Average across replications
            avg_cost = np.nanmean(costs_replications)

            if len(stats_replications) > 0 and stats_replications[0] is not None:
                avg_stats = {
                    'total_cost': avg_cost,
                    'corrective_cost': np.mean([s['total_corrective_cost'] for s in stats_replications if s]),
                    'preventive_cost': np.mean([s['total_preventive_cost'] for s in stats_replications if s]),
                    'n_corrective': np.mean([s['total_corrective_events'] for s in stats_replications if s]),
                    'n_preventive': np.mean([s['total_preventive_events'] for s in stats_replications if s]),
                    'failure_rate': np.mean([s['failure_rate'] for s in stats_replications if s]),
                    'cost_per_unit_time': np.mean([s['cost_per_unit_time'] for s in stats_replications if s])
                }
            else:
                avg_stats = None

            # Store results
            result = {
                'tau': tau,
                'alpha': alpha,
                'total_cost': avg_cost,
                'tau_index': i,
                'alpha_index': j
            }

            if avg_stats:
                result.update(avg_stats)

            results_grid.append(result)
            cost_matrix[i, j] = avg_cost

            print(f"Cost = ${avg_cost:,.0f}")

    # Find optimal (τ, α) pair
    min_cost = np.nanmin([r['total_cost'] for r in results_grid])
    optimal_result = [r for r in results_grid if r['total_cost'] == min_cost][0]

    optimal_tau = optimal_result['tau']
    optimal_alpha = optimal_result['alpha']

    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"\n🎯 OPTIMAL POLICY:")
    print(f"  • Optimal threshold: τ* = {optimal_tau:.2f}")
    print(f"  • Optimal probability: α* = {optimal_alpha:.3f}")
    print(f"  • Minimum total cost: ${min_cost:,.2f}")

    if 'corrective_cost' in optimal_result:
        print(f"\n📊 AT OPTIMAL POINT:")
        print(f"  • Corrective cost: ${optimal_result['corrective_cost']:,.2f}")
        print(f"  • Preventive cost: ${optimal_result['preventive_cost']:,.2f}")
        print(f"  • Failures: {optimal_result['n_corrective']:.1f}")
        print(f"  • Preventive: {optimal_result['n_preventive']:.1f}")
        print(f"  • Failure rate: {optimal_result['failure_rate']:.4f}")

    # Calculate improvement vs corners
    max_cost = np.nanmax([r['total_cost'] for r in results_grid])
    improvement = (max_cost - min_cost) / max_cost * 100

    print(f"\n💰 COST SAVINGS:")
    print(f"  • Worst case cost: ${max_cost:,.2f}")
    print(f"  • Best case cost: ${min_cost:,.2f}")
    print(f"  • Potential savings: ${max_cost - min_cost:,.2f} ({improvement:.1f}%)")

    return {
        'optimal_tau': optimal_tau,
        'optimal_alpha': optimal_alpha,
        'optimal_cost': min_cost,
        'results_grid': results_grid,
        'cost_matrix': cost_matrix,
        'tau_range': tau_range,
        'alpha_range': alpha_range,
        'optimal_result': optimal_result
    }


def visualize_2d_optimization(opt_results: dict,
                              save_path: str = '2d_optimization_results.png'):
    """
    Create comprehensive visualization of 2D optimization results.

    Includes:
    - 3D surface plot
    - Contour plot with optimal point
    - Heatmap
    - Marginal cost curves (varying τ at fixed α and vice versa)
    """
    results = opt_results['results_grid']
    tau_range = opt_results['tau_range']
    alpha_range = opt_results['alpha_range']
    cost_matrix = opt_results['cost_matrix']
    optimal_tau = opt_results['optimal_tau']
    optimal_alpha = opt_results['optimal_alpha']
    optimal_cost = opt_results['optimal_cost']

    # Create meshgrid for plotting
    TAU, ALPHA = np.meshgrid(tau_range, alpha_range)
    COST = cost_matrix.T  # Transpose for correct orientation

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # 1. 3D Surface Plot (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')

    surf = ax1.plot_surface(TAU, ALPHA, COST, cmap=cm.viridis,
                           linewidth=0, antialiased=True, alpha=0.8)

    # Mark optimal point
    opt_idx_tau = tau_range.index(optimal_tau)
    opt_idx_alpha = alpha_range.index(optimal_alpha)
    ax1.scatter([optimal_tau], [optimal_alpha], [optimal_cost],
               color='red', s=200, marker='*', edgecolors='black', linewidths=2,
               label=f'Optimal: (τ={optimal_tau:.2f}, α={optimal_alpha:.3f})')

    ax1.set_xlabel('Threshold τ', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability α', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Cost Surface: C(τ, α)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)

    # Add colorbar
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 2. Contour Plot (top right)
    ax2 = fig.add_subplot(gs[0, 2])

    contour = ax2.contourf(TAU, ALPHA, COST, levels=15, cmap='viridis')
    contour_lines = ax2.contour(TAU, ALPHA, COST, levels=10, colors='white',
                                linewidths=0.5, alpha=0.4)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')

    # Mark optimal point
    ax2.plot(optimal_tau, optimal_alpha, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2,
            label=f'Optimal\n(τ={optimal_tau:.2f}, α={optimal_alpha:.3f})')

    ax2.set_xlabel('Threshold τ', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Probability α', fontsize=11, fontweight='bold')
    ax2.set_title('Cost Contours', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.colorbar(contour, ax=ax2)

    # 3. Heatmap (middle right)
    ax3 = fig.add_subplot(gs[1, 2])

    im = ax3.imshow(COST, cmap='RdYlGn_r', aspect='auto',
                   extent=[tau_range[0], tau_range[-1],
                          alpha_range[0], alpha_range[-1]],
                   origin='lower')

    # Mark optimal
    ax3.plot(optimal_tau, optimal_alpha, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2)

    # Add text annotations for costs
    for r in results:
        ax3.text(r['tau'], r['alpha'], f"{r['total_cost']:.0f}",
                ha='center', va='center', fontsize=7, color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    ax3.set_xlabel('Threshold τ', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Probability α', fontsize=11, fontweight='bold')
    ax3.set_title('Cost Heatmap', fontsize=12, fontweight='bold')

    fig.colorbar(im, ax=ax3, label='Total Cost ($)')

    # 4. Marginal cost: varying τ at optimal α (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])

    costs_at_opt_alpha = [r['total_cost'] for r in results if r['alpha'] == optimal_alpha]
    tau_at_opt_alpha = [r['tau'] for r in results if r['alpha'] == optimal_alpha]

    ax4.plot(tau_at_opt_alpha, costs_at_opt_alpha, 'o-', linewidth=2.5,
            markersize=10, color='darkblue')
    ax4.axvline(optimal_tau, color='red', linestyle='--', linewidth=2,
               label=f'Optimal τ={optimal_tau:.2f}')
    ax4.axhline(optimal_cost, color='green', linestyle=':', linewidth=1.5, alpha=0.5)

    ax4.set_xlabel('Threshold τ', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Total Cost ($)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Cost vs τ (at α={optimal_alpha:.3f})', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Marginal cost: varying α at optimal τ (bottom center)
    ax5 = fig.add_subplot(gs[2, 1])

    costs_at_opt_tau = [r['total_cost'] for r in results if r['tau'] == optimal_tau]
    alpha_at_opt_tau = [r['alpha'] for r in results if r['tau'] == optimal_tau]

    ax5.plot(alpha_at_opt_tau, costs_at_opt_tau, 's-', linewidth=2.5,
            markersize=10, color='darkgreen')
    ax5.axvline(optimal_alpha, color='red', linestyle='--', linewidth=2,
               label=f'Optimal α={optimal_alpha:.3f}')
    ax5.axhline(optimal_cost, color='green', linestyle=':', linewidth=1.5, alpha=0.5)

    ax5.set_xlabel('Probability α', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Total Cost ($)', fontsize=11, fontweight='bold')
    ax5.set_title(f'Cost vs α (at τ={optimal_tau:.2f})', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics table (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    # Calculate some statistics
    cost_range = np.nanmax([r['total_cost'] for r in results]) - np.nanmin([r['total_cost'] for r in results])
    cost_std = np.nanstd([r['total_cost'] for r in results])

    summary_text = f"""
    OPTIMIZATION SUMMARY
    {'='*40}

    Optimal Policy:
    • τ* = {optimal_tau:.2f}
    • α* = {optimal_alpha:.3f}
    • Cost* = ${optimal_cost:,.0f}

    Search Space:
    • τ range: [{min(tau_range):.2f}, {max(tau_range):.2f}]
    • α range: [{min(alpha_range):.3f}, {max(alpha_range):.3f}]
    • Points tested: {len(results)}

    Cost Statistics:
    • Min cost: ${np.nanmin([r['total_cost'] for r in results]):,.0f}
    • Max cost: ${np.nanmax([r['total_cost'] for r in results]):,.0f}
    • Range: ${cost_range:,.0f}
    • Std dev: ${cost_std:,.0f}

    Sensitivity:
    • ∂C/∂τ at opt: {"High" if cost_range > optimal_cost * 0.2 else "Moderate"}
    • ∂C/∂α at opt: {"High" if cost_std > optimal_cost * 0.1 else "Moderate"}
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    # Overall title
    fig.suptitle('2D Optimization: Threshold (τ) and Probability Level (α)',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to '{save_path}'")

    return fig


def analyze_pareto_frontier(opt_results: dict):
    """
    Analyze the Pareto frontier: trade-off between failure cost and preventive cost.
    """
    results = opt_results['results_grid']

    # Extract costs
    data = []
    for r in results:
        if 'corrective_cost' in r and 'preventive_cost' in r:
            data.append({
                'tau': r['tau'],
                'alpha': r['alpha'],
                'corrective_cost': r['corrective_cost'],
                'preventive_cost': r['preventive_cost'],
                'total_cost': r['total_cost']
            })

    if not data:
        print("⚠️  Insufficient data for Pareto analysis")
        return None

    df = pd.DataFrame(data)

    # Create Pareto plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scatter = ax.scatter(df['corrective_cost'], df['preventive_cost'],
                        c=df['total_cost'], s=200, cmap='viridis',
                        edgecolors='black', linewidths=1, alpha=0.7)

    # Mark optimal point
    optimal_tau = opt_results['optimal_tau']
    optimal_alpha = opt_results['optimal_alpha']
    opt_data = df[(df['tau'] == optimal_tau) & (df['alpha'] == optimal_alpha)]

    if not opt_data.empty:
        ax.scatter(opt_data['corrective_cost'], opt_data['preventive_cost'],
                  color='red', s=400, marker='*', edgecolors='white',
                  linewidths=2, label='Optimal', zorder=5)

    # Annotate points with (τ, α)
    for _, row in df.iterrows():
        ax.annotate(f"({row['tau']:.2f},{row['alpha']:.2f})",
                   (row['corrective_cost'], row['preventive_cost']),
                   fontsize=7, alpha=0.6)

    ax.set_xlabel('Corrective Cost ($)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Preventive Cost ($)', fontsize=13, fontweight='bold')
    ax.set_title('Pareto Frontier: Corrective vs Preventive Costs',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax, label='Total Cost ($)')

    plt.tight_layout()
    plt.savefig('pareto_frontier.png', dpi=300, bbox_inches='tight')
    print("✓ Pareto frontier plot saved to 'pareto_frontier.png'")

    return fig


def save_optimization_results(opt_results: dict,
                              filename: str = '2d_optimization_results.csv'):
    """Save complete optimization results to CSV."""

    results = opt_results['results_grid']
    df = pd.DataFrame(results)
    df = df.sort_values('total_cost')

    df.to_csv(filename, index=False)
    print(f"✓ Optimization results saved to '{filename}'")

    # Also save a summary
    summary = {
        'Optimal_Tau': [opt_results['optimal_tau']],
        'Optimal_Alpha': [opt_results['optimal_alpha']],
        'Optimal_Cost': [opt_results['optimal_cost']],
        'Tau_Range': [str(opt_results['tau_range'])],
        'Alpha_Range': [str(opt_results['alpha_range'])],
        'N_Points_Tested': [len(results)]
    }

    if 'optimal_result' in opt_results:
        opt_res = opt_results['optimal_result']
        if 'corrective_cost' in opt_res:
            summary['Optimal_Corrective_Cost'] = [opt_res['corrective_cost']]
            summary['Optimal_Preventive_Cost'] = [opt_res['preventive_cost']]
            summary['Optimal_N_Failures'] = [opt_res['n_corrective']]
            summary['Optimal_N_Preventive'] = [opt_res['n_preventive']]

    summary_df = pd.DataFrame(summary)
    summary_filename = filename.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_filename, index=False)
    print(f"✓ Summary saved to '{summary_filename}'")

    return df


# ============================================================================
# COMPLETE EXAMPLE
# ============================================================================

def example_2d_optimization():
    """
    Complete example: optimize both τ and α simultaneously.
    """
    print("\n" + "="*80)
    print("COMPLETE 2D OPTIMIZATION EXAMPLE")
    print("="*80)

    # Configuration
    n_assets = 50
    H = 200
    Cc = 1000
    Cp = 100

    dist_type = 'weibull'
    params = {'shape': 2.5, 'scale': 0.015}
    p = 3.0

    print("\nConfiguration:")
    print(f"  • Assets: {n_assets}")
    print(f"  • Horizon: {H}")
    print(f"  • Corrective cost: Cc = ${Cc:,.0f}")
    print(f"  • Preventive cost: Cp = ${Cp:,.0f}")
    print(f"  • Distribution: {dist_type} (shape={params['shape']}, scale={params['scale']})")
    print(f"  • Power: p = {p}")

    # Run 2D optimization
    opt_results = optimize_tau_and_alpha(
        n_assets=n_assets,
        H=H,
        Cc=Cc,
        Cp=Cp,
        dist_type=dist_type,
        params=params,
        p=p,
        tau_range=[0.60, 0.65, 0.70, 0.75, 0.80],
        alpha_range=[0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20],
        n_simulations=1  # Increase for more robust results
    )

    # Visualize
    fig1 = visualize_2d_optimization(opt_results)

    # Pareto analysis
    fig2 = analyze_pareto_frontier(opt_results)

    # Save results
    df = save_optimization_results(opt_results)

    print("\n" + "="*80)
    print("✅ 2D OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\n📁 Generated files:")
    print("  • 2d_optimization_results.png - Comprehensive visualization")
    print("  • pareto_frontier.png - Trade-off analysis")
    print("  • 2d_optimization_results.csv - Complete results")
    print("  • 2d_optimization_results_summary.csv - Optimal policy")

    print("\n💡 Key Insights:")
    print(f"  • Optimal threshold: τ* = {opt_results['optimal_tau']:.2f}")
    print(f"  • Optimal probability: α* = {opt_results['optimal_alpha']:.3f}")
    print(f"  • Minimum cost: ${opt_results['optimal_cost']:,.2f}")
    print(f"  • Use this policy for lowest total maintenance cost!")

    print("\n📊 Next steps:")
    print("  1. Review the 3D surface plot to understand cost landscape")
    print("  2. Check contour plot for sensitivity near optimal point")
    print("  3. Examine Pareto frontier for trade-offs")
    print("  4. Test optimal policy on validation data")
    print("="*80 + "\n")

    plt.show()

    return opt_results, df


if __name__ == "__main__":
    try:
        opt_results, df = example_2d_optimization()
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
