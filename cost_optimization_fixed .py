"""
Cost Optimization for Preventive Maintenance
=============================================

This script performs sensitivity analysis and optimization to find:
- Optimal α (probability level) that minimizes total cost
- Optimal τ (threshold) for given cost structure
- Trade-off analysis between Cc and Cp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cost_based_maintenance_analysis import CostBasedMaintenanceScheduler
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


def optimize_alpha(n_assets: int, H: float, tau: float, Cc: float, Cp: float,
                   dist_type: str, params: dict, p: float,
                   alpha_range: List[float] = None) -> dict:
    """
    Find optimal α that minimizes total cost.

    Parameters:
    -----------
    n_assets : int
        Number of assets to simulate
    H : float
        Time horizon
    tau : float
        Threshold value
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
    alpha_range : List[float], optional
        List of α values to test (default: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20])

    Returns:
    --------
    dict with optimization results
    """
    if alpha_range is None:
        alpha_range = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    print("\n" + "="*70)
    print("OPTIMIZING α (PROBABILITY LEVEL)")
    print("="*70)
    print(f"\nTesting α values: {alpha_range}")
    print(f"Configuration: n={n_assets}, H={H}, τ={tau}, Cc=${Cc:,.0f}, Cp=${Cp:,.0f}")
    print("\n⏳ Running simulations...")

    results = []

    for i, alpha in enumerate(alpha_range):
        print(f"\n[{i+1}/{len(alpha_range)}] Testing α = {alpha:.3f}...")

        # Generate TTF data
        scheduler_temp = CostBasedMaintenanceScheduler.__new__(CostBasedMaintenanceScheduler)
        b_samples = scheduler_temp._generate_b_samples(dist_type, params, n_assets)

        if dist_type == 'gamma':
            ttf_samples = 1 / b_samples
        else:
            ttf_samples = (1 / b_samples) ** (1/p)

        # Create scheduler
        scheduler = CostBasedMaintenanceScheduler(
            ttf_data=ttf_samples,
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

        results.append({
            'alpha': alpha,
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
    optimal_alpha = results[min_idx]['alpha']
    optimal_cost = results[min_idx]['total_cost']

    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\n✓ Optimal α = {optimal_alpha:.3f}")
    print(f"✓ Minimum total cost = ${optimal_cost:,.2f}")
    print(f"✓ Cost reduction vs α=0.05: {(1 - optimal_cost/[r['total_cost'] for r in results if r['alpha']==0.05][0])*100:.1f}%")

    return {
        'optimal_alpha': optimal_alpha,
        'optimal_cost': optimal_cost,
        'results': results,
        'alpha_range': alpha_range
    }


def visualize_alpha_optimization(opt_results: dict, save_path: str = 'alpha_optimization.png'):
    """Visualize α optimization results."""

    results = opt_results['results']
    alpha_values = [r['alpha'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Total cost vs α
    ax = axes[0, 0]
    total_costs = [r['total_cost'] for r in results]
    ax.plot(alpha_values, total_costs, 'o-', linewidth=2.5, markersize=10, color='darkblue')
    ax.axvline(opt_results['optimal_alpha'], color='red', linestyle='--', linewidth=2,
              label=f'Optimal α={opt_results["optimal_alpha"]:.3f}')
    ax.set_xlabel('α (Probability Level)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Total Cost vs α', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Cost breakdown vs α
    ax = axes[0, 1]
    corrective_costs = [r['corrective_cost'] for r in results]
    preventive_costs = [r['preventive_cost'] for r in results]

    ax.plot(alpha_values, corrective_costs, 's-', linewidth=2, markersize=8,
           label='Corrective', color='red')
    ax.plot(alpha_values, preventive_costs, '^-', linewidth=2, markersize=8,
           label='Preventive', color='green')
    ax.axvline(opt_results['optimal_alpha'], color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('α (Probability Level)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Components vs α', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Maintenance counts vs α
    ax = axes[1, 0]
    n_corrective = [r['n_corrective'] for r in results]
    n_preventive = [r['n_preventive'] for r in results]

    ax.plot(alpha_values, n_corrective, 's-', linewidth=2, markersize=8,
           label='Corrective', color='red')
    ax.plot(alpha_values, n_preventive, '^-', linewidth=2, markersize=8,
           label='Preventive', color='green')
    ax.axvline(opt_results['optimal_alpha'], color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('α (Probability Level)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax.set_title('Maintenance Events vs α', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Failure rate vs α
    ax = axes[1, 1]
    failure_rates = [r['failure_rate'] for r in results]

    ax.plot(alpha_values, failure_rates, 'o-', linewidth=2.5, markersize=10, color='purple')
    ax.axvline(opt_results['optimal_alpha'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('α (Probability Level)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Failure Rate (per asset per time)', fontsize=12, fontweight='bold')
    ax.set_title('Failure Rate vs α', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Optimization of Probability Level α', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Optimization visualization saved to '{save_path}'")

    return fig


def sensitivity_analysis_cost_ratio(n_assets: int, H: float, tau: float, alpha: float,
                                   dist_type: str, params: dict, p: float,
                                   base_Cp: float = 100,
                                   cost_ratios: List[float] = None) -> dict:
    """
    Analyze sensitivity to Cc/Cp cost ratio.

    Parameters:
    -----------
    n_assets : int
        Number of assets
    H : float
        Time horizon
    tau : float
        Threshold
    alpha : float
        Probability level
    dist_type : str
        Distribution type
    params : dict
        Distribution parameters
    p : float
        Power parameter
    base_Cp : float
        Base preventive cost (Cc will vary)
    cost_ratios : List[float], optional
        List of Cc/Cp ratios to test

    Returns:
    --------
    dict with sensitivity results
    """
    if cost_ratios is None:
        cost_ratios = [2, 5, 10, 15, 20, 30]

    print("\n" + "="*70)
    print("COST RATIO SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"\nTesting Cc/Cp ratios: {cost_ratios}")
    print(f"Base Cp = ${base_Cp:.2f}")
    print(f"Configuration: n={n_assets}, H={H}, τ={tau}, α={alpha}")

    results = []

    # Generate common b samples for consistency
    scheduler_temp = CostBasedMaintenanceScheduler.__new__(CostBasedMaintenanceScheduler)
    b_samples = scheduler_temp._generate_b_samples(dist_type, params, n_assets)

    if dist_type == 'gamma':
        ttf_samples = 1 / b_samples
    else:
        ttf_samples = (1 / b_samples) ** (1/p)

    for i, ratio in enumerate(cost_ratios):
        Cc = base_Cp * ratio

        print(f"\n[{i+1}/{len(cost_ratios)}] Testing Cc/Cp = {ratio} (Cc=${Cc:,.0f})...")

        scheduler = CostBasedMaintenanceScheduler(
            ttf_data=ttf_samples,
            tau=tau,
            alpha=alpha,
            Cc=Cc,
            Cp=base_Cp,
            H=H
        )

        histories = scheduler.simulate_fleet_with_costs(
            n_assets=n_assets,
            dist_type=dist_type,
            params=params,
            p=p
        )

        stats = scheduler.compute_fleet_statistics(histories)

        results.append({
            'cost_ratio': ratio,
            'Cc': Cc,
            'Cp': base_Cp,
            'total_cost': stats['total_cost'],
            'corrective_cost': stats['total_corrective_cost'],
            'preventive_cost': stats['total_preventive_cost'],
            'n_corrective': stats['total_corrective_events'],
            'n_preventive': stats['total_preventive_events'],
            'corrective_cost_ratio': stats['corrective_cost_ratio']
        })

        print(f"  Total cost: ${stats['total_cost']:,.2f}")
        print(f"  Corrective %: {stats['corrective_cost_ratio']*100:.1f}%")

    return {
        'results': results,
        'cost_ratios': cost_ratios,
        'base_Cp': base_Cp
    }


def visualize_cost_ratio_sensitivity(sens_results: dict,
                                     save_path: str = 'cost_ratio_sensitivity.png'):
    """Visualize cost ratio sensitivity."""

    results = sens_results['results']
    cost_ratios = [r['cost_ratio'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Total cost vs ratio
    ax = axes[0, 0]
    total_costs = [r['total_cost'] for r in results]
    ax.plot(cost_ratios, total_costs, 'o-', linewidth=2.5, markersize=10, color='darkblue')
    ax.set_xlabel('Cc/Cp Cost Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Total Cost vs Cost Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Cost breakdown
    ax = axes[0, 1]
    corrective_costs = [r['corrective_cost'] for r in results]
    preventive_costs = [r['preventive_cost'] for r in results]

    width = (cost_ratios[1] - cost_ratios[0]) * 0.4
    x = np.array(cost_ratios)

    ax.bar(x - width/2, corrective_costs, width, label='Corrective', color='red', alpha=0.7)
    ax.bar(x + width/2, preventive_costs, width, label='Preventive', color='green', alpha=0.7)
    ax.set_xlabel('Cc/Cp Cost Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Corrective cost percentage
    ax = axes[1, 0]
    corrective_pct = [r['corrective_cost_ratio']*100 for r in results]
    ax.plot(cost_ratios, corrective_pct, 's-', linewidth=2.5, markersize=10, color='orange')
    ax.set_xlabel('Cc/Cp Cost Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Corrective Cost %', fontsize=12, fontweight='bold')
    ax.set_title('Corrective Cost Percentage', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Event counts
    ax = axes[1, 1]
    n_corrective = [r['n_corrective'] for r in results]
    n_preventive = [r['n_preventive'] for r in results]

    ax.bar(x - width/2, n_corrective, width, label='Corrective', color='red', alpha=0.7)
    ax.bar(x + width/2, n_preventive, width, label='Preventive', color='green', alpha=0.7)
    ax.set_xlabel('Cc/Cp Cost Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax.set_title('Maintenance Events', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Sensitivity to Cc/Cp Cost Ratio', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sensitivity visualization saved to '{save_path}'")

    return fig


def example_complete_optimization():
    """
    Complete example: Find optimal α for given cost structure.
    """
    print("\n" + "="*70)
    print("COST OPTIMIZATION - INTERACTIVE MODE")
    print("="*70)

    # Get user input
    print("\n" + "="*70)
    print("STEP 1: BASIC CONFIGURATION")
    print("="*70)
    
    n_assets = int(input("\nNumber of assets [50]: ") or "50")
    H = float(input("Time horizon H [200]: ") or "200")
    tau = float(input("HI threshold τ [0.7]: ") or "0.7")
    
    print("\n" + "="*70)
    print("STEP 2: COST PARAMETERS")
    print("="*70)
    
    Cc = float(input("\nCorrective maintenance cost Cc [1000]: ") or "1000")
    Cp = float(input("Preventive maintenance cost Cp [100]: ") or "100")
    
    print(f"\n  ✓ Cost ratio Cc/Cp = {Cc/Cp:.2f}:1")
    
    # Distribution selection
    print("\n" + "="*70)
    print("STEP 3: DEGRADATION MODEL")
    print("="*70)
    
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

    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\n  • Assets: {n_assets}")
    print(f"  • Horizon: {H}")
    print(f"  • Threshold: τ = {tau}")
    print(f"  • Corrective cost: Cc = ${Cc:,.0f}")
    print(f"  • Preventive cost: Cp = ${Cp:,.0f}")
    print(f"  • Cost ratio: {Cc/Cp:.0f}:1")
    print(f"  • Distribution: {dist_type}")
    print(f"  • Power parameter: p = {p}")

    # Optimize α
    print("\n" + "="*70)
    print("STEP 4: RUNNING OPTIMIZATION")
    print("="*70)
    
    opt_results = optimize_alpha(
        n_assets=n_assets,
        H=H,
        tau=tau,
        Cc=Cc,
        Cp=Cp,
        dist_type=dist_type,
        params=params,
        p=p,
        alpha_range=[0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    )

    # Visualize
    fig1 = visualize_alpha_optimization(opt_results)

    # Cost ratio sensitivity (using optimal α)
    print("\n" + "="*70)
    print("TESTING DIFFERENT COST RATIOS")
    print("="*70)

    sens_results = sensitivity_analysis_cost_ratio(
        n_assets=n_assets,
        H=H,
        tau=tau,
        alpha=opt_results['optimal_alpha'],
        dist_type=dist_type,
        params=params,
        p=p,
        base_Cp=Cp,
        cost_ratios=[2, 5, 10, 15, 20, 30, 50]
    )

    # Visualize
    fig2 = visualize_cost_ratio_sensitivity(sens_results)

    # Save summary
    summary_data = {
        'Configuration': ['Assets', 'Horizon', 'Threshold', 'Distribution', 'Optimal_Alpha',
                         'Cc', 'Cp', 'Cost_Ratio', 'Optimal_Total_Cost'],
        'Value': [n_assets, H, tau, dist_type, opt_results['optimal_alpha'],
                 Cc, Cp, Cc/Cp, opt_results['optimal_cost']]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('optimization_summary.csv', index=False)

    print("\n" + "="*70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  • alpha_optimization.png")
    print("  • cost_ratio_sensitivity.png")
    print("  • optimization_summary.csv")
    print("\n💡 Key findings:")
    print(f"  • Optimal α = {opt_results['optimal_alpha']:.3f} minimizes total cost")
    print(f"  • Minimum total cost = ${opt_results['optimal_cost']:,.2f}")
    print(f"  • As Cc/Cp ratio increases, preventive maintenance becomes more valuable")
    print("="*70 + "\n")

    plt.show()

    return opt_results, sens_results


if __name__ == "__main__":
    try:
        opt_results, sens_results = example_complete_optimization()
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
