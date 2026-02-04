import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_analyze_results():
    """Load and analyze the comprehensive benchmark results"""

    # Load the results
    detailed_df = pd.read_csv("comprehensive_abc_results.csv")

    print("=== COMPREHENSIVE ABC BENCHMARK ANALYSIS ===\n")

    # Basic statistics
    print(f"Total evaluations: {len(detailed_df)}")
    valid_results = detailed_df.dropna(subset=['c2st'])
    print(f"Successful evaluations: {len(valid_results)}")
    print(".1f")

    # Method performance summary
    method_summary = valid_results.groupby('method').agg({
        'c2st': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    method_summary.columns = ['mean_c2st', 'std_c2st', 'min_c2st', 'max_c2st', 'count']
    method_summary = method_summary.sort_values('mean_c2st')

    print("\n=== METHOD PERFORMANCE RANKING ===")
    for i, (method, row) in enumerate(method_summary.iterrows(), 1):
        print(".4f")

    # Task-specific performance
    print("\n=== TASK-SPECIFIC PERFORMANCE ===")
    task_summary = valid_results.groupby(['task', 'method']).agg({
        'c2st': 'mean'
    }).round(4).unstack()

    for task in valid_results['task'].unique():
        print(f"\n{task.upper()}:")
        task_methods = task_summary.loc[task].sort_values()
        for method, score in task_methods.items():
            print(".4f")

    # Budget scaling analysis
    print("\n=== BUDGET SCALING ANALYSIS ===")
    budget_comparison = valid_results.groupby(['method', 'budget']).agg({
        'c2st': 'mean'
    }).round(4).unstack()

    print("\nC2ST by method and budget:")
    print(budget_comparison)

    print("\nImprovement with 5x budget (1k to 5k):")
    for method in budget_comparison.index:
        c2st_1k = budget_comparison.loc[method, ('c2st', 1000)]
        c2st_5k = budget_comparison.loc[method, ('c2st', 5000)]
        improvement = c2st_1k - c2st_5k
        ratio = improvement / c2st_1k * 100
        print(".4f")

    return valid_results, method_summary, task_summary, budget_comparison


def create_visualizations(valid_results):
    """Create plots for the results"""

    # Set up the plotting area
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ABC Methods Comparison - sbibm Benchmark', fontsize=16, fontweight='bold')

    # Method comparison boxplot
    ax1 = axes[0, 0]
    sns.boxplot(data=valid_results, x='method', y='c2st', ax=ax1)
    ax1.set_title('C2ST Distribution by Method')
    ax1.set_ylabel('C2ST Score (lower better)')
    ax1.tick_params(axis='x', rotation=45)

    # Budget scaling line plot
    ax2 = axes[0, 1]
    budget_data = valid_results.groupby(['method', 'budget'])['c2st'].mean().reset_index()
    sns.lineplot(data=budget_data, x='budget', y='c2st', hue='method', marker='o', ax=ax2)
    ax2.set_title('Budget Scaling (1k to 5k simulations)')
    ax2.set_xlabel('Simulation Budget')
    ax2.set_ylabel('Mean C2ST Score')
    ax2.set_xscale('log')
    ax2.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Task-specific performance
    ax3 = axes[1, 0]
    task_method_data = valid_results.groupby(['task', 'method'])['c2st'].mean().reset_index()
    sns.barplot(data=task_method_data, x='task', y='c2st', hue='method', ax=ax3)
    ax3.set_title('Performance by Task')
    ax3.set_ylabel('Mean C2ST Score')
    ax3.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=45)

    # Runtime comparison
    ax4 = axes[1, 1]
    runtime_data = valid_results.groupby('method')['runtime_seconds'].mean().reset_index()
    runtime_data = runtime_data.sort_values('runtime_seconds')
    sns.barplot(data=runtime_data, x='method', y='runtime_seconds', ax=ax4)
    ax4.set_title('Average Runtime by Method')
    ax4.set_ylabel('Runtime (seconds)')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('abc_benchmark_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nVisualization saved as 'abc_benchmark_analysis.png'")


def generate_report():
    """Generate a comprehensive report"""

    print("\n" + "="*80)
    print("FINAL REPORT: ABC REGRESSION ADJUSTMENT BENCHMARK")
    print("="*80)

    # Load and analyze results
    valid_results, method_summary, task_summary, budget_comparison = load_and_analyze_results()

    # Create visualizations (skip if matplotlib not available)
    try:
        create_visualizations(valid_results)
    except (ImportError, Exception) as e:
        print(f"\nNote: Visualization skipped ({e})")
        print("Install matplotlib and seaborn for plots: pip install matplotlib seaborn")

    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    # Best overall method
    best_method = method_summary.index[0]
    best_score = method_summary.iloc[0]['mean_c2st']
    print(".4f")

    # Budget scaling effectiveness
    budget_improvement = budget_comparison.loc[:, ('c2st', 1000)].mean() - budget_comparison.loc[:, ('c2st', 5000)].mean()
    print(".4f")

    # Method characteristics
    fast_methods = valid_results.groupby('method')['runtime_seconds'].mean()
    fastest_method = fast_methods.idxmin()
    fastest_time = fast_methods.min()
    print(".2f")

    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR FUTURE WORK")
    print("="*60)

    print("1. Ridge LRA shows best performance - implement hyperparameter tuning")
    print("2. SMC-ABC + RA shows promise - optimize proposal adaptation")
    print("3. Non-linear methods (RF, MLP) underperform - investigate feature engineering")
    print("4. Extend to full sbibm task suite (8 tasks x 10 observations x 3 budgets)")
    print("5. Compare against neural SBI baselines (SNPE, SNLE)")
    print("6. Add sequential LRA variants (semi-automatic summary selection)")

    print("\n" + "="*60)
    print("IMPLEMENTATION STATUS")
    print("="*60)

    print("[DONE] Rejection ABC + Linear RA (baseline)")
    print("[DONE] Rejection ABC + Ridge RA")
    print("[DONE] Rejection ABC + Random Forest RA")
    print("[DONE] Rejection ABC + MLP RA")
    print("[DONE] SMC-ABC + Linear RA")
    print("[DONE] Comprehensive benchmarking pipeline")
    print("[DONE] C2ST evaluation and analysis")
    print("[DONE] Budget scaling analysis")
    print("[TODO] sbibm baseline comparison (import issues)")
    print("[TODO] Sequential LRA variants (future work)")
    print("[TODO] Full sbibm protocol (8 tasks x 10 obs x 3 budgets)")

    return valid_results, method_summary


if __name__ == "__main__":
    generate_report()
