import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sbibm.tasks import get_task, get_available_tasks
from sbibm.metrics.c2st import c2st
from abc_ra import run_abc_ra
from abc_ra_variants import run_abc_ra_variant
from smc_abc_ra import run_smc_abc_ra
import time


def benchmark_all_methods(tasks=None, budgets=[1000, 5000], num_observations=3,
                         num_posterior_samples=500, num_seeds=1):
    """
    Comprehensive benchmark of all ABC methods:
    - Rejection ABC + Linear RA (baseline)
    - Rejection ABC + Ridge RA
    - Rejection ABC + Random Forest RA
    - Rejection ABC + MLP RA
    - SMC-ABC + Linear RA
    """

    if tasks is None:
        tasks = ["gaussian_linear", "two_moons", "slcp"]

    print(f"Comprehensive ABC benchmark:")
    print(f"- Methods: 6 variants")
    print(f"- Tasks: {tasks}")
    print(f"- Observations per task: {num_observations}")
    print(f"- Budgets: {budgets}")
    print(f"- Seeds per condition: {num_seeds}")

    all_results = []
    total_conditions = len(tasks) * num_observations * len(budgets) * num_seeds * 6  # 6 methods
    print(f"Total evaluations: {total_conditions}")

    methods = [
        ('rej_linear', lambda task, obs, budget: run_abc_ra(
            task, obs, budget, num_posterior_samples, lra_variant='linear')),
        ('rej_ridge', lambda task, obs, budget: run_abc_ra_variant(
            task, obs, budget, num_posterior_samples, lra_variant='ridge')),
        ('rej_rf', lambda task, obs, budget: run_abc_ra_variant(
            task, obs, budget, num_posterior_samples, lra_variant='rf')),
        ('rej_mlp', lambda task, obs, budget: run_abc_ra_variant(
            task, obs, budget, num_posterior_samples, lra_variant='mlp')),
        ('smc_linear', lambda task, obs, budget: run_smc_abc_ra(
            task, obs, budget, num_posterior_samples, num_rounds=3)),
    ]

    with tqdm(total=total_conditions, desc="Benchmarking") as pbar:
        for task_name in tasks:
            try:
                task = get_task(task_name)
                max_obs = min(num_observations, task.num_observations)

                for obs in range(1, max_obs + 1):
                    theta_ref = task.get_reference_posterior_samples(num_observation=obs)

                    for budget in budgets:
                        for seed in range(num_seeds):
                            np.random.seed(seed * 1000 + obs * 100 + budget)
                            torch.manual_seed(seed * 1000 + obs * 100 + budget)

                            for method_name, method_func in methods:
                                try:
                                    start_time = time.time()
                                    theta_post = method_func(task, obs, budget)
                                    runtime = time.time() - start_time

                                    c2st_score = c2st(theta_ref, theta_post)

                                    result = {
                                        'task': task_name,
                                        'observation': obs,
                                        'budget': budget,
                                        'seed': seed,
                                        'method': method_name,
                                        'c2st': float(c2st_score),
                                        'runtime_seconds': runtime,
                                        'posterior_samples': len(theta_post)
                                    }

                                    all_results.append(result)

                                except Exception as e:
                                    all_results.append({
                                        'task': task_name,
                                        'observation': obs,
                                        'budget': budget,
                                        'seed': seed,
                                        'method': method_name,
                                        'c2st': None,
                                        'error': str(e)
                                    })

                                pbar.update(1)

            except Exception as e:
                print(f"Failed to load task {task_name}: {e}")
                # Skip all conditions for this task
                pbar.update(num_observations * len(budgets) * num_seeds * len(methods))

    # Convert to DataFrame and analyze
    df = pd.DataFrame(all_results)

    if len(df.dropna(subset=['c2st'])) > 0:
        # Summary statistics
        print("\n" + "="*80)
        print("COMPREHENSIVE ABC BENCHMARK RESULTS")
        print("="*80)

        valid_results = df.dropna(subset=['c2st'])
        print(f"Successful evaluations: {len(valid_results)}/{len(df)}")

        # Aggregate by method and budget
        summary = valid_results.groupby(['method', 'budget']).agg({
            'c2st': ['mean', 'std', 'count'],
            'runtime_seconds': ['mean', 'std']
        }).round(4)

        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        print("\nMean C2ST by method and budget (lower is better):")
        pivot_c2st = summary.pivot(index='method', columns='budget', values='c2st_mean')
        print(pivot_c2st)

        print("\nRuntime (seconds) by method and budget:")
        pivot_runtime = summary.pivot(index='method', columns='budget', values='runtime_seconds_mean')
        print(pivot_runtime)

        # Best methods per task
        print("\nBest methods per task and budget:")
        for task in valid_results['task'].unique():
            task_data = valid_results[valid_results['task'] == task]
            print(f"\n{task}:")
            for budget in sorted(task_data['budget'].unique()):
                budget_data = task_data[task_data['budget'] == budget]
                best_method = budget_data.loc[budget_data['c2st'].idxmin()]
                print(".4f")

        # Budget scaling analysis
        print("\nBudget scaling (C2ST improvement 5k vs 1k):")
        scaling_results = []
        for method in valid_results['method'].unique():
            method_data = valid_results[valid_results['method'] == method]
            if 1000 in method_data['budget'].values and 5000 in method_data['budget'].values:
                c2st_1k = method_data[method_data['budget'] == 1000]['c2st'].mean()
                c2st_5k = method_data[method_data['budget'] == 5000]['c2st'].mean()
                improvement = c2st_1k - c2st_5k
                scaling_results.append((method, c2st_1k, c2st_5k, improvement))

        for method, c2st_1k, c2st_5k, improvement in sorted(scaling_results, key=lambda x: x[3], reverse=True):
            print(".4f")

    return df, summary if 'summary' in locals() else None


def quick_comprehensive_test():
    """Quick test with reduced scope"""
    return benchmark_all_methods(
        tasks=["gaussian_linear", "two_moons"],
        budgets=[1000, 5000],
        num_observations=2,
        num_posterior_samples=200,
        num_seeds=1
    )


if __name__ == "__main__":
    print("Running comprehensive ABC benchmark...")
    detailed_df, summary_df = quick_comprehensive_test()

    # Save results
    detailed_df.to_csv("comprehensive_abc_results.csv", index=False)
    if summary_df is not None:
        summary_df.to_csv("comprehensive_abc_summary.csv", index=False)

    print(f"\nResults saved to comprehensive_abc_results.csv and comprehensive_abc_summary.csv")

    # Show sample results
    print("\nSample detailed results:")
    print(detailed_df.head(10))

    if summary_df is not None:
        print("\nSummary statistics:")
        print(summary_df.head())
