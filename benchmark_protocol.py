import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sbibm.tasks import get_task, get_available_tasks
from sbibm.metrics.c2st import c2st
from abc_ra import run_abc_ra
import time
import random


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def benchmark_full_protocol(tasks=None, budgets=[1000, 10000, 100000],
                          num_observations=10, num_posterior_samples=1000,
                          num_seeds=1, accept_frac=0.01, min_accept=200):
    """
    Run full benchmark protocol as described in the paper:
    - All tasks (excluding ODE tasks)
    - All 10 observations per task
    - Budget sweep: 1k, 10k, 100k simulations
    - Multiple seeds for variability
    - Aggregate results with mean/std across observations and seeds
    """

    if tasks is None:
        tasks = get_available_tasks()
        # Filter out ODE tasks that require Julia
        tasks = [t for t in tasks if t not in ['sir', 'lotka_volterra']]
        print(f"Using tasks: {tasks}")

    print(f"Benchmark protocol:")
    print(f"- Tasks: {len(tasks)}")
    print(f"- Observations per task: {num_observations}")
    print(f"- Budgets: {budgets}")
    print(f"- Seeds per condition: {num_seeds}")
    print(f"- Posterior samples: {num_posterior_samples}")
    print(f"- Acceptance: {accept_frac*100:.1f}% or {min_accept} minimum")

    all_results = []

    total_conditions = len(tasks) * num_observations * len(budgets) * num_seeds
    print(f"Total conditions: {total_conditions}")

    with tqdm(total=total_conditions, desc="Benchmarking") as pbar:
        for task_name in tasks:
            try:
                task = get_task(task_name)
                max_obs = min(num_observations, task.num_observations)

                for obs in range(1, max_obs + 1):
                    for budget in budgets:
                        for seed in range(num_seeds):
                            set_seed(seed * 1000 + obs * 100 + budget)  # Unique seed

                            try:
                                # Time the algorithm
                                start_time = time.time()

                                # Run ABC RA
                                theta_post = run_abc_ra(
                                    task=task,
                                    num_observation=obs,
                                    num_simulations=budget,
                                    num_posterior_samples=num_posterior_samples,
                                    accept_frac=accept_frac,
                                    min_accept=min_accept,
                                    use_weights=True
                                )

                                runtime = time.time() - start_time

                                # Get reference posterior
                                theta_ref = task.get_reference_posterior_samples(num_observation=obs)

                                # Compute C2ST
                                c2st_score = c2st(theta_ref, theta_post)

                                # Store result
                                result = {
                                    'task': task_name,
                                    'observation': obs,
                                    'budget': budget,
                                    'seed': seed,
                                    'c2st': float(c2st_score),
                                    'runtime_seconds': runtime,
                                    'algorithm': 'abc_ra'
                                }

                                all_results.append(result)

                            except Exception as e:
                                # Store failure
                                all_results.append({
                                    'task': task_name,
                                    'observation': obs,
                                    'budget': budget,
                                    'seed': seed,
                                    'c2st': None,
                                    'error': str(e),
                                    'algorithm': 'abc_ra'
                                })

                            pbar.update(1)

            except Exception as e:
                print(f"Failed to load task {task_name}: {e}")
                # Skip all conditions for this task
                pbar.update(num_observations * len(budgets) * num_seeds)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Compute aggregates
    if len(df.dropna(subset=['c2st'])) > 0:
        # Aggregate across seeds and observations
        summary_df = df.dropna(subset=['c2st']).groupby(['task', 'budget', 'algorithm']).agg({
            'c2st': ['mean', 'std', 'count'],
            'runtime_seconds': ['mean', 'std']
        }).round(4)

        # Flatten column names
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        summary_df = summary_df.reset_index()

        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)

        print(f"\nSuccessful evaluations: {len(df.dropna(subset=['c2st']))}/{len(df)}")

        # Overall statistics
        valid_results = df.dropna(subset=['c2st'])
        print("\nOverall C2ST statistics:")
        print(".4f")
        print(".4f")

        # Per-task results
        print("\nPer-task C2ST results (mean ± std across observations and seeds):")
        for task in sorted(valid_results['task'].unique()):
            task_data = valid_results[valid_results['task'] == task]
            print(f"\n{task}:")
            for budget in sorted(task_data['budget'].unique()):
                budget_data = task_data[task_data['budget'] == budget]
                mean_c2st = budget_data['c2st'].mean()
                std_c2st = budget_data['c2st'].std()
                count = len(budget_data)
                print(".4f")

        # Budget scaling analysis
        print("\nBudget scaling analysis:")
        for task in sorted(valid_results['task'].unique()):
            task_data = valid_results[valid_results['task'] == task]
            print(f"\n{task} - C2ST improvement with budget:")
            for i, budget1 in enumerate(sorted(task_data['budget'].unique())):
                for budget2 in sorted(list(task_data['budget'].unique())[i+1:], reverse=True):
                    c2st1 = task_data[task_data['budget'] == budget1]['c2st'].mean()
                    c2st2 = task_data[task_data['budget'] == budget2]['c2st'].mean()
                    improvement = c2st1 - c2st2
                    ratio = budget2 / budget1
                    print(".4f")

    return df, summary_df if 'summary_df' in locals() else None


def quick_protocol_test():
    """Quick test with reduced scope for development"""
    tasks = ["gaussian_linear", "two_moons"]
    budgets = [1000, 5000]  # Smaller budgets for testing

    return benchmark_full_protocol(
        tasks=tasks,
        budgets=budgets,
        num_observations=3,  # Fewer observations
        num_posterior_samples=500,
        num_seeds=1,
        accept_frac=0.02,
        min_accept=100
    )


def save_protocol_results(df, summary_df=None, prefix="protocol"):
    """Save detailed and summary results"""
    df.to_csv(f"{prefix}_detailed_results.csv", index=False)
    if summary_df is not None:
        summary_df.to_csv(f"{prefix}_summary_results.csv", index=False)
    print(f"\nResults saved with prefix: {prefix}")


if __name__ == "__main__":
    # Run full protocol
    print("Running full benchmark protocol...")
    detailed_df, summary_df = benchmark_full_protocol()

    # Save results
    save_protocol_results(detailed_df, summary_df, "abc_ra_protocol")

    # Show sample results
    print("\nSample detailed results:")
    print(detailed_df.head(10))

    if summary_df is not None:
        print("\nSummary results:")
        print(summary_df.head(10))
