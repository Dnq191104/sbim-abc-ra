import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sbibm.tasks import get_task, get_available_tasks
from sbibm.metrics.c2st import c2st
from abc_ra import run_abc_ra
import time


def benchmark_abc_ra(tasks=None, num_simulations=10000, num_posterior_samples=1000,
                     num_observations=10, accept_frac=0.01, min_accept=200):
    """
    Benchmark ABC RA across multiple tasks and observations.

    Args:
        tasks: List of task names to benchmark (default: all available)
        num_simulations: Simulation budget per task-observation pair
        num_posterior_samples: Posterior samples to generate
        num_observations: Number of observations to test per task (1-10)
        accept_frac: Acceptance fraction
        min_accept: Minimum accepted samples

    Returns:
        DataFrame with results
    """

    if tasks is None:
        tasks = get_available_tasks()
        # Filter out ODE tasks that require Julia
        tasks = [t for t in tasks if t not in ['sir', 'lotka_volterra']]

    print(f"Benchmarking ABC RA on tasks: {tasks}")
    print(f"Budget: {num_simulations} simulations, {num_posterior_samples} posterior samples")
    print(f"Acceptance: {accept_frac*100:.1f}% or {min_accept} minimum")

    results = []

    for task_name in tasks:
        print(f"\n{'='*50}")
        print(f"TASK: {task_name}")

        try:
            task = get_task(task_name)
        except Exception as e:
            print(f"Failed to load task {task_name}: {e}")
            continue

        for obs in range(1, min(num_observations, task.num_observations) + 1):
            print(f"  Observation {obs}/{task.num_observations}...")

            try:
                # Time the algorithm
                start_time = time.time()

                # Run ABC RA
                theta_post = run_abc_ra(
                    task=task,
                    num_observation=obs,
                    num_simulations=num_simulations,
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

                # Store results
                result = {
                    'task': task_name,
                    'observation': obs,
                    'num_simulations': num_simulations,
                    'num_posterior_samples': num_posterior_samples,
                    'accept_frac': accept_frac,
                    'min_accept': min_accept,
                    'c2st': float(c2st_score),
                    'runtime_seconds': runtime,
                    'posterior_mean': theta_post.mean(dim=0).tolist(),
                    'posterior_std': theta_post.std(dim=0).tolist(),
                    'reference_mean': theta_ref.mean(dim=0).tolist(),
                    'reference_std': theta_ref.std(dim=0).tolist()
                }

                results.append(result)

                print(".4f")
            except Exception as e:
                print(f"    FAILED: {e}")
                # Store failure
                results.append({
                    'task': task_name,
                    'observation': obs,
                    'num_simulations': num_simulations,
                    'c2st': None,
                    'error': str(e)
                })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Summary statistics
    if len(df) > 0 and 'c2st' in df.columns:
        valid_results = df.dropna(subset=['c2st'])
        if len(valid_results) > 0:
            print(f"\n{'='*50}")
            print("SUMMARY STATISTICS")
            print(f"Tasks evaluated: {len(valid_results.groupby('task'))}")
            print(f"Total evaluations: {len(valid_results)}")
            print(".4f")
            print(f"C2ST range: [{valid_results['c2st'].min():.4f}, {valid_results['c2st'].max():.4f}]")

            # Per-task summary
            task_summary = valid_results.groupby('task')['c2st'].agg(['mean', 'std', 'count'])
            print("\nPer-task C2ST (lower is better):")
            for task_name, row in task_summary.iterrows():
                print(".4f")
    return df


def quick_benchmark():
    """Quick benchmark on a few tasks for testing"""
    tasks = ["gaussian_linear", "two_moons", "slcp"]
    return benchmark_abc_ra(
        tasks=tasks,
        num_simulations=5000,  # Smaller budget for quick testing
        num_posterior_samples=500,
        num_observations=3,  # Test fewer observations
        accept_frac=0.02,
        min_accept=100
    )


def save_results(df, filename="abc_ra_benchmark_results.csv"):
    """Save results to CSV"""
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # Quick test
    print("Running quick benchmark...")
    results_df = quick_benchmark()

    # Save results
    save_results(results_df, "abc_ra_quick_benchmark.csv")

    # Show sample results
    print("\nSample results:")
    print(results_df.head())
