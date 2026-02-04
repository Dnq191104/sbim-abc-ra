import torch
import pandas as pd
from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st
from abc_ra import run_abc_ra
import time

# Import sbibm algorithms directly
try:
    from sbibm.algorithms import rej_abc, smc_abc
    print("Successfully imported sbibm algorithms")
except ImportError as e:
    print(f"Failed to import sbibm algorithms: {e}")
    # Fallback to None
    rej_abc = None
    smc_abc = None


def compare_algorithms(task_name: str, num_observation: int = 1, num_simulations: int = 5000,
                      num_posterior_samples: int = 1000):
    """
    Compare our ABC RA implementation against sbibm baselines on the same conditions.
    """
    print(f"\nComparing algorithms on {task_name}, observation {num_observation}, budget {num_simulations}")

    task = get_task(task_name)
    theta_ref = task.get_reference_posterior_samples(num_observation=num_observation)

    results = []

    # Test sbibm's REJ-ABC (with LRA enabled for fair comparison)
    if rej_abc is not None:
        print("Running sbibm REJ-ABC + LRA...")
        start_time = time.time()
        try:
            theta_sbibm_rej, sim_calls_rej, _ = rej_abc(
                task=task,
                num_observation=num_observation,
                num_simulations=num_simulations,
                num_samples=num_posterior_samples,
                lra=True  # Enable LRA for fair comparison
            )
            runtime_sbibm_rej = time.time() - start_time
            c2st_sbibm_rej = c2st(theta_ref, theta_sbibm_rej)

            results.append({
                'algorithm': 'sbibm_rej_abc_lra',
                'task': task_name,
                'observation': num_observation,
                'budget': num_simulations,
                'c2st': float(c2st_sbibm_rej),
                'runtime': runtime_sbibm_rej,
                'sim_calls': sim_calls_rej
            })
            print(".4f")
        except Exception as e:
            print(f"sbibm REJ-ABC + LRA failed: {e}")
    else:
        print("sbibm REJ-ABC not available")

    # Test sbibm's SMC-ABC (with LRA enabled)
    if smc_abc is not None:
        print("Running sbibm SMC-ABC + LRA...")
        start_time = time.time()
        try:
            theta_sbibm_smc, sim_calls_smc, _ = smc_abc(
                task=task,
                num_observation=num_observation,
                num_simulations=num_simulations,
                num_samples=num_posterior_samples,
                lra=True  # Enable LRA
            )
            runtime_sbibm_smc = time.time() - start_time
            c2st_sbibm_smc = c2st(theta_ref, theta_sbibm_smc)

            results.append({
                'algorithm': 'sbibm_smc_abc_lra',
                'task': task_name,
                'observation': num_observation,
                'budget': num_simulations,
                'c2st': float(c2st_sbibm_smc),
                'runtime': runtime_sbibm_smc,
                'sim_calls': sim_calls_smc
            })
            print(".4f")
        except Exception as e:
            print(f"sbibm SMC-ABC + LRA failed: {e}")
    else:
        print("sbibm SMC-ABC not available")

    # Test our ABC RA
    print("Running our ABC RA...")
    start_time = time.time()
    try:
        theta_ours = run_abc_ra(
            task=task,
            num_observation=num_observation,
            num_simulations=num_simulations,
            num_posterior_samples=num_posterior_samples
        )
        runtime_ours = time.time() - start_time
        c2st_ours = c2st(theta_ref, theta_ours)

        results.append({
            'algorithm': 'our_abc_ra',
            'task': task_name,
            'observation': num_observation,
            'budget': num_simulations,
            'c2st': float(c2st_ours),
            'runtime': runtime_ours,
            'sim_calls': num_simulations
        })
        print(".4f")
    except Exception as e:
        print(f"Our ABC RA failed: {e}")

    return results


def run_comparison_sweep():
    """Run comparison across multiple conditions"""
    # Test conditions matching our benchmark
    test_conditions = [
        ("gaussian_linear", 1, 5000),
        ("two_moons", 1, 5000),
        ("slcp", 1, 5000),
    ]

    all_results = []

    for task_name, obs, budget in test_conditions:
        try:
            results = compare_algorithms(task_name, obs, budget)
            all_results.extend(results)
        except Exception as e:
            print(f"Failed to compare on {task_name}: {e}")

    # Convert to DataFrame and analyze
    df = pd.DataFrame(all_results)

    if len(df) > 0:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)

        # Pivot table for easy comparison
        pivot = df.pivot_table(
            values='c2st',
            index=['task', 'budget'],
            columns='algorithm'
        )
        print("\nC2ST Scores (lower is better):")
        print(pivot)

        # Performance relative to sbibm REJ-ABC + LRA
        if 'sbibm_rej_abc_lra' in df['algorithm'].values:
            baseline_scores = df[df['algorithm'] == 'sbibm_rej_abc_lra'][['task', 'budget', 'c2st']]
            for _, baseline in baseline_scores.iterrows():
                our_score = df[
                    (df['algorithm'] == 'our_abc_ra') &
                    (df['task'] == baseline['task']) &
                    (df['budget'] == baseline['budget'])
                ]['c2st']
                if len(our_score) > 0:
                    diff = our_score.iloc[0] - baseline['c2st']
                    print(".4f")

        # Compare SMC-ABC vs REJ-ABC
        if 'sbibm_smc_abc_lra' in df['algorithm'].values and 'sbibm_rej_abc_lra' in df['algorithm'].values:
            print("\nSMC-ABC vs REJ-ABC comparison:")
            for task in df['task'].unique():
                for budget in df['budget'].unique():
                    smc_score = df[
                        (df['algorithm'] == 'sbibm_smc_abc_lra') &
                        (df['task'] == task) &
                        (df['budget'] == budget)
                    ]['c2st']
                    rej_score = df[
                        (df['algorithm'] == 'sbibm_rej_abc_lra') &
                        (df['task'] == task) &
                        (df['budget'] == budget)
                    ]['c2st']
                    if len(smc_score) > 0 and len(rej_score) > 0:
                        improvement = rej_score.iloc[0] - smc_score.iloc[0]
                        print(".4f")

        # Runtime comparison
        print("\nRuntime comparison (seconds):")
        runtime_pivot = df.pivot_table(
            values='runtime',
            index=['task', 'budget'],
            columns='algorithm'
        )
        print(runtime_pivot)

    return df


if __name__ == "__main__":
    results_df = run_comparison_sweep()
    results_df.to_csv("baseline_comparison.csv", index=False)
    print(f"\nDetailed results saved to baseline_comparison.csv")
