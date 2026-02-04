import torch
from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st
from smc_abc_ra import run_smc_abc_ra

def test_smc_two_moons():
    """Test SMC-ABC + Linear RA on Two Moons task"""

    task = get_task("two_moons")
    theta_ref = task.get_reference_posterior_samples(num_observation=1)

    budgets = [1000, 10000]  # 10^3 and 10^4 simulations
    num_posterior_samples = 1000

    print("Testing SMC-ABC + Linear RA on Two Moons")
    print("=" * 50)

    results = []

    for budget in budgets:
        print(f"\nRunning with {budget} simulations...")

        # Run SMC-ABC + RA
        theta_post = run_smc_abc_ra(
            task=task,
            num_observation=1,
            num_simulations=budget,
            num_posterior_samples=num_posterior_samples,
            num_rounds=3,  # 3 rounds as in our implementation
            accept_frac=0.01,
            min_accept=200
        )

        # Compute C2ST score
        c2st_score = c2st(theta_ref, theta_post)

        results.append({
            'budget': budget,
            'c2st_score': float(c2st_score),
            'posterior_shape': theta_post.shape
        })

        print(f"  Budget: {budget}")
        print(f"  Posterior samples shape: {theta_post.shape}")
        print(".4f")
        print(f"  Reference posterior mean: {theta_ref.mean(dim=0)}")
        print(f"  SMC posterior mean: {theta_post.mean(dim=0)}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print(".4f")
    print(".4f")

    improvement = results[0]['c2st_score'] - results[1]['c2st_score']
    print(".4f")

    return results

if __name__ == "__main__":
    results = test_smc_two_moons()

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("smc_two_moons_results.csv", index=False)
    print("\nResults saved to smc_two_moons_results.csv")
