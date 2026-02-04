import numpy as np
import torch
from typing import Optional, Tuple
from abc_ra import FeatureExtractor
from abc_ra_variants import LinearLRA, apply_lra_variant


def simulate_batch(task, proposal_dist, simulator, batch_size: int):
    """Simulate a batch from proposal distribution"""
    theta_batch = proposal_dist.sample((batch_size,))
    x_batch = simulator(theta_batch)
    return theta_batch, x_batch


def compute_distances(X: np.ndarray, X_obs: np.ndarray) -> np.ndarray:
    """Compute L2 distances"""
    return np.linalg.norm(X - X_obs, axis=1)


def compute_weights(distances: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
    """Compute Epanechnikov kernel weights"""
    if eps is None:
        eps = np.max(distances)

    u = distances / eps
    weights = np.maximum(1 - u**2, 0)  # Epanechnikov kernel

    return weights


def perturb_particles(theta: torch.Tensor, weights: np.ndarray,
                     perturbation_scale: float = 0.1) -> torch.Tensor:
    """
    Perturb particles using Gaussian kernel in parameter space.

    Args:
        theta: Current particles (n_particles, dim_theta)
        weights: Importance weights (n_particles,)
        perturbation_scale: Scale for perturbation kernel

    Returns:
        perturbed_theta: New proposal particles (n_particles, dim_theta)
    """
    n_particles, dim_theta = theta.shape

    # Sample which particles to copy (weighted by importance weights)
    indices = np.random.choice(n_particles, size=n_particles, p=weights/weights.sum())

    # Copy selected particles
    perturbed_theta = theta[indices].clone()

    # Add Gaussian noise
    noise = torch.randn_like(perturbed_theta) * perturbation_scale
    perturbed_theta += noise

    return perturbed_theta


def run_smc_abc_ra(task, num_observation: int, num_simulations: int, num_posterior_samples: int,
                   num_rounds: int = 5, accept_frac: float = 0.01, min_accept: int = 200,
                   use_weights: bool = True, batch_size: int = 1000,
                   perturbation_scale: float = 0.1) -> torch.Tensor:
    """
    Run SMC-ABC with Regression Adjustment.

    Args:
        task: sbibm Task instance
        num_observation: Which observation to use (1-10)
        num_simulations: Total simulation budget
        num_posterior_samples: Number of posterior samples to return
        num_rounds: Number of SMC rounds
        accept_frac: Fraction of simulations to accept per round
        min_accept: Minimum number to accept per round
        use_weights: Whether to use kernel weights
        batch_size: Batch size for simulation
        perturbation_scale: Scale for particle perturbation

    Returns:
        theta_posterior: Posterior samples (num_posterior_samples, dim_theta)
    """

    # Get task components
    prior = task.get_prior_dist()  # Use distribution for sampling
    prior_sample = task.get_prior()  # For initial sampling
    simulator = task.get_simulator(max_calls=num_simulations)
    x_obs = task.get_observation(num_observation)

    # Get transforms
    transforms = task._get_transforms(automatic_transforms_enabled=False)["parameters"]

    # Initialize feature extractor
    extractor = FeatureExtractor()

    print(f"Running SMC-ABC + RA with {num_rounds} rounds, budget {num_simulations}")

    # Calculate simulations per round
    sims_per_round = num_simulations // num_rounds

    # Initialize with prior
    current_proposal = prior
    all_theta = []
    all_weights = []

    for round_idx in range(num_rounds):
        print(f"\nRound {round_idx + 1}/{num_rounds}")

        # Simulate from current proposal
        theta_round = []
        x_round_raw = []

        remaining = sims_per_round
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            theta_batch, x_batch = simulate_batch(task, current_proposal, simulator, current_batch)

            theta_round.append(theta_batch)
            x_round_raw.append(x_batch)
            remaining -= current_batch

        theta_round = torch.cat(theta_round, dim=0)
        x_round_raw = torch.cat(x_round_raw, dim=0)

        # Extract features (fit on first round, transform on subsequent)
        if round_idx == 0:
            X_round = extractor.fit_transform(x_round_raw)
            X_obs_round = extractor.transform_single(x_obs.squeeze(0))
        else:
            X_round = extractor.transform(x_round_raw)
            X_obs_round = extractor.transform_single(x_obs.squeeze(0))

        print(f"  Simulated {theta_round.shape[0]} particles")

        # Compute distances and select accepted particles
        distances = compute_distances(X_round, X_obs_round)

        # For SMC, we accept a percentage and use all accepted particles
        n_accept = max(min_accept, int(accept_frac * len(distances)))
        accepted_indices = np.argsort(distances)[:n_accept]

        theta_acc = theta_round[accepted_indices]
        X_acc = X_round[accepted_indices]
        d_acc = distances[accepted_indices]

        print(f"  Accepted {len(theta_acc)} particles, max distance: {np.max(d_acc):.4f}")

        # Apply LRA to accepted particles
        weights = compute_weights(d_acc) if use_weights else None
        lra_model = LinearLRA()

        theta_adj = apply_lra_variant(theta_acc, X_acc, X_obs_round, lra_model, weights, transforms)

        # Store adjusted particles and weights
        all_theta.append(theta_adj)
        all_weights.append(weights if weights is not None else np.ones(len(theta_adj)))

        # Update proposal for next round (except last round)
        if round_idx < num_rounds - 1:
            # Combine all particles from this round
            combined_theta = torch.cat(all_theta, dim=0)
            combined_weights = np.concatenate(all_weights)
            combined_weights = combined_weights / combined_weights.sum()  # Normalize

            # Create new proposal by perturbing current particles
            perturbed_theta = perturb_particles(combined_theta, combined_weights, perturbation_scale)

            # Create new proposal distribution (fit multivariate normal to perturbed particles)
            from torch.distributions import MultivariateNormal
            theta_mean = perturbed_theta.mean(dim=0)
            theta_cov = torch.cov(perturbed_theta.T) + torch.eye(perturbed_theta.shape[1]) * 1e-6  # Add small diagonal for stability
            current_proposal = MultivariateNormal(theta_mean, theta_cov)

            print(f"  Updated proposal with {len(perturbed_theta)} particles")

    # Final round: combine all accepted particles with their weights
    final_theta = torch.cat(all_theta, dim=0)
    final_weights = np.concatenate(all_weights)
    final_weights = final_weights / final_weights.sum()  # Normalize

    print(f"\nFinal: {len(final_theta)} particles from {num_rounds} rounds")

    # Resample to get desired number of posterior samples
    n_final = final_theta.shape[0]
    if num_posterior_samples < n_final:
        # Sample with replacement using weights
        indices = np.random.choice(n_final, size=num_posterior_samples, p=final_weights)
        theta_post = final_theta[indices]
    else:
        # If we need more samples than we have, resample with replacement
        indices = np.random.choice(n_final, size=num_posterior_samples, p=final_weights)
        theta_post = final_theta[indices]

    print(f"Final posterior samples: {theta_post.shape}")

    return theta_post


def compare_smc_vs_rejection():
    """Compare SMC-ABC + RA vs Rejection ABC + RA"""
    from sbibm.tasks import get_task
    from sbibm.metrics.c2st import c2st
    from abc_ra import run_abc_ra

    task = get_task("two_moons")  # Use two_moons as it's more challenging
    theta_ref = task.get_reference_posterior_samples(num_observation=1)

    results = {}

    # Test Rejection ABC + RA
    print("Testing Rejection ABC + RA...")
    theta_rej = run_abc_ra(
        task=task,
        num_observation=1,
        num_simulations=5000,
        num_posterior_samples=1000
    )
    c2st_rej = c2st(theta_ref, theta_rej)
    results['rej_abc_ra'] = float(c2st_rej)
    print(".4f")

    # Test SMC-ABC + RA
    print("\nTesting SMC-ABC + RA...")
    theta_smc = run_smc_abc_ra(
        task=task,
        num_observation=1,
        num_simulations=5000,
        num_posterior_samples=1000,
        num_rounds=3
    )
    c2st_smc = c2st(theta_ref, theta_smc)
    results['smc_abc_ra'] = float(c2st_smc)
    print(".4f")

    # Comparison
    print("\nComparison:")
    print(".4f")
    if results['smc_abc_ra'] < results['rej_abc_ra']:
        improvement = results['rej_abc_ra'] - results['smc_abc_ra']
        print(".4f")
    else:
        print("  SMC-ABC did not improve over Rejection ABC")

    return results


if __name__ == "__main__":
    compare_smc_vs_rejection()
