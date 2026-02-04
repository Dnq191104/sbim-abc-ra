import torch
import numpy as np
from sbibm.tasks import get_task
from sbibm.algorithms.sbi.utils import run_lra
from abc_ra import run_abc_ra, FeatureExtractor, apply_lra, compute_weights


def validate_lra_implementations(task_name: str = "gaussian_linear", num_observation: int = 1):
    """
    Validate our LRA implementation against sbibm's run_lra.
    """
    print(f"Validating LRA on {task_name}, observation {num_observation}")

    task = get_task(task_name)

    # Get some accepted samples by running ABC with high acceptance rate
    theta_post = run_abc_ra(
        task=task,
        num_observation=num_observation,
        num_simulations=5000,
        num_posterior_samples=500,
        accept_frac=0.1,  # High acceptance to get more samples
        min_accept=100,
        use_weights=False  # Start with uniform weights
    )

    # Now let's get the raw accepted samples by simulating and accepting
    # We'll use a simplified approach: simulate, compute distances, accept closest
    prior = task.get_prior()
    simulator = task.get_simulator(max_calls=2000)
    x_obs = task.get_observation(num_observation)

    # Simulate samples
    theta_sims = prior(1000)
    x_sims = simulator(theta_sims)

    # Extract features
    extractor = FeatureExtractor()
    X_sims = extractor.fit_transform(x_sims)
    X_obs = extractor.transform_single(x_obs.squeeze(0))

    # Compute distances and select accepted
    distances = np.linalg.norm(X_sims - X_obs, axis=1)
    n_accept = 100
    accepted_idx = np.argsort(distances)[:n_accept]

    theta_acc = theta_sims[accepted_idx]
    X_acc = X_sims[accepted_idx]

    print(f"Got {len(theta_acc)} accepted samples")
    print(f"Feature dimensions: {X_acc.shape[1]}")

    # Get transforms
    transforms = task._get_transforms(automatic_transforms_enabled=False)["parameters"]

    # Test our LRA implementation
    print("\nTesting our LRA implementation...")
    weights = None  # Uniform weights for now
    theta_ours = apply_lra(theta_acc, X_acc, X_obs, weights, transforms)

    # Test sbibm's run_lra
    print("Testing sbibm's run_lra...")
    theta_sbibm = run_lra(
        theta=theta_acc,
        x=X_acc,
        observation=X_obs,
        sample_weight=weights,
        transforms=transforms
    )

    # Compare results
    print(f"\nComparison:")
    print(f"Our shape: {theta_ours.shape}")
    print(f"sbibm shape: {theta_sbibm.shape}")

    diff = torch.abs(theta_ours - theta_sbibm)
    print(f"Max absolute difference: {torch.max(diff):.6f}")
    print(f"Mean absolute difference: {torch.mean(diff):.6f}")

    # Per-dimension comparison
    print("\nPer-dimension comparison:")
    for i in range(theta_ours.shape[1]):
        diff_dim = torch.abs(theta_ours[:, i] - theta_sbibm[:, i])
        print(f"  Dim {i}: max_diff={torch.max(diff_dim):.6f}, mean_diff={torch.mean(diff_dim):.6f}")

    # Correlation
    correlations = []
    for i in range(theta_ours.shape[1]):
        corr = torch.corrcoef(torch.stack([theta_ours[:, i], theta_sbibm[:, i]]))[0, 1]
        correlations.append(corr.item())
        print(f"  Dim {i}: correlation={corr:.6f}")

    print(f"\nOverall correlation: {np.mean(correlations):.6f}")

    # Check if they're essentially the same
    tolerance = 1e-5
    max_diff = torch.max(diff)
    if max_diff < tolerance:
        print(f"SUCCESS: Implementations match within tolerance {tolerance}")
    else:
        print(f"FAILURE: Max difference {max_diff:.6f} exceeds tolerance {tolerance}")

    return theta_ours, theta_sbibm


def test_with_weights():
    """Test with kernel weights"""
    print("\n" + "="*50)
    print("TESTING WITH KERNEL WEIGHTS")

    task = get_task("gaussian_linear")

    # Get accepted samples
    prior = task.get_prior()
    simulator = task.get_simulator(max_calls=1000)
    x_obs = task.get_observation(1)

    theta_sims = prior(500)
    x_sims = simulator(theta_sims)

    extractor = FeatureExtractor()
    X_sims = extractor.fit_transform(x_sims)
    X_obs = extractor.transform_single(x_obs.squeeze(0))

    distances = np.linalg.norm(X_sims - X_obs, axis=1)
    accepted_idx = np.argsort(distances)[:50]

    theta_acc = theta_sims[accepted_idx]
    X_acc = X_sims[accepted_idx]
    d_acc = distances[accepted_idx]

    transforms = task._get_transforms(automatic_transforms_enabled=False)["parameters"]

    # Test with weights
    weights = compute_weights(d_acc)

    print("Testing with kernel weights...")
    theta_ours_w = apply_lra(theta_acc, X_acc, X_obs, weights, transforms)
    theta_sbibm_w = run_lra(
        theta=theta_acc,
        x=X_acc,
        observation=X_obs,
        sample_weight=weights,
        transforms=transforms
    )

    diff_w = torch.abs(theta_ours_w - theta_sbibm_w)
    print(f"With weights - Max diff: {torch.max(diff_w):.6f}, Mean diff: {torch.mean(diff_w):.6f}")

    if torch.max(diff_w) < 1e-5:
        print("SUCCESS: Weighted LRA implementations match")
    else:
        print("FAILURE: Weighted LRA implementations differ")


if __name__ == "__main__":
    # Test on different tasks
    tasks_to_test = ["gaussian_linear", "two_moons"]

    for task_name in tasks_to_test:
        validate_lra_implementations(task_name, 1)

    test_with_weights()
