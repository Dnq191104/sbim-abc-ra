import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import sbibm


class FeatureExtractor:
    """
    Handles feature extraction and standardization for ABC.
    Computes statistics across all simulated samples for standardization.
    """

    def __init__(self):
        self.scaler = None
        self.fitted = False

    def fit(self, X_batch: torch.Tensor):
        """
        Fit the feature extractor on a batch of simulated data.

        Args:
            X_batch: Simulated data, shape (batch_size, ...)
        """
        # Flatten each sample to 1D vector
        X_flat = np.array([x.flatten().numpy() for x in X_batch])

        # Fit standardizer
        self.scaler = StandardScaler()
        self.scaler.fit(X_flat)
        self.fitted = True

        return X_flat

    def transform(self, X_batch: torch.Tensor) -> np.ndarray:
        """
        Transform batch of data using fitted statistics.

        Args:
            X_batch: Data to transform, shape (batch_size, ...)

        Returns:
            Standardized features, shape (batch_size, feature_dim)
        """
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")

        # Flatten each sample to 1D vector
        X_flat = np.array([x.flatten().numpy() for x in X_batch])

        # Apply standardization
        return self.scaler.transform(X_flat)

    def transform_single(self, x: torch.Tensor) -> np.ndarray:
        """
        Transform a single observation.

        Args:
            x: Single observation, shape (...)

        Returns:
            Standardized features, shape (feature_dim,)
        """
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")

        x_flat = x.flatten().numpy().reshape(1, -1)
        return self.scaler.transform(x_flat)[0]

    def fit_transform(self, X_batch: torch.Tensor) -> np.ndarray:
        """Fit and transform in one step."""
        X_flat = self.fit(X_batch)
        return self.scaler.transform(X_flat)


def phi_batch(X_batch: torch.Tensor, extractor: FeatureExtractor = None) -> np.ndarray:
    """
    Extract features from a batch of simulation outputs.

    Args:
        X_batch: Raw simulation outputs, shape (batch_size, ...)
        extractor: Optional fitted FeatureExtractor for standardization

    Returns:
        Features, shape (batch_size, feature_dim)
    """
    # Flatten each sample to 1D vector
    X_flat = np.array([x.flatten().numpy() for x in X_batch])

    if extractor is not None and extractor.fitted:
        return extractor.transform(torch.tensor(X_flat).unsqueeze(1))
    else:
        return X_flat


def simulate_batch(task, prior, simulator, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate a batch of theta ~ prior, x = simulator(theta)

    Returns:
        theta_batch: (batch_size, dim_theta)
        x_batch: (batch_size, ...)
    """
    theta_batch = prior(batch_size)
    x_batch = simulator(theta_batch)
    return theta_batch, x_batch


def compute_distances(X: np.ndarray, X_obs: np.ndarray) -> np.ndarray:
    """
    Compute L2 distances between simulated summaries and observed summary.

    Args:
        X: Simulated summaries, shape (N, d_x)
        X_obs: Observed summary, shape (d_x,)

    Returns:
        distances: shape (N,)
    """
    return np.linalg.norm(X - X_obs, axis=1)


def select_accepted_samples(theta_all: torch.Tensor, X_all: np.ndarray, distances: np.ndarray,
                           accept_frac: float = 0.01, min_accept: int = 200) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Select accepted samples based on distance threshold.

    Args:
        theta_all: All simulated parameters (N, dim_theta)
        X_all: All simulated summaries (N, d_x)
        distances: Distances to observed (N,)
        accept_frac: Fraction of samples to accept
        min_accept: Minimum number to accept

    Returns:
        theta_acc, X_acc, d_acc: Accepted samples
    """
    N = len(distances)
    n_acc = max(min_accept, int(accept_frac * N))

    # Get indices of smallest distances
    accepted_indices = np.argsort(distances)[:n_acc]

    return theta_all[accepted_indices], X_all[accepted_indices], distances[accepted_indices]


def compute_weights(distances: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
    """
    Compute Epanechnikov kernel weights.

    Args:
        distances: Distances of accepted samples
        eps: Bandwidth parameter (default: max distance)

    Returns:
        weights: shape (n_acc,)
    """
    if eps is None:
        eps = np.max(distances)

    u = distances / eps
    weights = np.maximum(1 - u**2, 0)  # Epanechnikov kernel

    return weights


def apply_lra(theta_acc: torch.Tensor, X_acc: np.ndarray, X_obs: np.ndarray,
              weights: Optional[np.ndarray] = None, transforms = None) -> torch.Tensor:
    """
    Apply Linear Regression Adjustment exactly like sbibm's run_lra.

    Args:
        theta_acc: Accepted parameters (n_acc, dim_theta)
        X_acc: Accepted summaries (n_acc, d_x)
        X_obs: Observed summary (d_x,)
        weights: Sample weights (n_acc,) or None
        transforms: Parameter transforms

    Returns:
        theta_adjusted: Adjusted parameters (n_acc, dim_theta)
    """
    theta_np = theta_acc.numpy()
    n_acc, dim_theta = theta_np.shape

    # Apply transforms if provided
    if transforms is not None:
        theta_transformed = transforms(theta_acc).numpy()
    else:
        theta_transformed = theta_np.copy()

    # Adjust each parameter dimension
    for j in range(dim_theta):
        # Fit weighted linear regression
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X_acc, theta_transformed[:, j], sample_weight=weights)

        # Adjust: theta_adj = theta + pred(X_obs) - pred(X)
        pred_obs = reg.predict(X_obs.reshape(1, -1))[0]
        pred_acc = reg.predict(X_acc)

        theta_transformed[:, j] += pred_obs - pred_acc

    # Apply inverse transforms
    if transforms is not None:
        theta_adjusted = transforms.inv(torch.tensor(theta_transformed, dtype=torch.float32))
    else:
        theta_adjusted = torch.tensor(theta_transformed, dtype=torch.float32)

    return theta_adjusted


def resample_posterior(theta_adj: torch.Tensor, weights: Optional[np.ndarray],
                      num_samples: int) -> torch.Tensor:
    """
    Resample from adjusted posterior to get desired number of samples.

    Args:
        theta_adj: Adjusted parameters (n_acc, dim_theta)
        weights: Importance weights (n_acc,) or None for uniform
        num_samples: Desired number of posterior samples

    Returns:
        theta_post: Posterior samples (num_samples, dim_theta)
    """
    n_acc = theta_adj.shape[0]

    if weights is None:
        # Uniform sampling
        indices = torch.randint(0, n_acc, (num_samples,))
    else:
        # Weighted sampling with replacement
        weights_norm = weights / np.sum(weights)
        indices = np.random.choice(n_acc, size=num_samples, p=weights_norm)

    return theta_adj[indices]


def run_abc_ra(task, num_observation: int, num_simulations: int, num_posterior_samples: int,
               accept_frac: float = 0.01, min_accept: int = 200,
               use_weights: bool = True, batch_size: int = 1000) -> torch.Tensor:
    """
    Run Rejection ABC with Linear Regression Adjustment.

    Args:
        task: sbibm Task instance
        num_observation: Which observation to use (1-10)
        num_simulations: Total simulation budget
        num_posterior_samples: Number of posterior samples to return
        accept_frac: Fraction of simulations to accept
        min_accept: Minimum number of accepted samples
        use_weights: Whether to use kernel weights in LRA
        batch_size: Batch size for simulation

    Returns:
        theta_posterior: Posterior samples (num_posterior_samples, dim_theta)
    """

    # Get task components
    prior = task.get_prior()
    simulator = task.get_simulator(max_calls=num_simulations)
    x_obs = task.get_observation(num_observation)

    # Get transforms
    transforms = task._get_transforms(automatic_transforms_enabled=False)["parameters"]

    # Initialize feature extractor
    extractor = FeatureExtractor()

    print(f"Simulating {num_simulations} samples...")

    # Step 1: Simulate under budget and fit feature extractor
    theta_all = []
    x_all_raw = []

    remaining = num_simulations
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        theta_batch, x_batch = simulate_batch(task, prior, simulator, current_batch)

        theta_all.append(theta_batch)
        x_all_raw.append(x_batch)

        remaining -= current_batch

    theta_all = torch.cat(theta_all, dim=0)
    x_all_raw = torch.cat(x_all_raw, dim=0)

    # Fit feature extractor on all simulated data
    X_all = extractor.fit_transform(x_all_raw)

    # Transform observation
    X_obs = extractor.transform_single(x_obs.squeeze(0))  # x_obs is [1, dim], squeeze to [dim]

    print(f"Simulated {theta_all.shape[0]} samples, feature dim {X_all.shape[1]}")

    # Step 2: Rejection step
    distances = compute_distances(X_all, X_obs)
    theta_acc, X_acc, d_acc = select_accepted_samples(theta_all, X_all, distances,
                                                      accept_frac, min_accept)

    print(f"Accepted {theta_acc.shape[0]} samples, max distance: {np.max(d_acc):.4f}")

    # Step 3: Compute weights
    weights = compute_weights(d_acc) if use_weights else None

    # Step 4: Apply LRA
    theta_adj = apply_lra(theta_acc, X_acc, X_obs, weights, transforms)

    print(f"Applied LRA, adjusted samples shape: {theta_adj.shape}")

    # Step 5: Resample to get desired number of posterior samples
    theta_post = resample_posterior(theta_adj, weights, num_posterior_samples)

    print(f"Final posterior samples: {theta_post.shape}")

    return theta_post


# Quick test function
def test_abc_ra():
    """Test the implementation on a small task"""
    from sbibm.tasks import get_task

    task = get_task("gaussian_linear")

    theta_post = run_abc_ra(
        task=task,
        num_observation=1,
        num_simulations=1000,
        num_posterior_samples=100,
        accept_frac=0.05,
        min_accept=50
    )

    print(f"Test completed. Posterior shape: {theta_post.shape}")
    print(f"Posterior mean: {theta_post.mean(dim=0)}")
    print(f"Posterior std: {theta_post.std(dim=0)}")

    return theta_post


if __name__ == "__main__":
    test_abc_ra()
