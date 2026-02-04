import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from typing import Optional
from abc_ra import FeatureExtractor


class LRAVariant:
    """
    Base class for different LRA regression variants.
    """

    def __init__(self, **kwargs):
        self.model = None
        self.trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the regression model"""
        raise NotImplementedError



class LinearLRA(LRAVariant):
    """Original linear regression adjustment"""

    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.trained = True


class RidgeLRA(LRAVariant):
    """Ridge regression for stability with high-dimensional or correlated features"""

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.model = Ridge(alpha=alpha, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.trained = True


class ElasticNetLRA(LRAVariant):
    """ElasticNet for feature selection and stability"""

    def __init__(self, alpha=1.0, l1_ratio=0.5, **kwargs):
        super().__init__()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.trained = True


class RandomForestLRA(LRAVariant):
    """Random Forest for non-linear regression adjustment"""

    def __init__(self, n_estimators=100, max_depth=None, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.trained = True


class NeuralNetLRA(LRAVariant):
    """Neural network (MLP) for non-linear regression adjustment"""

    def __init__(self, hidden_layer_sizes=(100, 50), max_iter=1000, **kwargs):
        super().__init__()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.trained = True


def apply_lra_variant(theta_acc: torch.Tensor, X_acc: np.ndarray, X_obs: np.ndarray,
                     lra_model: LRAVariant, weights: Optional[np.ndarray] = None,
                     transforms = None) -> torch.Tensor:
    """
    Apply Linear Regression Adjustment with any regression model variant.

    Args:
        theta_acc: Accepted parameters (n_acc, dim_theta)
        X_acc: Accepted summaries (n_acc, d_x)
        X_obs: Observed summary (d_x,)
        lra_model: LRA variant model (must be fitted)
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
        # Fit regression model for this parameter dimension
        if hasattr(lra_model.model, 'fit'):
            # sklearn model
            if weights is not None:
                lra_model.model.fit(X_acc, theta_transformed[:, j], sample_weight=weights)
            else:
                lra_model.model.fit(X_acc, theta_transformed[:, j])
            lra_model.trained = True

            # Make predictions
            pred_obs = lra_model.model.predict(X_obs.reshape(1, -1))[0]
            pred_acc = lra_model.model.predict(X_acc)
        else:
            # Custom model
            if weights is not None:
                lra_model.fit(X_acc, theta_transformed[:, j], sample_weight=weights)
            else:
                lra_model.fit(X_acc, theta_transformed[:, j])

            # Make predictions
            pred_obs = lra_model.predict(X_obs.reshape(1, -1))[0]
            pred_acc = lra_model.predict(X_acc)

        theta_transformed[:, j] += pred_obs - pred_acc

    # Apply inverse transforms
    if transforms is not None:
        theta_adjusted = transforms.inv(torch.tensor(theta_transformed, dtype=torch.float32))
    else:
        theta_adjusted = torch.tensor(theta_transformed, dtype=torch.float32)

    return theta_adjusted


def run_abc_ra_variant(task, num_observation: int, num_simulations: int, num_posterior_samples: int,
                       lra_variant: str = 'linear', accept_frac: float = 0.01, min_accept: int = 200,
                       use_weights: bool = True, batch_size: int = 1000, **lra_kwargs) -> torch.Tensor:
    """
    Run Rejection ABC with different LRA variants.

    Args:
        task: sbibm Task instance
        num_observation: Which observation to use (1-10)
        num_simulations: Total simulation budget
        num_posterior_samples: Number of posterior samples to return
        lra_variant: Which LRA variant to use ('linear', 'ridge', 'elasticnet', 'rf', 'mlp')
        accept_frac: Fraction of simulations to accept
        min_accept: Minimum number of accepted samples
        use_weights: Whether to use kernel weights in LRA
        batch_size: Batch size for simulation
        **lra_kwargs: Additional arguments for the LRA variant

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

    print(f"Simulating {num_simulations} samples with {lra_variant} LRA...")

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
    X_obs = extractor.transform_single(x_obs.squeeze(0))

    print(f"Simulated {theta_all.shape[0]} samples, feature dim {X_all.shape[1]}")

    # Step 2: Rejection step
    distances = compute_distances(X_all, X_obs)
    theta_acc, X_acc, d_acc = select_accepted_samples(theta_all, X_all, distances,
                                                      accept_frac, min_accept)

    print(f"Accepted {theta_acc.shape[0]} samples, max distance: {np.max(d_acc):.4f}")

    # Step 3: Compute weights
    weights = compute_weights(d_acc) if use_weights else None

    # Step 4: Apply LRA with chosen variant
    if lra_variant == 'linear':
        lra_model = LinearLRA(**lra_kwargs)
    elif lra_variant == 'ridge':
        lra_model = RidgeLRA(**lra_kwargs)
    elif lra_variant == 'elasticnet':
        lra_model = ElasticNetLRA(**lra_kwargs)
    elif lra_variant == 'rf':
        lra_model = RandomForestLRA(**lra_kwargs)
    elif lra_variant == 'mlp':
        lra_model = NeuralNetLRA(**lra_kwargs)
    else:
        raise ValueError(f"Unknown LRA variant: {lra_variant}")

    theta_adj = apply_lra_variant(theta_acc, X_acc, X_obs, lra_model, weights, transforms)

    print(f"Applied {lra_variant} LRA, adjusted samples shape: {theta_adj.shape}")

    # Step 5: Resample to get desired number of posterior samples
    theta_post = resample_posterior(theta_adj, weights, num_posterior_samples)

    print(f"Final posterior samples: {theta_post.shape}")

    return theta_post


def simulate_batch(task, prior, simulator, batch_size: int):
    """Helper function to simulate a batch"""
    theta_batch = prior(batch_size)
    x_batch = simulator(theta_batch)
    return theta_batch, x_batch


def compute_distances(X: np.ndarray, X_obs: np.ndarray) -> np.ndarray:
    """Compute L2 distances"""
    return np.linalg.norm(X - X_obs, axis=1)


def select_accepted_samples(theta_all: torch.Tensor, X_all: np.ndarray, distances: np.ndarray,
                           accept_frac: float = 0.01, min_accept: int = 200):
    """Select accepted samples based on distance threshold"""
    N = len(distances)
    n_acc = max(min_accept, int(accept_frac * N))

    # Get indices of smallest distances
    accepted_indices = np.argsort(distances)[:n_acc]

    return theta_all[accepted_indices], X_all[accepted_indices], distances[accepted_indices]


def compute_weights(distances: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
    """Compute Epanechnikov kernel weights"""
    if eps is None:
        eps = np.max(distances)

    u = distances / eps
    weights = np.maximum(1 - u**2, 0)  # Epanechnikov kernel

    return weights


def resample_posterior(theta_adj: torch.Tensor, weights: Optional[np.ndarray],
                      num_samples: int) -> torch.Tensor:
    """Resample from adjusted posterior"""
    n_acc = theta_adj.shape[0]

    if weights is None:
        # Uniform sampling
        indices = torch.randint(0, n_acc, (num_samples,))
    else:
        # Weighted sampling with replacement
        weights_norm = weights / np.sum(weights)
        indices = np.random.choice(n_acc, size=num_samples, p=weights_norm)

    return theta_adj[indices]


# Test functions
def test_lra_variants():
    """Test different LRA variants on a small task"""
    from sbibm.tasks import get_task
    from sbibm.metrics.c2st import c2st

    task = get_task("gaussian_linear")
    theta_ref = task.get_reference_posterior_samples(num_observation=1)

    variants = ['linear', 'ridge', 'elasticnet', 'rf', 'mlp']
    results = {}

    for variant in variants:
        print(f"\nTesting {variant} LRA...")
        try:
            theta_post = run_abc_ra_variant(
                task=task,
                num_observation=1,
                num_simulations=2000,
                num_posterior_samples=500,
                lra_variant=variant,
                accept_frac=0.05,
                min_accept=50
            )

            c2st_score = c2st(theta_ref, theta_post)
            results[variant] = float(c2st_score)
            print(".4f")

        except Exception as e:
            print(f"Failed: {e}")
            results[variant] = None

    print("\n" + "="*50)
    print("LRA VARIANT COMPARISON")
    print("="*50)
    for variant, score in results.items():
        if score is not None:
            print(".4f")
        else:
            print(f"{variant}: FAILED")

    return results


if __name__ == "__main__":
    test_lra_variants()
