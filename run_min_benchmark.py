import numpy as np
import torch
import sbibm
from sbibm.metrics.c2st import c2st
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def simulate_in_batches(simulator, theta: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    xs = []
    for i in range(0, theta.shape[0], batch_size):
        xs.append(simulator(theta[i:i+batch_size]))
    return torch.cat(xs, dim=0)

def kde_sample_with_positive_cv(theta_acc_np: np.ndarray, num_samples: int, seed: int = 0):
    """
    Choose KDE bandwidth by CV over strictly positive grid, then sample.
    Returns: samples_np, best_bw
    """
    if theta_acc_np.ndim != 2:
        raise ValueError(f"theta_acc must be 2D, got shape {theta_acc_np.shape}")

    # Positive-only bandwidth grid
    bw_grid = np.logspace(-2, 0.8, 25)  # ~0.01 to ~6.3

    # If very small accepted set, reduce folds
    cv_folds = min(5, len(theta_acc_np))
    if cv_folds < 2:
        # Not enough points for CV; fallback to silverman-like scale
        # (simple heuristic: median pairwise distance)
        d = np.sqrt(((theta_acc_np[:, None, :] - theta_acc_np[None, :, :]) ** 2).sum(-1))
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        best_bw = float(med) if np.isfinite(med) and med > 0 else 1.0
        kde_best = KernelDensity(kernel="gaussian", bandwidth=best_bw).fit(theta_acc_np)
        samples = kde_best.sample(num_samples, random_state=np.random.RandomState(seed))
        return samples, best_bw

    gs = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        param_grid={"bandwidth": bw_grid},
        cv=cv_folds,
        n_jobs=-1,
        error_score="raise",
    )
    gs.fit(theta_acc_np)
    best_bw = float(gs.best_params_["bandwidth"])

    kde_best = KernelDensity(kernel="gaussian", bandwidth=best_bw)
    kde_best.fit(theta_acc_np)

    samples = kde_best.sample(num_samples, random_state=np.random.RandomState(seed))
    return samples, best_bw

def rej_abc_no_warning(
    task,
    num_observation: int,
    num_simulations: int,
    num_samples: int,
    num_top: int = 100,
    batch_size: int = 1024,
    seed: int = 0,
):
    """
    Simple REJ-ABC:
    - sample theta ~ prior
    - simulate x
    - accept top-k by L2 distance in x-space
    - KDE on accepted theta with positive-only CV bandwidth
    - sample num_samples from KDE
    """
    set_seeds(seed)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()
    x_obs = task.get_observation(num_observation)

    # Sample theta and simulate
    theta = prior.sample((num_simulations,))  # shape (N, d_theta)
    x = simulate_in_batches(simulator, theta, batch_size=batch_size)

    # Flatten x and x_obs for distance
    x_flat = x.reshape(num_simulations, -1)
    x_obs_flat = x_obs.reshape(1, -1)

    # L2 distances
    d = torch.norm(x_flat - x_obs_flat, dim=1)

    # Accept top-k
    k = min(num_top, num_simulations)
    idx = torch.topk(d, k=k, largest=False).indices
    theta_acc = theta[idx]  # (k, d_theta)

    # KDE resample with positive-only CV bandwidth
    theta_acc_np = theta_acc.detach().cpu().numpy()
    theta_samp_np, best_bw = kde_sample_with_positive_cv(theta_acc_np, num_samples, seed=seed)

    theta_samp = torch.as_tensor(theta_samp_np, dtype=theta.dtype)
    meta = {"num_top": k, "best_bw": best_bw, "max_dist_acc": float(d[idx].max().item())}
    return theta_samp, num_simulations, meta

# ---- Run across obs 1..10 and compute C2ST ----
task = sbibm.get_task("two_moons")
num_sims = 10_000
num_samples = 10_000

scores = []
for obs in range(1, 11):
    algo_samples, sim_calls, meta = rej_abc_no_warning(
        task=task,
        num_observation=obs,
        num_simulations=num_sims,
        num_samples=num_samples,
        num_top=100,        # matches common benchmark setting
        batch_size=2048,
        seed=0,
    )
    ref = task.get_reference_posterior_samples(num_observation=obs)

    score = float(c2st(ref, algo_samples, seed=0))
    scores.append(score)
    print("obs", obs, "C2ST", score, "sim_calls", sim_calls, "meta", meta)

print("mean", float(np.mean(scores)), "std", float(np.std(scores)))
