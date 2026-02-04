import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

@torch.no_grad()
def simulate_in_batches(simulator, theta: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    xs = []
    for i in range(0, theta.shape[0], batch_size):
        xs.append(simulator(theta[i:i+batch_size]))
    return torch.cat(xs, dim=0)

def standardize(X: torch.Tensor, x_obs: torch.Tensor, eps: float = 1e-8):
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True)
    sd = torch.where(sd < eps, torch.ones_like(sd), sd)  # avoid divide-by-zero
    Xs = (X - mu) / sd
    xos = (x_obs - mu.squeeze(0)) / sd.squeeze(0)
    return Xs, xos, mu.squeeze(0), sd.squeeze(0)

def epanechnikov_weights(d_acc: torch.Tensor) -> torch.Tensor:
    eps = torch.max(d_acc)
    if float(eps) <= 0:
        return torch.ones_like(d_acc)
    u = d_acc / eps
    w = torch.clamp(1.0 - u*u, min=0.0)
    if float(w.sum()) <= 0:
        w = torch.ones_like(d_acc)
    return w

def run_lra_like_sbibm(theta_acc: torch.Tensor, X_acc: torch.Tensor, x_obs: torch.Tensor, w=None):
    """
    theta_adj = theta + pred(x_obs) - pred(X)
    Implements the same adjustment rule as sbibm's run_lra.
    """
    theta_adj = theta_acc.clone()

    X_np = X_acc.detach().cpu().numpy()
    xo_np = x_obs.detach().cpu().numpy().reshape(1, -1)
    w_np = None if w is None else w.detach().cpu().numpy()

    # Multi-output regression: y is (n, d_theta)
    y_np = theta_acc.detach().cpu().numpy()
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X_np, y_np, sample_weight=w_np)

    pred_xo = reg.predict(xo_np)          # (1, d_theta)
    pred_X  = reg.predict(X_np)           # (n, d_theta)

    theta_adj_np = y_np + pred_xo - pred_X
    return torch.as_tensor(theta_adj_np, dtype=theta_acc.dtype, device=theta_acc.device)

def kde_sample_pos_cv(theta_np: np.ndarray, num_samples: int, seed: int = 0):
    # strictly positive grid => no negative bw warnings
    bw_grid = np.logspace(-2, 0.8, 25)  # ~0.01..6.3

    cv_folds = min(5, len(theta_np))
    if cv_folds < 2:
        best_bw = 0.1
        kde = KernelDensity(kernel="gaussian", bandwidth=best_bw).fit(theta_np)
        samples = kde.sample(num_samples, random_state=np.random.RandomState(seed))
        return samples, best_bw

    gs = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": bw_grid},
        cv=cv_folds,
        n_jobs=-1,
        error_score="raise",
    )
    gs.fit(theta_np)
    best_bw = float(gs.best_params_["bandwidth"])

    kde = KernelDensity(kernel="gaussian", bandwidth=best_bw)
    kde.fit(theta_np)
    samples = kde.sample(num_samples, random_state=np.random.RandomState(seed))
    return samples, best_bw

def rej_abc_ra(
    task,
    num_observation: int,
    num_simulations: int,
    num_samples: int,
    num_top: int = 100,
    feature_standardize: bool = True,
    use_kernel_weights: bool = True,
    kde_bandwidth: str | float = "pos_cv",
    seed: int = 0,
):
    # seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    prior = task.get_prior_dist()
    simulator = task.get_simulator()
    x_obs = task.get_observation(num_observation)

    # 1) simulate bank
    theta = prior.sample((num_simulations,))
    x = simulate_in_batches(simulator, theta)

    # 2) features (flatten)
    X = x.reshape(num_simulations, -1)
    xo = x_obs.reshape(-1)

    # 3) standardize features (recommended)
    if feature_standardize:
        X, xo, mu, sd = standardize(X, xo)
    else:
        mu, sd = None, None

    # 4) distances + accept top-k
    d = torch.norm(X - xo.unsqueeze(0), dim=1)
    k = min(num_top, num_simulations)
    idx = torch.topk(d, k=k, largest=False).indices

    theta_acc = theta[idx]
    X_acc = X[idx]
    d_acc = d[idx]

    # 5) weights
    w = epanechnikov_weights(d_acc) if use_kernel_weights else None

    # 6) regression adjustment
    theta_adj = run_lra_like_sbibm(theta_acc, X_acc, xo, w=w)

    # 7) KDE resample
    theta_adj_np = theta_adj.detach().cpu().numpy()

    if kde_bandwidth == "pos_cv":
        theta_samp_np, bw = kde_sample_pos_cv(theta_adj_np, num_samples, seed=seed)
    elif isinstance(kde_bandwidth, (float, int)):
        bw = float(kde_bandwidth)
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(theta_adj_np)
        theta_samp_np = kde.sample(num_samples, random_state=np.random.RandomState(seed))
    elif kde_bandwidth in ("scott", "silverman"):
        # sklearn KernelDensity doesn't accept "scott"/"silverman" strings directly.
        # If you want these rules, compute bw as float yourself. For now, force pos_cv or float.
        raise ValueError("Use kde_bandwidth='pos_cv' or a positive float for KDE bandwidth.")
    else:
        raise ValueError(f"Unknown kde_bandwidth={kde_bandwidth}")

    theta_samp = torch.as_tensor(theta_samp_np, dtype=theta.dtype)

    meta = {
        "num_top": k,
        "feature_standardize": feature_standardize,
        "use_kernel_weights": use_kernel_weights,
        "kde_bandwidth": bw,
        "max_dist_acc": float(d_acc.max().item()),
        "ess": float((w.sum().item() ** 2) / (w.pow(2).sum().item())) if w is not None else None,
    }
    return theta_samp, num_simulations, meta
