import math
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


BW_GRID = np.logspace(-2, 0.8, 25)
KDE_CV_FOLDS_MAX = 5


def weighted_mean_std(X: np.ndarray, w: np.ndarray | None = None, eps: float = 1e-12):
    if w is None:
        mu = X.mean(axis=0)
        var = X.var(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
    else:
        w = np.asarray(w, dtype=float)
        w = np.clip(w, 0.0, None)
        wsum = float(w.sum())
        if wsum <= eps:
            mu = X.mean(axis=0)
            var = X.var(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
        else:
            mu = (w[:, None] * X).sum(axis=0) / wsum
            var = (w[:, None] * (X - mu[None, :]) ** 2).sum(axis=0) / wsum

    std = np.sqrt(np.maximum(var, 0.0))
    std = np.where(std > eps, std, 1.0)
    return mu, std


def distance_weights(d_acc: np.ndarray | None) -> np.ndarray | None:
    if d_acc is None:
        return None
    eps = float(np.max(d_acc))
    if not math.isfinite(eps) or eps <= 0:
        return None
    u = np.clip(d_acc / (eps + 1e-12), 0.0, 1.0)
    return 1.0 - u**2


def linear_reg_adjust(
    theta_acc: np.ndarray,
    x_acc: np.ndarray,
    x_obs: np.ndarray,
    d_acc: np.ndarray | None,
    smc_weights: np.ndarray | None,
    use_distance_weights: bool,
    standardize_x: bool,
) -> np.ndarray:
    # Combine SMC weights and distance weights (multiply then normalize)
    w = None
    if smc_weights is not None:
        w = np.asarray(smc_weights, dtype=float)
        w = np.clip(w, 0.0, None)

    if use_distance_weights and (d_acc is not None):
        dw = distance_weights(d_acc)
        if dw is not None:
            w = dw if w is None else w * dw

    if standardize_x:
        mu, std = weighted_mean_std(x_acc, w=w)
        x_acc_s = (x_acc - mu[None, :]) / std[None, :]
        x_obs_s = (x_obs - mu) / std
    else:
        x_acc_s = x_acc
        x_obs_s = x_obs

    X = x_acc_s - x_obs_s[None, :]
    Y = theta_acc

    reg = LinearRegression()
    reg.fit(X, Y, sample_weight=w)
    B = reg.coef_.T
    theta_adj = Y - X @ B
    return theta_adj


def kde_sample_cv(theta_np: np.ndarray, num_samples: int, seed: int) -> tuple[np.ndarray, float]:
    if theta_np.ndim != 2:
        raise ValueError(f"theta_np must be 2D, got {theta_np.shape}")

    cv_folds = min(KDE_CV_FOLDS_MAX, len(theta_np))
    if cv_folds < 2:
        best_bw = 0.1
    else:
        gs = GridSearchCV(
            KernelDensity(kernel="gaussian"),
            param_grid={"bandwidth": BW_GRID},
            cv=cv_folds,
            n_jobs=-1,
            error_score="raise",
        )
        gs.fit(theta_np)
        best_bw = float(gs.best_params_["bandwidth"])

    kde = KernelDensity(kernel="gaussian", bandwidth=best_bw).fit(theta_np)
    samp = kde.sample(num_samples, random_state=np.random.RandomState(seed))
    return samp, best_bw


def in_support_mask(prior_dist, theta_np: np.ndarray) -> np.ndarray:
    theta_t = torch.as_tensor(theta_np, dtype=torch.float32)
    with torch.no_grad():
        lp = prior_dist.log_prob(theta_t)
    lp_np = lp.detach().cpu().numpy()
    return np.isfinite(lp_np)


def resample_kde_with_bounds(prior_dist, theta_adj: np.ndarray, num_samples: int, seed: int):
    """
    KDE resampling with reject/resample to enforce parameter support.
    """
    collected = []
    remaining = num_samples
    # Draw in batches until enough in-bounds samples are collected.
    while remaining > 0:
        batch = max(remaining, 1000)
        samp_np, bw = kde_sample_cv(theta_adj, batch, seed=seed)
        mask = in_support_mask(prior_dist, samp_np)
        if mask.any():
            accepted = samp_np[mask]
            collected.append(accepted)
            remaining -= accepted.shape[0]
        else:
            # Avoid infinite loop if KDE is badly outside support.
            remaining -= 0
    out = np.concatenate(collected, axis=0)[:num_samples]
    return out, bw

