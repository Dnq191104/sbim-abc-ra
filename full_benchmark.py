import time
import math
import numpy as np
import torch
import sbibm
from sbibm.metrics.c2st import c2st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


# ============================================================
# Config (paper-like)
# ============================================================
TASKS = [
    "gaussian_linear",
    "gaussian_linear_uniform",
    "slcp",
    "slcp_distractors",
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "gaussian_mixture",
    "two_moons",
    "sir",
    "lotka_volterra",
]

BUDGETS = [1_000, 10_000, 100_000]   # paper shows 10^3, 10^4, 10^5
OBS_IDS = list(range(1, 11))        # 10 observations
NUM_POST_SAMPLES = 10_000           # paper uses 10k reference samples; we match output size
NUM_TOP = 100                       # common REJ-ABC choice in sbibm usage
BATCH_SIZE_SIM = 2048

# Metric settings
COMPUTE_MMD_MEDDIST_ONLY_FOR_TWO_MOONS = True
MMD_SUBSAMPLE = 2000       # set to None to use all 10k (can be expensive)
MEDDIST_PPC_SIMS = 2000    # posterior predictive sims for meddist (set 10_000 to match size, but expensive)

# Algorithm settings for RA
USE_DISTANCE_WEIGHTS = True
STANDARDIZE_X_FOR_RA = True

# KDE settings (CV only, as you requested)
BW_GRID = np.logspace(-2, 0.8, 25)  # ~0.01..~6.3
KDE_CV_FOLDS_MAX = 5

# Repetitions: paper averages over 10 observations; algorithm randomness is separate.
# If you want, set SEEDS=range(10) to average over RNG too.
SEEDS = [0]  # change to list(range(10)) if you want multiple RNG seeds


# ============================================================
# Utilities
# ============================================================
def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def simulate_in_batches(simulator, theta: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    xs = []
    for i in range(0, theta.shape[0], batch_size):
        xs.append(simulator(theta[i:i+batch_size]))
    return torch.cat(xs, dim=0)

def t_crit_975(df: int) -> float:
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.975, df))
    except Exception:
        return 1.96  # normal approx fallback

def mean_ci95(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, (m, m)
    s = float(x.std(ddof=1))
    half = t_crit_975(n - 1) * s / math.sqrt(n)
    return m, (m - half, m + half)


# ============================================================
# KDE (CV-only)
# ============================================================
def kde_sample_cv(theta_np: np.ndarray, num_samples: int, seed: int) -> tuple[np.ndarray, float]:
    if theta_np.ndim != 2:
        raise ValueError(f"theta_np must be 2D, got {theta_np.shape}")

    cv_folds = min(KDE_CV_FOLDS_MAX, len(theta_np))
    if cv_folds < 2:
        # rare with k=100; keep runnable
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


# ============================================================
# Regression adjustment (Beaumont-style local linear)
# ============================================================
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

def linear_reg_adjust(
    theta_acc: np.ndarray,
    x_acc: np.ndarray,
    x_obs: np.ndarray,
    d_acc: np.ndarray | None,
    use_distance_weights: bool,
    standardize_x: bool,
) -> np.ndarray:
    # weights (Epanechnikov-style)
    w = None
    if use_distance_weights and (d_acc is not None):
        eps = float(np.max(d_acc))
        u = np.clip(d_acc / (eps + 1e-12), 0.0, 1.0)
        w = 1.0 - u**2

    # standardize x for numerical stability
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
    B = reg.coef_.T  # (d_x, d_theta)

    theta_adj = Y - X @ B
    return theta_adj


# ============================================================
# REJ-ABC core (paired simulation for baseline + RA)
# ============================================================
def simulate_and_accept_topk(task, obs_id: int, num_simulations: int, num_top: int, batch_size: int, seed: int):
    set_seeds(seed)

    prior = task.get_prior()
    simulator = task.get_simulator()
    x_obs = task.get_observation(num_observation=obs_id)

    theta = prior(num_samples=num_simulations)
    x = simulate_in_batches(simulator, theta, batch_size=batch_size)

    x_flat = x.reshape(num_simulations, -1)
    x_obs_flat = x_obs.reshape(-1)

    d = torch.norm(x_flat - x_obs_flat[None, :], dim=1)

    k = min(int(num_top), int(num_simulations))
    idx = torch.topk(d, k=k, largest=False).indices

    theta_acc = theta[idx].detach().cpu().numpy()
    x_acc = x_flat[idx].detach().cpu().numpy()
    d_acc = d[idx].detach().cpu().numpy()
    x_obs_np = x_obs_flat.detach().cpu().numpy()

    meta = {
        "k": k,
        "eps_max": float(d[idx].max().item()),
    }
    return theta_acc, x_acc, x_obs_np, d_acc, meta


def run_rej_abc_and_ra(task, obs_id: int, budget: int, num_samples_out: int, seed: int):
    # Paired simulation
    theta_acc, x_acc, x_obs_np, d_acc, meta_sim = simulate_and_accept_topk(
        task=task,
        obs_id=obs_id,
        num_simulations=budget,
        num_top=NUM_TOP,
        batch_size=BATCH_SIZE_SIM,
        seed=seed,
    )

    # Baseline REJ-ABC: KDE on accepted thetas
    t0 = time.perf_counter()
    rej_np, rej_bw = kde_sample_cv(theta_acc, num_samples_out, seed=seed)
    t_rej = time.perf_counter() - t0

    # RA: adjust then KDE
    t1 = time.perf_counter()
    theta_adj = linear_reg_adjust(
        theta_acc=theta_acc,
        x_acc=x_acc,
        x_obs=x_obs_np,
        d_acc=d_acc,
        use_distance_weights=USE_DISTANCE_WEIGHTS,
        standardize_x=STANDARDIZE_X_FOR_RA,
    )
    ra_np, ra_bw = kde_sample_cv(theta_adj, num_samples_out, seed=seed)
    t_ra = time.perf_counter() - t1

    meta = {
        **meta_sim,
        "rej_bw": float(rej_bw),
        "ra_bw": float(ra_bw),
        "rej_overhead_sec": float(t_rej),
        "ra_overhead_sec": float(t_ra),
    }

    # return torch
    rej = torch.as_tensor(rej_np, dtype=torch.float32)
    ra = torch.as_tensor(ra_np, dtype=torch.float32)
    return rej, ra, meta


# ============================================================
# Extra metrics (paper shows these esp. on Two Moons)
# ============================================================
def _subsample_rows(X: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    if n is None or X.shape[0] <= n:
        return X
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:n]
    return X[idx]

def mmd2_rbf_median_heuristic(X: torch.Tensor, Y: torch.Tensor, seed: int = 0, subsample: int | None = 2000) -> float:
    """
    MMD^2 with RBF kernel; bandwidth via median heuristic on combined set.
    Uses an unbiased estimator. Subsamples to control O(n^2) cost.
    """
    X = _subsample_rows(X, subsample, seed=seed)
    Y = _subsample_rows(Y, subsample, seed=seed+1)

    X = X.reshape(X.shape[0], -1).float()
    Y = Y.reshape(Y.shape[0], -1).float()

    Z = torch.cat([X, Y], dim=0)

    # median heuristic on a subsample of pairs (cheap)
    # compute pairwise distances to first 500 points (or fewer)
    m = min(500, Z.shape[0])
    Zm = Z[:m]
    d2 = torch.cdist(Zm, Zm, p=2.0) ** 2
    # take median of off-diagonal entries
    mask = ~torch.eye(m, dtype=torch.bool, device=d2.device)
    med = torch.median(d2[mask]).item()
    sigma2 = med if (med > 1e-12 and math.isfinite(med)) else 1.0
    gamma = 1.0 / (2.0 * sigma2)

    def k(a, b):
        return torch.exp(-gamma * (torch.cdist(a, b, p=2.0) ** 2))

    m = X.shape[0]
    n = Y.shape[0]

    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)

    # unbiased MMD^2
    mmd2 = (
        (Kxx.sum() - torch.diagonal(Kxx).sum()) / (m * (m - 1) + 1e-12)
        + (Kyy.sum() - torch.diagonal(Kyy).sum()) / (n * (n - 1) + 1e-12)
        - 2.0 * Kxy.mean()
    )
    return float(mmd2.item())

@torch.no_grad()
def meddist(task, theta_samples: torch.Tensor, obs_id: int, num_ppc: int, seed: int = 0) -> float:
    """
    MEDDIST = median || x_sim(theta) - x_obs || in x-space (posterior predictive check).
    """
    simulator = task.get_simulator()
    x_obs = task.get_observation(num_observation=obs_id).reshape(-1)

    # subsample theta_samples for PPC sims
    theta_ppc = _subsample_rows(theta_samples, num_ppc, seed=seed)
    x_sim = simulate_in_batches(simulator, theta_ppc, batch_size=min(2048, theta_ppc.shape[0]))
    x_sim = x_sim.reshape(x_sim.shape[0], -1)

    d = torch.norm(x_sim - x_obs[None, :], dim=1)
    return float(d.median().item())


# ============================================================
# Main benchmark runner
# ============================================================
def run_benchmark():
    results = []  # list of dict rows

    for task_name in TASKS:
        task = sbibm.get_task(task_name)

        for budget in BUDGETS:
            # per obs arrays (for CI across the 10 observations)
            c2st_rej = []
            c2st_ra = []
            mmd_rej = []
            mmd_ra = []
            med_rej = []
            med_ra = []

            for obs_id in OBS_IDS:
                # choose a deterministic seed per (task,budget,obs,seedrep)
                for seedrep in SEEDS:
                    seed = (abs(hash(task_name)) % 10_000_000) + 10_000 * seedrep + 100 * obs_id + (budget // 1000)

                    rej_samp, ra_samp, meta = run_rej_abc_and_ra(
                        task=task,
                        obs_id=obs_id,
                        budget=budget,
                        num_samples_out=NUM_POST_SAMPLES,
                        seed=seed,
                    )

                    ref = task.get_reference_posterior_samples(num_observation=obs_id)

                    # C2ST (paper’s primary reported metric)
                    c_rej = float(c2st(ref, rej_samp, seed=seedrep))
                    c_ra  = float(c2st(ref, ra_samp,  seed=seedrep))
                    c2st_rej.append(c_rej)
                    c2st_ra.append(c_ra)

                    # Optional extra metrics (paper shows prominently for Two Moons)
                    do_extras = (task_name == "two_moons") if COMPUTE_MMD_MEDDIST_ONLY_FOR_TWO_MOONS else True
                    if do_extras:
                        # MMD^2 on theta samples (subsampled by default)
                        m_rej = mmd2_rbf_median_heuristic(ref, rej_samp, seed=seedrep, subsample=MMD_SUBSAMPLE)
                        m_ra  = mmd2_rbf_median_heuristic(ref, ra_samp,  seed=seedrep, subsample=MMD_SUBSAMPLE)
                        mmd_rej.append(m_rej)
                        mmd_ra.append(m_ra)

                        # MEDDIST (posterior predictive)
                        md_rej = meddist(task, rej_samp, obs_id, num_ppc=MEDDIST_PPC_SIMS, seed=seedrep)
                        md_ra  = meddist(task, ra_samp,  obs_id, num_ppc=MEDDIST_PPC_SIMS, seed=seedrep)
                        med_rej.append(md_rej)
                        med_ra.append(md_ra)

                # Store per-observation rows too (useful for debugging)
                results.append({
                    "task": task_name,
                    "budget": budget,
                    "obs": obs_id,
                    "seed_reps": len(SEEDS),
                    "c2st_rej_mean_over_seedreps": float(np.mean(c2st_rej[-len(SEEDS):])),
                    "c2st_ra_mean_over_seedreps": float(np.mean(c2st_ra[-len(SEEDS):])),
                    "rej_bw": meta["rej_bw"],
                    "ra_bw": meta["ra_bw"],
                    "eps_max": meta["eps_max"],
                })

            # Summaries (mean + 95% CI across obs; if SEEDS>1, it’s across obs×seedrep)
            c2st_rej_m, c2st_rej_ci = mean_ci95(np.array(c2st_rej))
            c2st_ra_m,  c2st_ra_ci  = mean_ci95(np.array(c2st_ra))
            diff_m, diff_ci = mean_ci95(np.array(c2st_ra) - np.array(c2st_rej))

            print(f"\n[{task_name}] budget={budget}")
            print(f"  C2ST  REJ mean={c2st_rej_m:.4f} CI=[{c2st_rej_ci[0]:.4f},{c2st_rej_ci[1]:.4f}]")
            print(f"        RA  mean={c2st_ra_m:.4f} CI=[{c2st_ra_ci[0]:.4f},{c2st_ra_ci[1]:.4f}]")
            print(f"        (RA-REJ) mean={diff_m:.4f} CI=[{diff_ci[0]:.4f},{diff_ci[1]:.4f}]")

            do_extras = (task_name == "two_moons") if COMPUTE_MMD_MEDDIST_ONLY_FOR_TWO_MOONS else True
            if do_extras and len(mmd_rej) > 0:
                mmd_rej_m, mmd_rej_ci = mean_ci95(np.array(mmd_rej))
                mmd_ra_m,  mmd_ra_ci  = mean_ci95(np.array(mmd_ra))
                mmd_diff_m, mmd_diff_ci = mean_ci95(np.array(mmd_ra) - np.array(mmd_rej))

                med_rej_m, med_rej_ci = mean_ci95(np.array(med_rej))
                med_ra_m,  med_ra_ci  = mean_ci95(np.array(med_ra))
                med_diff_m, med_diff_ci = mean_ci95(np.array(med_ra) - np.array(med_rej))

                print(f"  MMD^2 REJ mean={mmd_rej_m:.6f} CI=[{mmd_rej_ci[0]:.6f},{mmd_rej_ci[1]:.6f}]")
                print(f"       RA  mean={mmd_ra_m:.6f} CI=[{mmd_ra_ci[0]:.6f},{mmd_ra_ci[1]:.6f}]")
                print(f"       (RA-REJ) mean={mmd_diff_m:.6f} CI=[{mmd_diff_ci[0]:.6f},{mmd_diff_ci[1]:.6f}]")

                print(f"  MED   REJ mean={med_rej_m:.6f} CI=[{med_rej_ci[0]:.6f},{med_rej_ci[1]:.6f}]")
                print(f"       RA  mean={med_ra_m:.6f} CI=[{med_ra_ci[0]:.6f},{med_ra_ci[1]:.6f}]")
                print(f"       (RA-REJ) mean={med_diff_m:.6f} CI=[{med_diff_ci[0]:.6f},{med_diff_ci[1]:.6f}]")

    # Save a CSV with the per-observation details (handy for plotting)
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv("rej_abc_vs_ra_benchmark_obslevel.csv", index=False)
        print("\nSaved: rej_abc_vs_ra_benchmark_obslevel.csv")
    except Exception as e:
        print("\nCould not write CSV (pandas missing?), but benchmark finished. Error:", str(e))


if __name__ == "__main__":
    run_benchmark()
