import numpy as np
import torch
from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st
from sbibm.algorithms import snpe as npe

from ra_core import (
    kde_sample_cv,
    linear_reg_adjust,
    resample_kde_with_bounds,
)


def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def simulate_in_batches(simulator, theta: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    xs = []
    for i in range(0, theta.shape[0], batch_size):
        xs.append(simulator(theta[i:i + batch_size]))
    return torch.cat(xs, dim=0)


def _linear_reg_adjust(theta_acc, x_acc, x_obs, d_acc):
    return linear_reg_adjust(
        theta_acc=theta_acc,
        x_acc=x_acc,
        x_obs=x_obs,
        d_acc=d_acc,
        smc_weights=None,
        use_distance_weights=True,
        standardize_x=True,
    )


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

    return theta_acc, x_acc, x_obs_np, d_acc


def run_rej_abc_and_ra(task, obs_id: int, budget: int, num_samples_out: int, seed: int):
    theta_acc, x_acc, x_obs_np, d_acc = simulate_and_accept_topk(
        task=task,
        obs_id=obs_id,
        num_simulations=budget,
        num_top=100,
        batch_size=2048,
        seed=seed,
    )

    rej_np, _ = kde_sample_cv(theta_acc, num_samples_out, seed=seed)

    theta_adj = _linear_reg_adjust(theta_acc, x_acc, x_obs_np, d_acc)
    prior_dist = task.get_prior_dist()
    ra_np, _ = resample_kde_with_bounds(prior_dist, theta_adj, num_samples_out, seed=seed)

    rej = torch.as_tensor(rej_np, dtype=torch.float32)
    ra = torch.as_tensor(ra_np, dtype=torch.float32)
    return rej, ra


NUM_POST_SAMPLES = 10_000


def evaluate_methods(task_name="gaussian_linear", budgets=(1000, 10000), obs_list=(1,)):
    task = get_task(task_name)

    results = []

    for budget in budgets:
        for obs in obs_list:
            set_seeds(0)

            rej_samples, ra_samples = run_rej_abc_and_ra(
                task=task,
                obs_id=obs,
                budget=budget,
                num_samples_out=NUM_POST_SAMPLES,
                seed=0,
            )

            # NPE via sbibm wrapper
            npe_samples, _, _ = npe(
                task=task,
                num_observation=obs,
                num_simulations=budget,
                num_samples=NUM_POST_SAMPLES,
            )

            ref = task.get_reference_posterior_samples(num_observation=obs)

            rej_score = float(c2st(ref, rej_samples, seed=0))
            ra_score = float(c2st(ref, ra_samples, seed=0))
            npe_score = float(c2st(ref, npe_samples, seed=0))

            results.append(
                {
                    "task": task_name,
                    "budget": budget,
                    "obs": obs,
                    "rej_c2st": rej_score,
                    "ra_c2st": ra_score,
                    "npe_c2st": npe_score,
                }
            )

            print(
                f"[{task_name}] obs={obs} budget={budget} | "
                f"REJ={rej_score:.4f} RA={ra_score:.4f} NPE={npe_score:.4f}"
            )

    return results


if __name__ == "__main__":
    # Start with Gaussian Linear, obs=1, budgets 1k and 10k
    results = evaluate_methods(
        task_name="gaussian_linear",
        budgets=(1000, 10000),
        obs_list=(1,),
    )


