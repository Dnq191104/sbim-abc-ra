import numpy as np
import torch
from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st

from ra_core import (
    linear_reg_adjust,
    resample_kde_with_bounds,
)


TASK_NAME = "gaussian_linear"
OBS_ID = 1
BUDGET = 1000
NUM_POST_SAMPLES = 10_000
NUM_TOP = 100
BATCH_SIZE = 2048
NUM_ROUNDS = 3


def run_smc_abc_ra_same_logic(task, obs_id: int, budget: int, num_samples_out: int, seed: int):
    """
    SMC-ABC + RA using the same RA logic as in compare_rej_ra_npe_gaussian_linear.py.
    - Split budget into rounds
    - For each round, sample from prior (simple SMC baseline)
    - Accept top-k by L2 distance
    - Pool accepted samples across rounds
    - Apply the same linear_reg_adjust (with distance weights + standardization)
    - KDE resample
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    prior = task.get_prior()
    simulator = task.get_simulator()
    x_obs = task.get_observation(num_observation=obs_id)

    sims_per_round = budget // NUM_ROUNDS

    theta_acc = None
    x_acc = None
    d_acc = None

    for r in range(NUM_ROUNDS):
        theta = prior(num_samples=sims_per_round)
        x = simulator(theta)

        x_flat = x.reshape(sims_per_round, -1)
        x_obs_flat = x_obs.reshape(-1)

        d = torch.norm(x_flat - x_obs_flat[None, :], dim=1)
        k = min(int(NUM_TOP), int(sims_per_round))
        idx = torch.topk(d, k=k, largest=False).indices

        # Keep only the final round for RA
        theta_acc = theta[idx]
        x_acc = x_flat[idx]
        d_acc = d[idx]
    x_obs_np = x_obs.reshape(-1).detach().cpu().numpy()

    # Apply the SAME RA logic (distance weights + standardization)
    theta_adj = linear_reg_adjust(
        theta_acc=theta_acc.detach().cpu().numpy(),
        x_acc=x_acc.detach().cpu().numpy(),
        x_obs=x_obs_np,
        d_acc=d_acc.detach().cpu().numpy(),
        smc_weights=np.ones(len(theta_acc)),
        use_distance_weights=True,
        standardize_x=True,
    )

    prior_dist = task.get_prior_dist()
    ra_np, _ = resample_kde_with_bounds(prior_dist, theta_adj, num_samples_out, seed=seed)
    ra = torch.as_tensor(ra_np, dtype=torch.float32)
    return ra


def main():
    task = get_task(TASK_NAME)
    ref = task.get_reference_posterior_samples(num_observation=OBS_ID)

    ra_samples = run_smc_abc_ra_same_logic(
        task=task,
        obs_id=OBS_ID,
        budget=BUDGET,
        num_samples_out=NUM_POST_SAMPLES,
        seed=0,
    )

    score = float(c2st(ref, ra_samples, seed=0))
    print(f"Task={TASK_NAME} obs={OBS_ID} budget={BUDGET} | SMC-ABC+RA C2ST={score:.4f}")


if __name__ == "__main__":
    main()

