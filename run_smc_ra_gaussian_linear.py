import numpy as np
import torch
from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st

from ra_core import (
    distance_weights,
    linear_reg_adjust,
)


TASK_NAME = "gaussian_linear"
OBS_ID = 1
BUDGET = 1000
NUM_POST_SAMPLES = 10_000
BUDGET_TOPK_RULES = [
    (100_000, 2000),
    (10_000, 1000),
    (0, 500),
]
BATCH_SIZE = 2048
NUM_ROUNDS = 3
USE_PCA = True
PCA_DIM = 50


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
    num_top = next(k for b, k in BUDGET_TOPK_RULES if budget >= b)

    theta_acc = None
    x_acc = None
    d_acc = None

    for r in range(NUM_ROUNDS):
        theta = prior(num_samples=sims_per_round)
        x = simulator(theta)

        x_flat = x.reshape(sims_per_round, -1)
        x_obs_flat = x_obs.reshape(-1)

        d = torch.norm(x_flat - x_obs_flat[None, :], dim=1)
        k = min(int(num_top), int(sims_per_round))
        idx = torch.topk(d, k=k, largest=False).indices

        # Keep only the final round for RA
        theta_acc = theta[idx]
        x_acc = x_flat[idx]
        d_acc = d[idx]
    x_obs_np = x_obs.reshape(-1).detach().cpu().numpy()

    # Apply the SAME RA logic (distance weights + standardization)
    dw = distance_weights(d_acc.detach().cpu().numpy())
    if dw is None:
        smc_weights = np.ones(len(theta_acc))
    else:
        smc_weights = dw / np.maximum(dw.sum(), 1e-12)

    theta_adj = linear_reg_adjust(
        theta_acc=theta_acc.detach().cpu().numpy(),
        x_acc=x_acc.detach().cpu().numpy(),
        x_obs=x_obs_np,
        d_acc=d_acc.detach().cpu().numpy(),
        smc_weights=smc_weights,
        use_distance_weights=False,
        standardize_x=True,
        pca_dim=PCA_DIM if USE_PCA else None,
    )
    ra = torch.as_tensor(theta_adj, dtype=torch.float32)
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

