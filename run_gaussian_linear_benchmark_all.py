import numpy as np
import torch

from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st

# NPE (sbibm wrapper)
from sbibm.algorithms import snpe as npe

# Original sbibm algorithms
from sbibm.algorithms.sbi.mcabc import run as rej_abc_original
from sbibm.algorithms.sbi.smcabc import run as smc_abc_original

# Our RA implementations
from compare_rej_ra_npe_gaussian_linear import run_rej_abc_and_ra
from run_smc_ra_gaussian_linear import run_smc_abc_ra_same_logic


TASK_NAME = "gaussian_linear"
OBS_ID = 1
BUDGET = 1000
NUM_POST_SAMPLES = 10_000


def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seeds(0)
    task = get_task(TASK_NAME)
    ref = task.get_reference_posterior_samples(num_observation=OBS_ID)

    print(f"Task={TASK_NAME} obs={OBS_ID} budget={BUDGET}")
    print("=" * 60)

    # NPE (sbibm wrapper)
    npe_samples, _, _ = npe(
        task=task,
        num_observation=OBS_ID,
        num_simulations=BUDGET,
        num_samples=NUM_POST_SAMPLES,
    )
    npe_score = float(c2st(ref, npe_samples, seed=0))
    print(f"NPE                 C2ST: {npe_score:.4f}")

    # REJ-ABC (original sbibm)
    try:
        rej_samples, _, _ = rej_abc_original(
            task=task,
            num_observation=OBS_ID,
            num_simulations=BUDGET,
            num_samples=NUM_POST_SAMPLES,
        )
        rej_score = float(c2st(ref, rej_samples, seed=0))
        print(f"REJ-ABC (original)  C2ST: {rej_score:.4f}")
    except Exception as e:
        print(f"REJ-ABC (original)  failed: {e}")

    # REJ-ABC + RA (same logic as compare_rej_ra_npe_gaussian_linear.py)
    rej_ra, ra_samples = run_rej_abc_and_ra(
        task=task,
        obs_id=OBS_ID,
        budget=BUDGET,
        num_samples_out=NUM_POST_SAMPLES,
        seed=0,
    )
    ra_score = float(c2st(ref, ra_samples, seed=0))
    print(f"REJ-ABC + RA        C2ST: {ra_score:.4f}")

    # SMC-ABC (original sbibm)
    try:
        smc_samples, _, _ = smc_abc_original(
            task=task,
            num_observation=OBS_ID,
            num_simulations=BUDGET,
            num_samples=NUM_POST_SAMPLES,
        )
        smc_score = float(c2st(ref, smc_samples, seed=0))
        print(f"SMC-ABC (original)  C2ST: {smc_score:.4f}")
    except Exception as e:
        print(f"SMC-ABC (original)  failed: {e}")

    # SMC-ABC + RA (same logic as run_smc_ra_gaussian_linear.py)
    smc_ra_samples = run_smc_abc_ra_same_logic(
        task=task,
        obs_id=OBS_ID,
        budget=BUDGET,
        num_samples_out=NUM_POST_SAMPLES,
        seed=0,
    )
    smc_ra_score = float(c2st(ref, smc_ra_samples, seed=0))
    print(f"SMC-ABC + RA        C2ST: {smc_ra_score:.4f}")


if __name__ == "__main__":
    main()

