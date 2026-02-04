import numpy as np
import torch
from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st

# Original sbibm algorithms
from sbibm.algorithms.sbi.mcabc import run as rej_abc_original
from sbibm.algorithms.sbi.smcabc import run as smc_abc_original

# Our RA implementations
from compare_rej_ra_npe_gaussian_linear import run_rej_abc_and_ra
from smc_abc_ra import run_smc_abc_ra


TASK_NAME = "gaussian_linear"
OBS_ID = 1
BUDGET = 1000
NUM_POST_SAMPLES = 10_000


def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_original_methods(task):
    """Run original sbibm REJ-ABC and SMC-ABC (no RA)."""
    try:
        rej_samples, _, _ = rej_abc_original(
            task=task,
            num_observation=OBS_ID,
            num_simulations=BUDGET,
            num_samples=NUM_POST_SAMPLES,
        )
    except Exception as e:
        print(f"[ERROR] Original REJ-ABC failed: {e}")
        rej_samples = None

    try:
        smc_samples, _, _ = smc_abc_original(
            task=task,
            num_observation=OBS_ID,
            num_simulations=BUDGET,
            num_samples=NUM_POST_SAMPLES,
        )
    except Exception as e:
        print(f"[ERROR] Original SMC-ABC failed: {e}")
        smc_samples = None

    return rej_samples, smc_samples


def run_ra_methods(task):
    """Run our REJ-ABC+RA and SMC-ABC+RA implementations."""
    rej_samples, ra_samples = run_rej_abc_and_ra(
        task=task,
        obs_id=OBS_ID,
        budget=BUDGET,
        num_samples_out=NUM_POST_SAMPLES,
        seed=0,
    )

    smc_ra_samples = run_smc_abc_ra(
        task=task,
        num_observation=OBS_ID,
        num_simulations=BUDGET,
        num_posterior_samples=NUM_POST_SAMPLES,
        num_rounds=3,
    )

    return rej_samples, ra_samples, smc_ra_samples


def main():
    set_seeds(0)
    task = get_task(TASK_NAME)
    ref = task.get_reference_posterior_samples(num_observation=OBS_ID)

    # Original methods
    rej_orig, smc_orig = run_original_methods(task)

    # RA methods
    rej_ra, ra_samples, smc_ra = run_ra_methods(task)

    print(f"\nTask: {TASK_NAME} | obs={OBS_ID} | budget={BUDGET}")
    print("=" * 60)

    if rej_orig is not None:
        rej_score = float(c2st(ref, rej_orig, seed=0))
        print(f"REJ-ABC (original)     C2ST: {rej_score:.4f}")
    if smc_orig is not None:
        smc_score = float(c2st(ref, smc_orig, seed=0))
        print(f"SMC-ABC (original)     C2ST: {smc_score:.4f}")

    ra_score = float(c2st(ref, ra_samples, seed=0))
    smc_ra_score = float(c2st(ref, smc_ra, seed=0))
    print(f"REJ-ABC + RA (ours)    C2ST: {ra_score:.4f}")
    print(f"SMC-ABC + RA (ours)    C2ST: {smc_ra_score:.4f}")


if __name__ == "__main__":
    main()

