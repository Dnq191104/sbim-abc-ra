import numpy as np
import torch
from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st
from abc_ra import run_abc_ra



def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

# Using abc_ra.run_abc_ra instead of custom REJ-ABC

# ---- Run across obs 1..10 and compute C2ST ----
task = get_task("two_moons")
num_sims = 10_000
num_samples = 10_000

print("Running ABC-RA with Linear Regression Adjustment")
print("This includes: feature standardization + LRA + kernel weights")
print("=" * 60)

scores = []
for obs in range(1, 11):
    set_seeds(0)  # For reproducibility

    # Use our ABC-RA implementation with Linear RA
    algo_samples = run_abc_ra(
        task=task,
        num_observation=obs,
        num_simulations=num_sims,
        num_posterior_samples=num_samples,
        accept_frac=100/num_sims,
        min_accept=100,
        use_weights=True,
        batch_size=2048,
        # add: resample_method="kde", kde_bandwidth="pos_cv"
    )

    ref = task.get_reference_posterior_samples(num_observation=obs)
    score = float(c2st(ref, algo_samples, seed=0))
    scores.append(score)
    print("obs", obs, "C2ST", score, "sim_calls", num_sims)

print("=" * 60)
print(".4f")
print("\nABC-RA includes:")
print("- Feature extraction with standardization")
print("- Rejection ABC acceptance")
print("- Linear Regression Adjustment (LRA)")
print("- Epanechnikov kernel weighting")
print("- Posterior resampling")
