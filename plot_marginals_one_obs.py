import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sbibm.tasks import get_task
from sbibm.algorithms.sbi.mcabc import run as rej_abc_original
from sbibm.algorithms.sbi.smcabc import run as smc_abc_original

from compare_rej_ra_npe_gaussian_linear import run_rej_abc_and_ra
from run_smc_ra_gaussian_linear import run_smc_abc_ra_same_logic


METHODS = ["rej", "rej_ra", "smc", "smc_ra", "ref"]


def get_samples(task, method: str, obs_id: int, budget: int, num_post: int, seed: int) -> torch.Tensor:
    if method == "ref":
        return task.get_reference_posterior_samples(num_observation=obs_id)
    if method == "rej":
        samples, _, _ = rej_abc_original(
            task=task,
            num_observation=obs_id,
            num_simulations=budget,
            num_samples=num_post,
        )
        return samples
    if method == "smc":
        samples, _, _ = smc_abc_original(
            task=task,
            num_observation=obs_id,
            num_simulations=budget,
            num_samples=num_post,
        )
        return samples
    if method == "rej_ra":
        _, ra = run_rej_abc_and_ra(
            task=task,
            obs_id=obs_id,
            budget=budget,
            num_samples_out=num_post,
            seed=seed,
        )
        return ra
    if method == "smc_ra":
        ra = run_smc_abc_ra_same_logic(
            task=task,
            obs_id=obs_id,
            budget=budget,
            num_samples_out=num_post,
            seed=seed,
        )
        return ra
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--obs-id", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--num-post", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    task = get_task(args.task)
    if args.output_dir is None:
        output_dir = Path("results") / "marginals" / args.task / f"budget={args.budget}" / f"obs={args.obs_id}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_map = {}
    for method in METHODS:
        samples_map[method] = get_samples(task, method, args.obs_id, args.budget, args.num_post, args.seed)

    dim = samples_map[METHODS[0]].shape[1]
    for d in range(dim):
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in METHODS:
            x = samples_map[method].detach().cpu().numpy()
            ax.hist(x[:, d], bins=50, density=True, histtype="step", linewidth=1.5, label=method)
        ax.set_title(f"Marginal theta[{d}]")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"marginal_theta_{d}.png", dpi=150)
        plt.close(fig)

    print(f"Wrote marginal plots to {output_dir}")


if __name__ == "__main__":
    main()


