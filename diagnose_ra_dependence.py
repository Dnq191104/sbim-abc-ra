import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sbibm.tasks import get_task
from sbibm.algorithms.sbi.mcabc import run as rej_abc_original
from sbibm.algorithms.sbi.smcabc import run as smc_abc_original

from compare_rej_ra_npe_gaussian_linear import run_rej_abc_and_ra
from run_smc_ra_gaussian_linear import run_smc_abc_ra_same_logic
from ra_core import resample_with_replacement


METHODS = ["rej", "rej_ra", "smc", "smc_ra", "ref"]


def corr_mat(samples: torch.Tensor) -> np.ndarray:
    x = samples.detach().cpu().numpy()
    if x.ndim != 2 or x.shape[0] < 2:
        return np.eye(x.shape[1] if x.ndim == 2 else 1)
    c = np.corrcoef(x, rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return c


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
        ra_np = resample_with_replacement(ra.detach().cpu().numpy(), num_post, seed=seed)
        return torch.as_tensor(ra_np, dtype=ra.dtype)
    if method == "smc_ra":
        ra = run_smc_abc_ra_same_logic(
            task=task,
            obs_id=obs_id,
            budget=budget,
            num_samples_out=num_post,
            seed=seed,
        )
        ra_np = resample_with_replacement(ra.detach().cpu().numpy(), num_post, seed=seed)
        return torch.as_tensor(ra_np, dtype=ra.dtype)
    raise ValueError(f"Unknown method: {method}")


def plot_corrs(corrs: dict[str, np.ndarray], output_dir: Path) -> None:
    methods = list(corrs.keys())
    n = len(methods)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for i, method in enumerate(methods):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        im = ax.imshow(corrs[method], vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmaps.png", dpi=150)
    plt.close(fig)


def plot_pairs(samples_map: dict[str, torch.Tensor], pairs: list[tuple[int, int]], output_dir: Path) -> None:
    methods = list(samples_map.keys())
    nrows = len(methods)
    ncols = len(pairs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, method in enumerate(methods):
        x = samples_map[method].detach().cpu().numpy()
        for j, (a, b) in enumerate(pairs):
            ax = axes[i][j]
            ax.scatter(x[:, a], x[:, b], s=4, alpha=0.3)
            if i == 0:
                ax.set_title(f"theta[{a}] vs theta[{b}]")
            if j == 0:
                ax.set_ylabel(method)
    fig.tight_layout()
    fig.savefig(output_dir / "pair_scatter.png", dpi=150)
    plt.close(fig)


def plot_marginals(samples_map: dict[str, torch.Tensor], output_dir: Path) -> None:
    methods = list(samples_map.keys())
    dim = samples_map[methods[0]].shape[1]
    for d in range(dim):
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in methods:
            x = samples_map[method].detach().cpu().numpy()
            ax.hist(x[:, d], bins=50, density=True, alpha=0.4, label=method)
        ax.set_title(f"Marginal theta[{d}]")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"marginal_theta_{d}.png", dpi=150)
        plt.close(fig)


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
        output_dir = Path("results") / "diagnostics" / args.task / f"budget={args.budget}" / f"obs={args.obs_id}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_map = {}
    for method in METHODS:
        samples = get_samples(task, method, args.obs_id, args.budget, args.num_post, args.seed)
        samples_map[method] = samples

    corrs = {m: corr_mat(samples_map[m]) for m in METHODS}
    ref_corr = corrs.get("ref")
    if ref_corr is not None:
        for m in METHODS:
            if m == "ref":
                continue
            diff = np.linalg.norm(corrs[m] - ref_corr, ord="fro")
            print(f"{m}: corr Frobenius vs ref = {diff:.4f}")

    plot_corrs(corrs, output_dir)

    dim = samples_map[METHODS[0]].shape[1]
    if dim >= 2:
        pairs = [(0, 1)]
        if dim >= 3:
            pairs += [(0, 2), (1, 2)]
        plot_pairs(samples_map, pairs, output_dir)

    plot_marginals(samples_map, output_dir)
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()

