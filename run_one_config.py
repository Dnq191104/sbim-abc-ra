# --- PyTorch 2.6 compat: sbibm/slcp_distractors Pyro checkpoints ---
import inspect
import torch

_sig = None
try:
    _sig = inspect.signature(torch.load)
except Exception:
    _sig = None

_HAS_WEIGHTS_ONLY = _sig is not None and ("weights_only" in _sig.parameters)

if _HAS_WEIGHTS_ONLY:
    _torch_load_orig = torch.load

    def torch_load_compat(*args, **kwargs):
        # restore pre-2.6 default behavior
        kwargs.setdefault("weights_only", False)
        return _torch_load_orig(*args, **kwargs)

    torch.load = torch_load_compat

# Allowlist Pyro MixtureSameFamily for safe unpickling when weights_only=True is used.
try:
    from pyro.distributions.torch import MixtureSameFamily
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([MixtureSameFamily])
except Exception:
    pass
# --------------------------------------------------------------------------

import argparse
import json
import os
import time
import hashlib
import random
from pathlib import Path
import numpy as np

from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st
from metrics_marginal import mean_mmd_over_dims

# sbibm wrappers
from sbibm.algorithms import snpe as npe
from sbibm.algorithms.sbi.mcabc import run as rej_abc_original
from sbibm.algorithms.sbi.smcabc import run as smc_abc_original

# RA implementations
from compare_rej_ra_npe_gaussian_linear import run_rej_abc_and_ra
from run_smc_ra_gaussian_linear import run_smc_abc_ra_same_logic


VARIANTS = {
    "rej_ra_base": dict(reg="ols", topk=100, pca_cap=False),
    "rej_ra_ridge": dict(reg="ridge", alpha=1.0, topk=100, pca_cap=False),
    "rej_ra_topk500": dict(reg="ols", topk=500, pca_cap=False),
    "rej_ra_ridge_topk500": dict(reg="ridge", alpha=1.0, topk=500, pca_cap=False),
    "rej_ra_ridge_topk500_pcapcap": dict(reg="ridge", alpha=1.0, topk=500, pca_cap=True),
    "smc_ra_base": dict(reg="ols", topk=100, pca_cap=False),
    "smc_ra_ridge": dict(reg="ridge", alpha=1.0, topk=100, pca_cap=False),
    "smc_ra_topk500": dict(reg="ols", topk=500, pca_cap=False),
    "smc_ra_ridge_topk500": dict(reg="ridge", alpha=1.0, topk=500, pca_cap=False),
    "smc_ra_ridge_topk500_pcapcap": dict(reg="ridge", alpha=1.0, topk=500, pca_cap=True),
}


METHODS = {"npe", "rej", "rej_ra", "smc", "smc_ra"} | set(VARIANTS.keys())


def derive_seed(task: str, method: str, budget: int, obs_id: int, base_seed: int) -> int:
    key = f"{task}|{method}|{budget}|{obs_id}|{base_seed}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    return int(h[:8], 16)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_output_root() -> str:
    scratch_root = "/scratch/dangn5_sbibm_bench/results"
    if os.path.exists(scratch_root):
        return scratch_root
    return "results"


def default_output_path(
    task: str, method: str, budget: int, obs_id: int, seed: int, output_root: str
) -> str:
    return os.path.join(
        output_root,
        task,
        method,
        f"budget={budget}",
        f"obs={obs_id}",
        f"seed={seed}.json",
    )


def resolve_ra_settings(
    method: str,
    ra_topk: int | None,
    ra_reg: str | None,
    ra_alpha: float | None,
    ra_pca_dim: int | None,
    ra_pca_cap: bool | None,
    ra_use_weights: bool | None,
) -> dict:
    settings = {
        "topk": 100,
        "reg": "ols",
        "alpha": 1.0,
        "pca_dim": 50,
        "pca_cap": False,
        "use_weights": True,
    }
    if method in VARIANTS:
        settings.update(VARIANTS[method])

    if ra_topk is not None:
        settings["topk"] = ra_topk
    if ra_reg is not None:
        settings["reg"] = ra_reg
    if ra_alpha is not None:
        settings["alpha"] = ra_alpha
    if ra_pca_dim is not None:
        settings["pca_dim"] = ra_pca_dim
    if ra_pca_cap is not None:
        settings["pca_cap"] = ra_pca_cap
    if ra_use_weights is not None:
        settings["use_weights"] = ra_use_weights
    return settings


def run_method(
    task,
    method: str,
    obs_id: int,
    budget: int,
    num_post: int,
    seed: int,
    ra_settings: dict | None,
):
    if method == "npe":
        samples, _, _ = npe(
            task=task,
            num_observation=obs_id,
            num_simulations=budget,
            num_samples=num_post,
        )
        return samples

    if method == "rej":
        samples, _, _ = rej_abc_original(
            task=task,
            num_observation=obs_id,
            num_simulations=budget,
            num_samples=num_post,
        )
        return samples

    if method == "rej_ra" or method.startswith("rej_ra_"):
        settings = ra_settings or {}
        _, ra = run_rej_abc_and_ra(
            task=task,
            obs_id=obs_id,
            budget=budget,
            num_samples_out=num_post,
            seed=seed,
            topk=settings.get("topk", 100),
            reg_type=settings.get("reg", "ols"),
            alpha=settings.get("alpha", 1.0),
            pca_dim=settings.get("pca_dim", 50),
            pca_cap=settings.get("pca_cap", False),
            use_weights=settings.get("use_weights", True),
        )
        return ra

    if method == "smc":
        samples, _, _ = smc_abc_original(
            task=task,
            num_observation=obs_id,
            num_simulations=budget,
            num_samples=num_post,
        )
        return samples

    if method == "smc_ra" or method.startswith("smc_ra_"):
        settings = ra_settings or {}
        samples = run_smc_abc_ra_same_logic(
            task=task,
            obs_id=obs_id,
            budget=budget,
            num_samples_out=num_post,
            seed=seed,
            topk=settings.get("topk", 100),
            reg_type=settings.get("reg", "ols"),
            alpha=settings.get("alpha", 1.0),
            pca_dim=settings.get("pca_dim", 50),
            pca_cap=settings.get("pca_cap", False),
            use_weights=settings.get("use_weights", True),
        )
        return samples

    raise ValueError(f"Unknown method: {method}")


def should_skip(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        prev = json.loads(output_path.read_text(encoding="utf-8"))
        return prev.get("status") == "ok"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--obs-id", required=True, type=int)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-post", type=int, default=10_000)
    parser.add_argument("--ra-topk", type=int, default=None)
    parser.add_argument("--ra-reg", type=str, choices=["ols", "ridge"], default=None)
    parser.add_argument("--ra-alpha", type=float, default=None)
    parser.add_argument("--ra-pca-dim", type=int, default=None)
    pca_group = parser.add_mutually_exclusive_group()
    pca_group.add_argument("--ra-pca-cap", action="store_true", default=None)
    pca_group.add_argument("--ra-no-pca-cap", action="store_false", dest="ra_pca_cap", default=None)
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument("--ra-use-weights", action="store_true", default=None)
    weight_group.add_argument("--ra-no-weights", action="store_false", dest="ra_use_weights", default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    if args.config is not None and args.index is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            line = f.readlines()[args.index].strip()
        cfg = json.loads(line)
        task_name = cfg["task"]
        method = cfg["method"]
        budget = int(cfg["budget"])
        obs_id = int(cfg["obs_id"])
        base_seed = int(cfg.get("base_seed", args.base_seed))
        seed = int(cfg.get("seed", 0))
        if seed == 0:
            seed = derive_seed(task_name, method, budget, obs_id, base_seed)
        output_path = cfg.get("output_path")
        output_root = cfg.get("output_root", args.output_root)
        ra_topk = cfg.get("ra_topk", args.ra_topk)
        ra_reg = cfg.get("ra_reg", args.ra_reg)
        ra_alpha = cfg.get("ra_alpha", args.ra_alpha)
        ra_pca_dim = cfg.get("ra_pca_dim", args.ra_pca_dim)
        ra_pca_cap = cfg.get("ra_pca_cap", args.ra_pca_cap)
        ra_use_weights = cfg.get("ra_use_weights", args.ra_use_weights)
    else:
        task_name = args.task
        method = args.method
        budget = args.budget
        obs_id = args.obs_id
        base_seed = args.base_seed
        seed = args.seed if args.seed is not None else derive_seed(task_name, method, budget, obs_id, base_seed)
        output_path = args.output
        output_root = args.output_root
        ra_topk = args.ra_topk
        ra_reg = args.ra_reg
        ra_alpha = args.ra_alpha
        ra_pca_dim = args.ra_pca_dim
        ra_pca_cap = args.ra_pca_cap
        ra_use_weights = args.ra_use_weights

    if output_path is None:
        if output_root is None:
            output_root = default_output_root()
        output_path = default_output_path(task_name, method, budget, obs_id, seed, output_root)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if should_skip(output_path):
        print(f"Skipping existing result: {output_path}")
        return
    if output_path.exists():
        print(f"Re-running because existing result is not ok: {output_path}")

    ra_settings = resolve_ra_settings(
        method=method,
        ra_topk=ra_topk,
        ra_reg=ra_reg,
        ra_alpha=ra_alpha,
        ra_pca_dim=ra_pca_dim,
        ra_pca_cap=ra_pca_cap,
        ra_use_weights=ra_use_weights,
    )

    result = {
        "task": task_name,
        "method": method,
        "budget": budget,
        "obs_id": obs_id,
        "seed": seed,
        "ra_settings": ra_settings if method.startswith("rej_ra") or method.startswith("smc_ra") else None,
        "num_post_requested": None,
        "num_post_returned": None,
        "theta_dim": None,
        "metric": None,
        "mmd_mean": None,
        "mmd_median": None,
        "mmd_per_dim": None,
        "num_method": None,
        "num_ref_used": None,
        "c2st": None,
        "runtime_sec": None,
        "status": "error",
        "error": None,
    }

    try:
        set_seeds(seed)
        task = get_task(task_name)
        ref = task.get_reference_posterior_samples(num_observation=obs_id)

        t0 = time.perf_counter()
        samples = run_method(task, method, obs_id, budget, args.num_post, seed, ra_settings)
        runtime = time.perf_counter() - t0

        result["num_post_requested"] = int(args.num_post)
        result["num_post_returned"] = int(samples.shape[0])
        result["theta_dim"] = int(samples.shape[1])

        output_dir = output_path.parent
        np.save(output_dir / "theta.npy", samples.detach().cpu().numpy().astype(np.float32))
        np.save(output_dir / "theta_ref.npy", ref.detach().cpu().numpy().astype(np.float32))
        theta_np = samples.detach().cpu().numpy()
        ref_np = ref.detach().cpu().numpy()
        n = theta_np.shape[0]
        rng = np.random.default_rng(seed)
        if ref_np.shape[0] > n:
            idx = rng.choice(ref_np.shape[0], size=n, replace=False)
            ref_cmp = ref_np[idx]
        else:
            ref_cmp = ref_np

        mmd_stats = mean_mmd_over_dims(theta_np, ref_cmp)
        result.update(mmd_stats)
        result["metric"] = "mmd_1d_rbf"
        result["num_method"] = int(n)
        result["num_ref_used"] = int(ref_cmp.shape[0])

        score = float(c2st(ref, samples, seed=0))
        result.update(
            {
                "c2st": score,
                "runtime_sec": runtime,
                "status": "ok",
                "error": None,
            }
        )
    except Exception as e:
        result["error"] = str(e)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

