import argparse
import json
import os
import time
import hashlib
import random
import numpy as np
import torch

from sbibm.tasks import get_task
from sbibm.metrics.c2st import c2st

# sbibm wrappers
from sbibm.algorithms import snpe as npe
from sbibm.algorithms.sbi.mcabc import run as rej_abc_original
from sbibm.algorithms.sbi.smcabc import run as smc_abc_original

# RA implementations
from compare_rej_ra_npe_gaussian_linear import run_rej_abc_and_ra
from run_smc_ra_gaussian_linear import run_smc_abc_ra_same_logic


METHODS = {"npe", "rej", "rej_ra", "smc", "smc_ra"}


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


def run_method(task, method: str, obs_id: int, budget: int, num_post: int, seed: int):
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

    if method == "rej_ra":
        _, ra = run_rej_abc_and_ra(
            task=task,
            obs_id=obs_id,
            budget=budget,
            num_samples_out=num_post,
            seed=seed,
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

    if method == "smc_ra":
        samples = run_smc_abc_ra_same_logic(
            task=task,
            obs_id=obs_id,
            budget=budget,
            num_samples_out=num_post,
            seed=seed,
        )
        return samples

    raise ValueError(f"Unknown method: {method}")


def main():
    # Allow safe deserialization of MixtureSameFamily used in reference posteriors.
    try:
        import pyro.distributions.torch  # noqa: F401
        from torch.serialization import add_safe_globals
        from torch.distributions import MixtureSameFamily

        add_safe_globals([MixtureSameFamily])
    except Exception:
        pass

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
    else:
        task_name = args.task
        method = args.method
        budget = args.budget
        obs_id = args.obs_id
        base_seed = args.base_seed
        seed = args.seed if args.seed is not None else derive_seed(task_name, method, budget, obs_id, base_seed)
        output_path = args.output
        output_root = args.output_root

    if output_path is None:
        if output_root is None:
            output_root = default_output_root()
        output_path = default_output_path(task_name, method, budget, obs_id, seed, output_root)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Skip if already done
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("status") == "ok":
                print(f"Skipping existing result: {output_path}")
                return
        except Exception:
            pass

    result = {
        "task": task_name,
        "method": method,
        "budget": budget,
        "obs_id": obs_id,
        "seed": seed,
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
        samples = run_method(task, method, obs_id, budget, args.num_post, seed)
        runtime = time.perf_counter() - t0

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

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

