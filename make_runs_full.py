# make_runs_full.py
from pathlib import Path

TASKS = [
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "two_moons",
    "slcp",
    "slcp_distractors",
]
EXCLUDE = {"sir", "lotka_volterra"}
TASKS = [t for t in TASKS if t not in EXCLUDE]

BUDGETS = [1_000, 10_000, 100_000]
OBS = list(range(1, 11))  # your logs show obs=5,6 etc => 1-based
CPU_METHODS = ["rej", "rej_ra", "smc", "smc_ra"]
GPU_METHODS = ["npe"]

def stable_seed(task: str, method: str, budget: int, obs: int) -> int:
    # deterministic seed; avoids accidental collisions
    return (abs(hash((task, method, budget, obs))) % 2_000_000_000) + 1

def write_tsv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("task\tmethod\tbudget\tobs\tseed\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"Wrote {path} with {len(rows)} runs")

def main():
    repo = Path(__file__).resolve().parent
    cpu_rows = []
    gpu_rows = []

    for task in TASKS:
        for budget in BUDGETS:
            for obs in OBS:
                for method in CPU_METHODS:
                    cpu_rows.append((task, method, budget, obs, stable_seed(task, method, budget, obs)))
                for method in GPU_METHODS:
                    gpu_rows.append((task, method, budget, obs, stable_seed(task, method, budget, obs)))

    write_tsv(repo / "runs" / "cpu_runs_full.tsv", cpu_rows)
    write_tsv(repo / "runs" / "gpu_runs_full.tsv", gpu_rows)

if __name__ == "__main__":
    main()
