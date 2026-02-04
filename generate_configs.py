from sbibm.tasks import get_available_tasks


CPU_METHODS = ["rej", "rej_ra", "smc", "smc_ra"]
GPU_METHODS = ["npe"]
BUDGETS = [1_000, 10_000, 100_000]
OBS_IDS = list(range(1, 11))
BASE_SEED = 0
NUM_POST = 10_000


def write_manifest(path: str, tasks: list[str], methods: list[str]) -> int:
    rows = []
    for task in tasks:
        for method in methods:
            for budget in BUDGETS:
                for obs_id in OBS_IDS:
                    rows.append(
                        {
                            "task": task,
                            "method": method,
                            "budget": budget,
                            "obs_id": obs_id,
                            "base_seed": BASE_SEED,
                            "num_post": NUM_POST,
                        }
                    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("task\tmethod\tbudget\tobs_id\tbase_seed\tnum_post\n")
        for row in rows:
            f.write(
                f"{row['task']}\t{row['method']}\t{row['budget']}\t"
                f"{row['obs_id']}\t{row['base_seed']}\t{row['num_post']}\n"
            )

    return len(rows)


def main():
    tasks = get_available_tasks()
    cpu_count = write_manifest("cpu_runs.tsv", tasks, CPU_METHODS)
    gpu_count = write_manifest("gpu_runs.tsv", tasks, GPU_METHODS)

    print(f"Wrote {cpu_count} rows to cpu_runs.tsv")
    print(f"Wrote {gpu_count} rows to gpu_runs.tsv")


if __name__ == "__main__":
    main()

