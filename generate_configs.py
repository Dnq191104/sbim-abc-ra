import json
from sbibm.tasks import get_available_tasks


METHODS = ["npe", "rej", "rej_ra", "smc", "smc_ra"]
BUDGETS = [1_000, 10_000, 100_000]
OBS_IDS = list(range(1, 11))
BASE_SEED = 0


def main():
    tasks = get_available_tasks()
    configs = []

    for task in tasks:
        for method in METHODS:
            for budget in BUDGETS:
                for obs_id in OBS_IDS:
                    configs.append(
                        {
                            "task": task,
                            "method": method,
                            "budget": budget,
                            "obs_id": obs_id,
                            "base_seed": BASE_SEED,
                        }
                    )

    with open("configs.jsonl", "w", encoding="utf-8") as f:
        for cfg in configs:
            f.write(json.dumps(cfg) + "\n")

    print(f"Wrote {len(configs)} configs to configs.jsonl")


if __name__ == "__main__":
    main()

