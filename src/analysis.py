import pandas as pd
import time
from dataclasses import dataclass
import knapsack


@dataclass
class Config:
    data_filepaths: list[str]
    num_trials: int
    save_path: str


def test_algorithms(data_filepath, num_trials):
    results = []

    for _ in range(num_trials):
        optimal_solution, capacity, items = knapsack.read_data(data_filepath)
        num_items = len(items)

        start_time = time.time()
        pbil_solution, _, _, _ = knapsack.pbil(items=items, total_capacity=capacity)
        pbil_time = time.time() - start_time

        start_time = time.time()
        astar_solution, _, _ = knapsack.a_star(items=items, total_capacity=capacity)
        astar_time = time.time() - start_time

        results.append(
            {
                "num_items": num_items,
                "capacity": capacity,
                "correlation_level": data_filepath.split("/")[1],
                "optimal_solution": optimal_solution,
                "pbil_solution": pbil_solution,
                "astar_solution": astar_solution,
                "pbil_time": pbil_time,
                "astar_time": astar_time,
            }
        )

    return pd.DataFrame(results)


def main(config: Config):
    all_results = []
    for filepath in config.data_filepaths:
        results = test_algorithms(filepath, config.num_trials)
        all_results.append(results)

    final_results = pd.concat(all_results, ignore_index=True)
    print(final_results)

    summary = (
        final_results.groupby(["num_items", "capacity", "correlation_level"])
        .agg(["max", "mean", "std"])
        .reset_index()
    )
    summary.to_csv(config.save_path)
    print(summary)


if __name__ == "__main__":
    from pathlib import Path

    correlation = "uncorrelated"
    config = Config(
        data_filepaths=list(map(str, Path("data").glob(f"{correlation}/*.csv"))),
        save_path=f"output/{correlation}.csv",
        num_trials=10,
    )
    main(config)
