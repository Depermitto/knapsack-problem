import pandas as pd
import numpy as np
import time
import knapsack

from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Config:
    data_filepaths: list[str]
    num_trials: int
    save_path: str
    max_items: int
    population_size: int = 100
    num_generations: int = 100
    num_best: int = 10
    learning_rate: float = 0.1
    mutation_probability: float = 0.05
    mutation_std: float = 0.15
    threshold: float = 1e-4


def ecdf(data):
    """Calculate ECDF curve for data

    Args:
        data (array-like): input data

    Returns:
        tuple[array-like, array-like]: x, y data for calculated ECDF
    """
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y


def test_algorithms(config: Config):
    all_results = []
    for filepath in config.data_filepaths:
        optimal_solution, capacity, items = knapsack.read_data(filepath)
        num_items = len(items)
        if num_items > config.max_items:
            continue
        print(filepath, num_items, config.max_items)

        pbil_results = []
        a_star_results = []
        for _ in range(config.num_trials):
            start_time = time.time()
            solution, _, best_values, _ = knapsack.pbil(
                total_capacity=capacity,
                items=items,
                population_size=config.population_size,
                num_generations=config.num_generations,
                num_best=config.num_best,
                learning_rate=config.learning_rate,
                mutation_probability=config.mutation_probability,
                mutation_std=config.mutation_std,
                threshold=config.threshold,
            )
            elapsed = time.time() - start_time
            pbil_results.append(
                {
                    "num_items": num_items,
                    "capacity": capacity,
                    "generations_or_iterations": len(best_values),
                    "optimal_solution": optimal_solution,
                    "solution": solution,
                    "time": elapsed,
                }
            )

            start_time = time.time()
            solution, _, best_values = knapsack.a_star(
                items=items, total_capacity=capacity
            )
            elapsed = time.time() - start_time
            a_star_results.append(
                {
                    "num_items": num_items,
                    "capacity": capacity,
                    "generations_or_iterations": len(best_values),
                    "optimal_solution": optimal_solution,
                    "solution": solution,
                    "time": elapsed,
                }
            )
        a_star_results_df = pd.DataFrame(a_star_results)
        pbil_results_df = pd.DataFrame(pbil_results)
        all_results.append(pbil_results_df)
        all_results.append(a_star_results_df)

        plt.figure(figsize=(8, 5))

        x, y = ecdf(pbil_results_df["solution"])
        plt.step(x, y, where="post", color="b", label="PBIL")

        x, y = ecdf(a_star_results_df["solution"])
        plt.step(x, y, where="post", color="r", label="A*")

        plt.xlabel("Wartość")
        plt.ylabel("Odsetek wartości mniejszych")
        plt.title("Empiryczna Funkcja Dystrybucji (ECDF)")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"{config.save_path}ecdf-{num_items}-{capacity}.png",
            format="png",
        )

    final_results = pd.concat(all_results, ignore_index=True)
    summary = (
        final_results.groupby(["num_items", "capacity"])
        .agg(["max", "mean", "std"])
        .reset_index()
    )
    summary.to_csv(config.save_path + "results.csv")


def main():
    from pathlib import Path

    correlations = [
        # "uncorrelated",
        "small",
        # "medium_correlation",
        # "strong_correlation",
    ]
    for correlation in correlations:
        config = Config(
            data_filepaths=list(map(str, Path("data").glob(f"{correlation}/*.csv"))),
            save_path=f"output/{correlation}/",
            num_trials=100,
            max_items=1000,
        )
        test_algorithms(config)


if __name__ == "__main__":
    main()
