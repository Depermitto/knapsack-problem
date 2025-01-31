import pandas as pd
import numpy as np
import time
import knapsack

from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import interp1d


def __numfmt(x):
    if x == 0:
        return "0"
    elif int(x) == x:
        return str(int(x))
    elif abs(x) < 1e-3:
        return f"{x:.2e}"
    else:
        return f"{x:.2f}"


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
    mutation_probability: float = 0.1
    mutation_std: float = 0.1
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
    all_pbil_results = []
    all_a_star_results = []
    for filepath in config.data_filepaths:
        optimal_solution, capacity, items = knapsack.read_data(filepath)
        num_items = len(items)
        if num_items > config.max_items:
            continue
        print(filepath, num_items, config.max_items)

        pbil_results = []
        pbil_ecdf_results = []
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
                    "generations": len(best_values),
                    "optimal_solution": optimal_solution,
                    "solution": solution,
                    "time": elapsed,
                }
            )
            pbil_ecdf_results.append(ecdf([np.max(topN) for topN in best_values]))

            start_time = time.time()
            solution, _, best_values = knapsack.a_star(
                items=items, total_capacity=capacity
            )
            elapsed = time.time() - start_time
            a_star_results.append(
                {
                    "num_items": num_items,
                    "capacity": capacity,
                    "iterations": len(best_values),
                    "optimal_solution": optimal_solution,
                    "solution": solution,
                    "time": elapsed,
                }
            )
        pbil_results_df = pd.DataFrame(pbil_results)
        all_pbil_results.append(pbil_results_df)

        a_star_results_df = pd.DataFrame(a_star_results)
        all_a_star_results.append(a_star_results_df)

        plt.figure(figsize=(8, 5))

        all_x_vals = [r[0] for r in pbil_ecdf_results]
        all_y_vals = [r[1] for r in pbil_ecdf_results]
        
        # Define a common set of x-values (linear space covering all trials)
        x_common = np.linspace(min(map(np.min, all_x_vals)), max(map(np.max, all_x_vals)), 500)
        
        # Interpolate each ECDF onto the common x-values
        interpolated_y = [interp1d(x, y, bounds_error=False, fill_value=(0, 1))(x_common) for x, y in zip(all_x_vals, all_y_vals)]
        
        # Compute the mean ECDF
        mean_y = np.mean(interpolated_y, axis=0)
        plt.step(x_common, mean_y, where="post", color="b", label="PBIL")

        plt.xlabel("Wartość")
        plt.ylabel("Odsetek wartości mniejszych")
        plt.title("Empiryczna Funkcja Dystrybucji (ECDF)")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"{config.save_path}ecdf-{num_items}-{capacity}.png",
            format="png",
        )

    final_results = pd.concat(all_pbil_results, ignore_index=True)
    summary = (
        final_results.groupby(["num_items", "capacity", "optimal_solution"])
        .agg(["max", "mean", "std"])
        .reset_index()
    )
    summary.to_csv(config.save_path + "pbil.csv", float_format=__numfmt, index=False)

    final_results = pd.concat(all_a_star_results, ignore_index=True)
    summary = (
        final_results.groupby(["num_items", "capacity", "optimal_solution"])
        .agg(["max", "mean", "std"])
        .reset_index()
    )
    summary.to_csv(config.save_path + "a_star.csv", float_format=__numfmt, index=False)


def process_correlation(correlation):
    config = Config(
        data_filepaths=list(map(str, Path("data").glob(f"{correlation}/*.csv"))),
        save_path=f"output/{correlation}/",
        num_trials=10,
        max_items=1000,
    )
    test_algorithms(config)


def main():
    correlations = [
        "uncorrelated",
        "small",
        "medium_correlation",
        "strong_correlation",
    ]

    with ProcessPoolExecutor() as executor:
        executor.map(process_correlation, correlations)


if __name__ == "__main__":
    # main()
    process_correlation("uncorrelated")
