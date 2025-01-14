import random
import pandas as pd
import numpy as np
from knapsack import a_star, Item


def generate_uncorrelated(sizes: list[int], max_weight):
    sizes = sorted(sizes)
    weights = [random.randint(1, max_weight) for _ in range(sizes[-1])]
    values = [random.randint(1, max_weight) for _ in range(sizes[-1])]
    for size in sizes:
        mean: int = int(np.mean(values[:size]))
        total_weight = random.randint(mean, int(mean * size / 2))
        items = [Item(values[i], weights[i]) for i in range(size)]
        best_value, _, _ = a_star(total_weight, items)
        prob_weights = [total_weight] + [item.weight for item in items]
        prob_values = [best_value] + [item.value for item in items]
        df = pd.DataFrame({"value": prob_values, "weight": prob_weights})
        df.to_csv(f"data/uncorrelated/knapPI_{size}_{total_weight}.csv", index=False)


def generate_medium_correlation(sizes: list[int], max_weight):
    sizes = sorted(sizes)
    weights = [random.randint(1, max_weight) for _ in range(sizes[-1])]
    values = [int(random.gauss(weight, 0.2) * max_weight) for weight in weights]
    for size in sizes:
        mean: int = int(np.mean(values[:size]))
        total_weight = random.randint(mean, int(mean * size / 2))
        items = [Item(values[i], weights[i]) for i in range(size)]
        best_value, _, _ = a_star(total_weight, items)
        prob_weights = [total_weight] + [item.weight for item in items]
        prob_values = [best_value] + [item.value for item in items]
        df = pd.DataFrame({"value": prob_values, "weight": prob_weights})
        df.to_csv(
            f"data/medium_correlation/knapPI_{size}_{total_weight}.csv", index=False
        )


def generate_strong_correlation(sizes: list[int], max_weight):
    sizes = sorted(sizes)
    weights = [random.randint(1, max_weight) for _ in range(sizes[-1])]
    values = [int(weight + 0.2 * max_weight) for weight in weights]
    for size in sizes:
        mean: int = int(np.mean(values[:size]))
        total_weight = random.randint(mean, int(mean * size / 2))
        items = [Item(values[i], weights[i]) for i in range(size)]
        best_value, _, _ = a_star(total_weight, items)
        prob_weights = [total_weight] + [item.weight for item in items]
        prob_values = [best_value] + [item.value for item in items]
        df = pd.DataFrame({"value": prob_values, "weight": prob_weights})
        df.to_csv(
            f"data/strong_correlation/knapPI_{size}_{total_weight}.csv", index=False
        )


if __name__ == "__main__":
    SIZES = [5, 10, 15, 20, 25, 30]
    # generate_uncorrelated(SIZES, 100)
    generate_medium_correlation(SIZES, 100)
    # generate_strong_correlation(SIZES, 100)
