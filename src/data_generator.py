import os
import random
from typing import List, Tuple, Literal
import pandas as pd

from knapsack import a_star, Item


def generate_items(
    num_items: int,
    correlation: Literal["uncorrelated", "medium_correlation", "strong_correlation"],
    min_weight: int = 5,
    max_weight: int = 100,
    min_value: int = 10,
    max_value: int = 500,
) -> List[Tuple[int, int]]:
    """
    Generate a list of (value, weight) pairs based on correlation type.
    """
    weights = [random.randint(min_weight, max_weight) for _ in range(num_items)]
    if correlation == "uncorrelated":
        values = [random.randint(min_value, max_value) for _ in range(num_items)]
    elif correlation == "medium_correlation":
        values = [int(w * 0.5 + random.uniform(-0.5, 0.5) * w) + 1 for w in weights]
    elif correlation == "strong_correlation":
        values = [int(w * (random.uniform(0.9, 1.1))) for w in weights]
    return list(zip(values, weights))


def generate_dataset():
    """
    Generate and save knapsack datasets for different sizes and correlation types.
    """
    CORR_TYPES = ["uncorrelated", "medium_correlation", "strong_correlation"]
    ITEM_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]
    BASE_DIR = "data"
    os.makedirs(BASE_DIR, exist_ok=True)

    for correlation in CORR_TYPES:
        correlation_dir = os.path.join(BASE_DIR, correlation)
        os.makedirs(correlation_dir, exist_ok=True)

        for num_items in ITEM_SIZES:
            print(f"\nGenerating: {correlation} - {num_items}")
            items = generate_items(num_items, correlation)  # type: ignore

            # capacity is 10% more than the sum of weights of half the items
            half_items = items[: num_items // 2]
            capacity = int(sum(w for _, w in half_items) * 1.1)

            # get optimal value and items
            optimal_value, res, _ = a_star(capacity, [Item(i[0], i[1]) for i in items])

            # print some info about dataset
            idf = pd.DataFrame(items, columns=["value", "weight"])
            print(
                f"Optimal value for {num_items} items ({correlation}): {optimal_value}; items: {sum(res)}; correlation:"
            )
            print(idf.corr())

            # save to csv
            filename = os.path.join(correlation_dir, f"knapsack_{num_items}.csv")
            # add optimal value and capacity first
            idf.loc[-1] = [optimal_value, capacity]
            idf.index += 1
            idf = idf.sort_index()
            idf.to_csv(filename, index=False)


if __name__ == "__main__":
    generate_dataset()
