import knapsack
import os
import json
from time import time
import matplotlib.pyplot as plt
import numpy as np

FILEPATH = "data/small/"
RESULT_FILE = "results/a_star/strong_correlation.jsonl"


def save_results(fname, optimal, time, best_values):
    res = {
        "filename": fname,
        "optimal": optimal,
        "time": time,
        "best_values": best_values,
    }
    with open(RESULT_FILE, "a") as f:
        json.dump(res, f)
        f.write("\n")


def main():
    FILENAMES = os.listdir(FILEPATH)
    for fname in FILENAMES:
        optimal, capacity, items = knapsack.read_data(FILEPATH + fname)
        start = time()
        value, taken_items, best_values = knapsack.a_star(capacity, items)
        end = time()
        print(
            f"Expected value: {optimal}, Computed value: {value}, Iterations: {best_values[-1][0]}"
        )
        save_results(fname, optimal, end - start, best_values)


if __name__ == "__main__":
    optimal, capacity, items = knapsack.read_data(FILEPATH + "f1_l-d_kp_10_269.csv")
    value, taken_items, best_values, probs = knapsack.pbil(
        capacity, items, threshold=0.1, num_generations=1000
    )
    print(f"Expected value: {optimal}, Computed value: {value}")
    x_values = range(len(best_values) + 1)
    for x, values in zip(x_values, best_values):
        plt.scatter([x] * len(values), values)
    plt.suptitle(f"PBIL: Expected value: {optimal}, Computed value: {value}")
    plt.title(f"{np.round(probs, 2)}")
    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.show()
