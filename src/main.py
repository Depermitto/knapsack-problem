import knapsack
import os
import json
from time import time

FILEPATH = 'data/small/'
RESULT_FILE = 'results/a_star/small.jsonl'

def save_results(fname, optimal, time, best_values):
    res = {
        'filename': fname,
        'optimal': optimal,
        'time': time,
        'best_values': best_values
    }
    with open(RESULT_FILE, 'a') as f:
        json.dump(res, f)
        f.write('\n')

def main():
    FILENAMES = os.listdir(FILEPATH)
    for fname in FILENAMES:
        optimal, capacity, items = knapsack.read_data(FILEPATH + fname)
        start = time()
        value, taken_items, best_values = knapsack.a_star(capacity, items)
        end = time()
        print(f"Expected value: {optimal}, Computed value: {value}, Iterations: {best_values[-1][0]}")
        save_results(fname, optimal, end - start, best_values)

if __name__ == '__main__':
    main()
