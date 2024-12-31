import pandas as pd
from .typing import Item


def read_data(file_path: str) -> tuple[int | float, int | float, list[Item]]:
    """
    Read data from a CSV file and return the optimal value, total capacity, and a list of items.
    # Args:
        file_path (`str`) - the path to the CSV file.
    # Returns:
        result (`tuple[int | float, int | float, list[Item]]`) - a tuple containing the optimal value, total capacity, and a list of items.
    """
    df = pd.read_csv(file_path)
    optimal_value = df.iloc[0, 0]
    total_capacity = df.iloc[0, 1]
    items = []
    for i in range(1, len(df)):
        item = Item(df.iloc[i, 0], df.iloc[i, 1])
        items.append(item)
    return optimal_value, total_capacity, items
