from .typing import Item
from functools import total_ordering
import numpy as np
import random
from tqdm import tqdm


@total_ordering
class Specimen:
    """
    Class to represent a specimen in the PBIL algorithm.
    # Fields:
    items (`list[bool]`) - the representation vector of items picked for the specimen.
    value (`int | float`) - the total value of the items picked (0 if the weight is too much).
    """

    items: list[bool]
    value: int | float

    def __init__(self, items, probabilities, capacity_limit) -> None:
        self.items = [random.random() < p for p in probabilities]
        picked_items = [item for item, is_picked in zip(items, self.items) if is_picked]
        self.value = sum(item.value for item in picked_items)
        if sum(item.weight for item in picked_items) > capacity_limit:  # penalty
            self.value -= sum(item.value for item in items)

    def __repr__(self) -> str:
        return f"Specimen(value={self.value}, items={self.items})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Specimen):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Specimen):
            return NotImplemented
        return self.value < other.value


def pbil(
    total_capacity: int | float,
    items: list[Item],
    population_size: int = 100,
    num_generations: int = 1000,
    num_best: int = 10,
    learning_rate: float = 0.1,
    mutation_probability: float = 0.1,
    mutation_std: float = 0.1,
    threshold: float = 1e-4,
    show_progress: bool = False,
) -> tuple[int | float, list[bool], list[list[int | float]], list[float]]:
    """
    Solve the knapsack problem using the Population-Based Incremental Learning (PBIL) algorithm.
    # Args:
        total_capacity (`int | float`): the total capacity of the knapsack.
        items (`list[Item]`): the list of items.
        population_size (`int`): the size of the population.
        num_generations (`int`): the number of generations.
        num_best (`int`): the number of best specimens to keep.
        learning_rate (`float`): the learning rate.
        mutation_probability (`float`): the chance of mutation for a single item probability.
        mutation_std (`float`): the standard deviation for the gaussian mutation (exceeding 0.33 is not advised).
        threshold (`float`): the threshold for the change in the probability vector to stop the algorithm.
        show_progress (`bool`): whether to display the progress bar.
    # Returns:
        `int | float`: best value found.
        `list[bool]`: the list of booleans indicating whether the item at the corresponding index is picked.
        `list[list[int | float]]`: list of best specimen values in each generation.
        `list[float]`: the probability vector.
    """
    num_items = len(items)
    p = np.full(num_items, 1 / 2)
    p_prev = None
    best_value = 0
    best_specimen = None
    best_values: list[list[int | float]] = []
    # optionally display the progress bar
    iterator = (
        tqdm(range(1, num_generations + 1))
        if show_progress
        else range(1, num_generations + 1)
    )
    for i in iterator:
        # generate population
        population = [
            Specimen(items, p, total_capacity) for _ in range(population_size)
        ]
        population = sorted(population, reverse=True)
        # save best specimen
        if population[0].value > best_value:
            best_value = population[0].value
            best_specimen = population[0]
        # select best specimens
        selected = population[:num_best]
        # keep track of best values
        best_values.append([spec.value for spec in selected])

        occurrence_counts = [
            sum(col) for col in zip(*[spec.items for spec in selected])
        ]

        # update the probability vector
        for i in range(num_items):
            p[i] = (1 - learning_rate) * p[i] + learning_rate * (
                occurrence_counts[i] / num_best
            )

        # apply mutation
        for i in range(num_items):
            if random.random() < mutation_probability:
                mutation = random.gauss(0, mutation_std)
                p[i] = min(max(p[i] + mutation, 0), 1)

        # additional stop condition
        if p_prev is not None:
            l1_change = np.sum(np.abs(p - p_prev))
            max_change = np.max(np.abs(p - p_prev))
            # if the change is too small, stop
            if l1_change < threshold and max_change < threshold:
                break
        p_prev = np.copy(p)

    assert p is not None
    if best_specimen is None:
        representation_vector = [False] * len(items)
    else:
        representation_vector = best_specimen.items
    return best_value, representation_vector, best_values, p.tolist()
