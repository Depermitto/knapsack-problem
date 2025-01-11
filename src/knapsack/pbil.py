from .typing import Item
import random
from functools import total_ordering


@total_ordering
class Specimen:
    """
    Class to represent a specimen in the PBIL algorithm.
    # Fields:
    items (`list[Item]`) - the list of items picked for the specimen. \\
    value (`int | float`) - the total value of the items picked (0 if the weight is too much).
    """

    items: list[Item]
    value: int | float

    def __init__(self, items, probabilities, capacity_limit) -> None:
        mask = [random.random() < p for p in probabilities]
        self.items = [item for item, is_picked in zip(items, mask) if is_picked]
        if sum(item.weight for item in self.items) > capacity_limit:
            self.value = 0
        else:
            self.value = sum(item.value for item in self.items)

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
    num_generations: int = 100,
    num_best: int = 10,
    learning_rate: float = 0.1,
    mutation_rate: float = 0.05,
    mutation_shift: float = 0.05,
) -> tuple[int | float, list[bool], list[list[int | float]], list[float]]:
    """
    Solve the knapsack problem using the Population-Based Incremental Learning (PBIL) algorithm.
    # Args:
        total_capacity (`int | float`): the total capacity of the knapsack. \\
        items (`list[Item]`): the list of items. \\
        population_size (`int`): the size of the population. \\
        num_generations (`int`): the number of generations. \\
        num_best (`int`): the number of best specimens to keep. \\
        learning_rate (`float`): the learning rate. \\
        mutation_rate (`float`): the mutation rate. \\
        mutation_shift (`float`): the mutation shift. \\
    # Returns:
        `int | float`: best value found.
        `list[bool]`: the list of booleans indicating whether the item at the corresponding index is picked.
        `list[list[int | float]]`: list of best specimen values in each generation.
        `list[float]`: the probability vector.
    """
    items = sorted(
        items, key=lambda x: x.ratio, reverse=True
    )  # sort items by value/weight ratio to match A* algorithm
    num_items = len(items)
    probability_vector = [0.5] * num_items
    best_value = 0
    best_specimen = None
    best_values: list[list[int | float]] = []
    for i in range(1, num_generations + 1):
        # generate population
        population = [
            Specimen(items, probability_vector, total_capacity)
            for _ in range(population_size)
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

        # count occurrences of each item in the best specimens
        occurrence_counts = [0] * num_items
        for specimen in selected:
            for i, item in enumerate(items):
                if item in specimen.items:
                    occurrence_counts[i] += 1

        # update the probability vector
        for i in range(num_items):
            probability_vector[i] = (1 - learning_rate) * probability_vector[
                i
            ] + learning_rate * (occurrence_counts[i] / num_best)

        # aply mutation
        for i in range(num_items):
            if random.random() < mutation_rate:
                mutation = random.uniform(-mutation_shift, mutation_shift)
                probability_vector[i] = min(max(probability_vector[i] + mutation, 0), 1)

    representation_vector = [True if item in best_specimen.items else False for item in items]  # type: ignore
    return best_value, representation_vector, best_values, probability_vector
