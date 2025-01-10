from .typing import Item
from heapq import heappop, heappush
from functools import total_ordering


@total_ordering
class State:
    """
    Class to represent a state in the A* search for the knapsack problem.
    # Fields:
    total_value (`int | float`) - the total value of the items in the knapsack so far. \\
    current_weight (`int | float`) - the total weight of the items in the knapsack so far. \\
    remaining_capacity (`int | float`) - the remaining capacity of the knapsack. \\
    curr_item_index (`int`) - the current index in the item list. \\
    heuristic_value (`int | float`) - the heuristic function value. \\
    picked_items (`list[Item]`) - the list of picked items.
    """

    current_value: int | float
    current_weight: int | float
    remaining_capacity: int | float
    heuristic_value: int | float
    curr_item_index: int
    picked_items: list[Item]

    def __init__(self, value, weight, capacity, heuristic, index, picked_items):
        self.current_value = value
        self.current_weight = weight
        self.remaining_capacity = capacity
        self.heuristic_value = heuristic
        self.curr_item_index = index
        self.picked_items = picked_items

    def __repr__(self):
        return f"State(total_value={self.current_value}, current_weight={self.current_weight}, remaining_capacity={self.remaining_capacity}, heuristic_value={self.heuristic_value}, curr_item_index={self.curr_item_index}, picked_items={self.picked_items})"

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return (
            self.current_value + self.heuristic_value
            == other.current_value + other.heuristic_value
        )

    def __lt__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return ( # reverse order for heapq to be more efficient
            self.current_value + self.heuristic_value
            >= other.current_value + other.heuristic_value
        )


def a_star(
    total_capacity: int | float, items: list[Item]
) -> tuple[int | float, list[Item]]:
    """
    Solve the knapsack problem using A* search.
    # Args:
        total_capacity (`int | float`): the total capacity of the knapsack. \\
        items (`list[Item]`): List of the items that are available to be picked.

    # Returns:
        `tuple[int | float, list[Item]]`: the total value of the items picked and the list of items picked.
    """

    def heuristic(items: list[Item], capacity: int | float, item_index: int):
        """
        Calculate the heuristic value for the given items, capacity, and item index.
        # Args:
            items (`list[Item]`): list of items that can be picked. \\
            capacity (`int | float`): the remaining capacity of the knapsack. \\
            item_index (`int`): the index of the current item under consideration.

        # Returns:
            `int | float`: the calculated heuristic value.
        """
        remaining_capacity = capacity
        value = 0
        for i in range(item_index, len(items)):
            if items[i].weight <= remaining_capacity:
                remaining_capacity -= items[i].weight
                value += items[i].value
            else:
                value += items[i].ratio * remaining_capacity
                break
        return value

    # sort the items by value-to-weight ratio
    items.sort(key=lambda item: item.ratio, reverse=True)

    # initialize the queue with the initial state
    queue = []
    initial_state = State(
        0, 0, total_capacity, heuristic(items, total_capacity, 0), 0, []
    )
    heappush(queue, initial_state)

    best_value = 0
    best_items = []

    while queue:
        current_state: State = heappop(queue)

        # if it's the last item, check if it's the best solution
        if current_state.curr_item_index == len(items):
            if current_state.current_value > best_value:
                best_value = current_state.current_value
                best_items = current_state.picked_items
            break

        # retrieve the item under consideration
        item: Item = items[current_state.curr_item_index]

        # create a state where the item is picked (if it fits)
        if current_state.current_weight + item.weight <= total_capacity:
            new_state = State(
                current_state.current_value + item.value,
                current_state.current_weight + item.weight,
                current_state.remaining_capacity - item.weight,
                heuristic(
                    items,
                    current_state.remaining_capacity - item.weight,
                    current_state.curr_item_index + 1,
                ),
                current_state.curr_item_index + 1,
                current_state.picked_items + [item],
            )
            heappush(queue, new_state)

        # create a state where the item is not picked
        new_state = State(
            current_state.current_value,
            current_state.current_weight,
            current_state.remaining_capacity,
            heuristic(
                items,
                current_state.remaining_capacity,
                current_state.curr_item_index + 1,
            ),
            current_state.curr_item_index + 1,
            current_state.picked_items,
        )
        heappush(queue, new_state)

    return best_value, best_items
