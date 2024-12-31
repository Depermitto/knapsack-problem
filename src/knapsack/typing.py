from functools import total_ordering

@total_ordering
class Item:
    """
    Class to represent an item in the knapsack problem.
    # Fields:
    value (`int | float`) - the value of the item. \\
    weight (`int | float`) - the weight of the item. \\
    ratio (`float`) - the value-to-weight ratio of the item. \\
    picked (`bool | None`) - whether the item is picked or not. \\
    """

    value: int | float
    weight: int | float
    ratio: float
    picked: bool | None

    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
        self.ratio = value / weight
        self.picked = None

    def __repr__(self):
        return f"Item(value={self.value}, weight={self.weight}, ratio={self.ratio}, picked={self.picked})"

    def __eq__(self, other):
        if not isinstance(other, Item):
            return NotImplemented
        return self.ratio == other.ratio
    
    def __lt__(self, other):
        if not isinstance(other, Item):
            return NotImplemented
        return self.ratio < other.ratio
