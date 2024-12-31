class Item:
    value: int | float
    weight: int | float
    ratio: float
    picked: bool | None
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
        self.ratio = value / weight
        self.picked = None