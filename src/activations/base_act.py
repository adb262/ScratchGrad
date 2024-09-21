from abc import ABC
from typing import Callable


class ActivationFunction(ABC):
    def __init__(self, fn: Callable[[float], float]):
        self.fn = fn

    def __call__(self, x: float) -> float:
        return self.fn(x)

    def backward(self, x: float) -> float:
        return self.fn(x) * (1 - self.fn(x))
