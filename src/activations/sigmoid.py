
import math
from src.activations.base_act import ActivationFunction


class Sigmoid(ActivationFunction):
    epsilon = 1e-5

    def __init__(self):
        def fn(x):
            return 1 / (1 + math.exp(-x + self.epsilon))
        super().__init__(fn)

    def backward(self, x: float) -> float:
        return self.fn(x) * (1 - self.fn(x))
