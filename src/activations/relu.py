
from src.activations.base_act import ActivationFunction


class ReLU(ActivationFunction):
    def __init__(self):
        def fn(x):
            return max(0, x)
        super().__init__(fn)

    def backward(self, x: float) -> float:
        return 1 if x > 0 else 0
