
from abc import ABC
import math
from typing import Callable


class LossFn(ABC):
    def __init__(self, fn: Callable[[list[float], list[float]], float]):
        self.fn = fn

    def __call__(self, outputs: list[float], expecteds: list[float]):
        return self.fn(outputs, expecteds)

    def backward(self, outputs: list[float], expecteds: list[float]):
        raise NotImplementedError


class MSE(LossFn):
    def __init__(self):
        def fn(preds: list[float], expected: list[float]):
            assert len(preds) == len(expected)
            return sum((pred - exp) ** 2 for pred, exp in zip(preds, expected)) / len(preds)

        super().__init__(fn)

    def backward(self, preds: list[float], expected: list[float]):
        assert len(preds) == len(expected)
        return [2 * (pred - exp) for pred, exp in zip(preds, expected)]


class BinaryCrossEntropy(LossFn):
    epsilon = 1e-15

    def __init__(self):
        def fn(preds: list[float], expected: list[float]):
            assert len(preds) == len(expected)
            out = sum(exp * math.log(pred + self.epsilon) + (1 - exp)
                      * math.log(1 - pred + self.epsilon) for pred, exp in zip(preds, expected))
            return -1 * out / len(preds)
        super().__init__(fn)

    def backward(self, preds: list[float], expected: list[float]):
        assert len(preds) == len(expected)
        return [- (exp / (pred + self.epsilon) - (1 - exp) / (1 - pred + self.epsilon)) / len(preds) for pred, exp in zip(preds, expected)]
