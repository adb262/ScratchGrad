
from abc import ABC
from src.loss.base_loss import LossFn

from src.network import Network


class BaseOptimizer(ABC):
    def __init__(self, network: Network, learning_rate: float, loss_fn: LossFn):
        self.network = network
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

    def _step(self):
        raise NotImplementedError("Step method not implemented")

    def zero_grad(self):
        for layer in self.network.neurons_by_layer:
            for neuron in layer:
                for param in neuron.input_parameters:
                    param.zero_grad()

    def _backward(self, loss_grads: list[float]):
        raise NotImplementedError("Backward method not implemented")

    def step(self, expected: list[float], preds: list[float]):
        loss_grad = self.loss_fn.backward(expected, preds)
        self._backward(loss_grad)
        self._step()
