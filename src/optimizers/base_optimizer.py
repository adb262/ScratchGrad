
from abc import ABC

from src.network import Network


class BaseOptimizer(ABC):
    def __init__(self, network: Network, learning_rate: float):
        self.network = network
        self.learning_rate = learning_rate

    def step(self):
        raise NotImplementedError("Step method not implemented")

    def zero_grad(self):
        for layer in self.network.neurons_by_layer:
            for neuron in layer:
                for param in neuron.input_parameters:
                    param.zero_grad()
