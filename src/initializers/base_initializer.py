
from abc import ABC

from src.network import Network


class BaseInitializer(ABC):
    def __init__(self, network: Network):
        self.neurons_by_layer = network.neurons_by_layer

    def _normalize_fn(self):
        raise NotImplementedError

    def init(self):
        self._normalize_fn()
