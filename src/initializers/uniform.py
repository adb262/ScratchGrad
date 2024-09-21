
import random
from src.initializers.base_initializer import BaseInitializer


class UniformInitializer(BaseInitializer):
    def _normalize_fn(self):
        # For now, just set the values of the parameters to be uniformly random
        for layer in self.neurons_by_layer:
            for neuron in layer:
                for param in neuron.input_parameters:
                    param.value = random.uniform(0, 1) - 0.5
