
import math
import random
from src.initializers.base_initializer import BaseInitializer


class XavierInitializer(BaseInitializer):
    def _normalize_fn(self):
        for i, layer in enumerate(self.neurons_by_layer):
            for neuron in layer:
                inputs = len(neuron.input_parameters)
                outputs = len(self.neurons_by_layer[i + 1]) if i + 1 < len(self.neurons_by_layer) else 1
                min_val = -math.sqrt(6 / (inputs + outputs))
                max_val = math.sqrt(6 / (inputs + outputs))
                for param in neuron.input_parameters:
                    param.value = random.uniform(min_val, max_val)
