

from src.optimizers.base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    def step(self):
        for layer in self.network.neurons_by_layer:
            for neuron in layer:
                # Only steps on input parameters
                for param in neuron.input_parameters:
                    # print(f"{param.name}: {param.value} -> {param.value - lr * param.grad}, grad: {param.grad}")
                    param.update(self.learning_rate)
