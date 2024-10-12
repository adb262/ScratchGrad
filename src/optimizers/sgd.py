

from src.neuron import Neuron
from src.optimizers.base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    def _step(self):
        for layer in self.network.neurons_by_layer:
            for neuron in layer:
                # Only steps on input parameters
                for param in neuron.input_parameters:
                    print(f"{param.name}: {param.value} -> {param.value - self.learning_rate * param.grad}, grad: {param.grad}")

                    param.update(self.learning_rate)

    def _backward(self, loss_grads: list[float]):
        for loss in loss_grads:
            upstream_grad = [loss]
            for layer in self.network.neurons_by_layer[::-1]:
                next_upstream_grad = [0.0] * len(layer[0].input_parameters)
                for i, neuron in enumerate(layer):
                    delta = self._neuron_backward(neuron, upstream_grad[i])
                    for j in range(len(next_upstream_grad)):
                        # Just multiply for now
                        origin = neuron.input_parameters[j].origin
                        next_upstream_grad[origin] += delta * neuron.input_parameters[j].value
                upstream_grad = next_upstream_grad

    def _neuron_backward(self, neuron: Neuron, upstream_grad: float):
        act_grad = neuron.act.backward(neuron.in_comb)
        delta = upstream_grad * act_grad

        for param, input_value in zip(neuron.input_parameters, neuron.last_input_values):
            param.grad += delta * input_value

        return delta
