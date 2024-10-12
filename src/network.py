from src.activations.relu import ReLU
from src.activations.sigmoid import Sigmoid
from src.neuron import Neuron
from src.parameter import Parameter


class Network:
    def __init__(self, input_dimensions: int, hidden_dimensions: int, output_dimensions: int):
        hidden_out_params = [Parameter(f"hidden_out_{i}", 0, i) for i in range(hidden_dimensions)]
        hidden_in_params = [Parameter(f"hidden_in_{i}", 0, i) for i in range(input_dimensions)]

        # Init
        out_neurons = [Neuron(hidden_out_params, Sigmoid()) for _ in range(output_dimensions)]
        hidden_neurons = [Neuron(hidden_in_params, ReLU()) for _ in range(hidden_dimensions)]

        self._neurons_by_layer = [hidden_neurons, out_neurons]

    @property
    def neurons_by_layer(self):
        return self._neurons_by_layer

    def forward(self, input_values: list[list[float]]) -> list[list[float]]:
        predictions = []
        for inputs in input_values:
            for layer in self._neurons_by_layer:
                out_values = []
                for neuron in layer:
                    out_values.append(neuron.forward(inputs))
                inputs = out_values
            predictions.append(out_values)
        return predictions
