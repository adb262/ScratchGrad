from src.activations.base_act import ActivationFunction
from src.parameter import Parameter


class Neuron:
    def __init__(
            self, input_parameters: list[Parameter],
            activation_function: ActivationFunction):
        self.input_parameters = input_parameters
        self.act = activation_function
        self.in_comb: float = 0
        self.out: float = 0

    def forward(self, input_values: list[float]) -> float:
        assert len(input_values) == len(self.input_parameters)
        self.last_input_values = input_values  # Store for backward pass
        self.in_comb = sum([(param * input_value).value for param,
                           input_value in zip(self.input_parameters, input_values)])
        self.out = self.act(self.in_comb)
        return self.out
