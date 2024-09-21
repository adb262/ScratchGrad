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

    def zero_grad(self):
        for param in self.input_parameters:
            param.zero_grad()

    def step(self, lr: float):
        # Only steps on input parameters
        for param in self.input_parameters:
            # print(f"{param.name}: {param.value} -> {param.value - lr * param.grad}, grad: {param.grad}")
            param.update(lr)

    def backward(self, upstream_grad):
        act_grad = self.act.backward(self.in_comb)
        delta = upstream_grad * act_grad

        for param, input_value in zip(self.input_parameters, self.last_input_values):
            param.grad += delta * input_value

        return delta
