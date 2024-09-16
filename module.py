

from abc import ABC
import math
import random
from typing import Callable

from sklearn.datasets import make_moons


class Parameter:
    def __init__(self, name: str, value: float, origin: int):
        self.origin = origin
        self.name = name
        self.value = value
        self.grad = 0.0  # Initialize gradient to 0
        self.grad_fn = lambda: None

    def zero_grad(self):
        self.grad = 0.0  # Reset gradient to 0

    def update(self, lr: float):
        self.value -= lr * self.grad

    def __mul__(self, other) -> "Parameter":
        if isinstance(other, int) or isinstance(other, float):
            other = Parameter(f"{other}", other, self.origin)
        out_val = Parameter(f"{self.name}*{other.name}", self.value * other.value, self.origin)

        def grad_fn():
            self.grad += other.value * other.grad
            other.grad += self.value * self.grad

        out_val.grad_fn = grad_fn
        return out_val

    def __add__(self, other) -> "Parameter":
        if isinstance(other, float):
            other = Parameter(f"{other}", other, self.origin)
        out_val = Parameter(f"{self.name}+{other.name}", self.value + other.value, self.origin)

        def grad_fn():
            self.grad += other.grad
            other.grad += self.grad

        out_val.grad_fn = grad_fn
        return out_val


class ActivationFunction(ABC):
    def __init__(self, fn: Callable[[float], float]):
        self.fn = fn

    def __call__(self, x: float) -> float:
        return self.fn(x)

    def backward(self, x: float) -> float:
        return self.fn(x) * (1 - self.fn(x))


class Sigmoid(ActivationFunction):
    epsilon = 1e-5

    def __init__(self):
        def fn(x):
            return 1 / (1 + math.exp(-x + self.epsilon))
        super().__init__(fn)

    def backward(self, x: float) -> float:
        return self.fn(x) * (1 - self.fn(x))


class ReLU(ActivationFunction):
    def __init__(self):
        def fn(x):
            return max(0, x)
        super().__init__(fn)

    def backward(self, x: float) -> float:
        return 1 if x > 0 else 0


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


class Network:
    def __init__(self, input_dimensions: int, hidden_dimensions: int, output_dimensions: int):
        hidden_out_params = [Parameter(f"hidden_out_{i}", 0, i) for i in range(hidden_dimensions)]
        hidden_in_params = [Parameter(f"hidden_in_{i}", 0, i) for i in range(input_dimensions)]

        # Init
        out_neurons = [Neuron(hidden_out_params, Sigmoid()) for _ in range(output_dimensions)]
        hidden_neurons = [Neuron(hidden_in_params, ReLU()) for _ in range(hidden_dimensions)]

        self._neurons_by_layer = [hidden_neurons, out_neurons]

    def normalize_layers(self):
        # For now, just set the values of the parameters to be uniformly random
        for layer in self._neurons_by_layer:
            for neuron in layer:
                for param in neuron.input_parameters:
                    param.value = random.normalvariate(0, 1)

    def zero_grad(self):
        for layer in self._neurons_by_layer:
            for neuron in layer:
                neuron.zero_grad()

    def forward(self, input_values: list[float]) -> list[float]:
        for layer in self._neurons_by_layer:
            out_values = []
            for neuron in layer:
                out_values.append(neuron.forward(input_values))
            input_values = out_values
        return out_values

    def backward(self, loss: float):
        upstream_grad = [loss]
        for layer in self._neurons_by_layer[::-1]:
            next_upstream_grad = [0.0] * len(layer[0].input_parameters)
            for i, neuron in enumerate(layer):
                delta = neuron.backward(upstream_grad[i])
                for j in range(len(next_upstream_grad)):
                    # Just multiply for now
                    origin = neuron.input_parameters[j].origin
                    next_upstream_grad[origin] += delta * neuron.input_parameters[j].value
            upstream_grad = next_upstream_grad

    def step(self, lr: float):
        for layer in self._neurons_by_layer:
            for neuron in layer:
                neuron.step(lr)


class LossFn(ABC):
    def __init__(self, fn: Callable[[float, float], float]):
        self.fn = fn

    def __call__(self, outputs, expecteds):
        return self.fn(outputs, expecteds)

    def backward(self, outputs, expecteds):
        raise NotImplementedError


class MSE(LossFn):
    def __init__(self):
        def fn(output, expected): return (output - expected) ** 2
        super().__init__(fn)

    def backward(self, output, expected):
        return 2 * (output - expected)


class BinaryCrossEntropy(LossFn):
    epsilon = 1e-15

    def __init__(self):
        def fn(output, expected):
            out = expected * math.log(output + self.epsilon) - (1 - expected) * math.log(1 - output + self.epsilon)
            return -1 * out
        super().__init__(fn)

    def backward(self, output, expected):
        return - (expected / (output + self.epsilon) - (1 - expected) / (1 - output + self.epsilon))


# generate 2d classification dataset
X, y = make_moons(n_samples=10000, shuffle=True,
                  noise=0.00, random_state=42)
X = X.tolist()
y = y.tolist()

net = Network(2, 1, 1)
net.normalize_layers()
lr = 0.1
epochs = 10
batch_size = 2

loss_fn = BinaryCrossEntropy()

for epoch in range(epochs):
    epoch_losses = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        batch_loss = 0
        for x, y_true in zip(batch_X, batch_y):
            preds = net.forward(x)[0]
            loss = loss_fn(preds, y_true)
            batch_loss += loss
            net.backward(loss_fn.backward(preds, y_true))

        net.zero_grad()
        epoch_losses.append(batch_loss / batch_size)

    print(f"Epoch {epoch+1}, Average Loss: {sum(epoch_losses) / len(epoch_losses):.4f}")

# Test
accuracy = 0
for i in range(len(X)):
    preds = net.forward(X[i])[0]
    if preds > 0.5:
        preds = 1
    else:
        preds = 0
    if preds == y[i]:
        accuracy += 1
print(f"Accuracy: {accuracy / len(X)}")
