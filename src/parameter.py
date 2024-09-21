class Parameter:
    def __init__(self, name: str, value: float, origin: int):
        self.origin = origin
        self.name = name
        self.value = value
        self.grad = 0.0  # Initialize gradient to 0
        self.grad_fn = lambda: None

    def zero_grad(self):
        self.grad = 0.0  # Reset gradient to 0
        self.grad_fn = lambda: None

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
