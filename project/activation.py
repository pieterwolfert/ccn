import numpy as np


class Tanh:
    """Tanh activation function for forward and backward pass."""
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * top_diff
