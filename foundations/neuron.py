import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        z = np.sum(w * x) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        if activation == 'sigmoid':
            return round(1/(1+np.exp(-z)), 5)
        # ReLU: max(0, z)
        elif activation == 'relu':
            return round(max(0.0, z), 5)