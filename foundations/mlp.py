import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        
        n_layers = len(weights)

        a = x

        for ly in range(n_layers):
            
            z = np.einsum('i, ij -> j', a, weights[ly]) + biases[ly]
            print(z)
            a = np.where(z>0, z, 0)
            print(a)

        return np.round(z, 5)


        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
