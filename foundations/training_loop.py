import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))



        w_initial = np.zeros(np.shape(X)[1])
        b_initial = 0

        i = 0

        w = w_initial
        b = b_initial

        n = len(y)

        while i < epochs:

            y_out = X @ w + b

            dL_dw = (2/n) * ((y_out - y) @ X)
            dL_db = (2/n) * np.sum((y_out - y))

            w = w - lr * dL_dw
            b = b - lr * dL_db

            i = i + 1

        return (np.round(w,5), round(b, 5))
