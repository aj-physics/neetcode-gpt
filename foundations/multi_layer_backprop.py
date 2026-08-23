import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:

        z1 = np.einsum('ij,j->i', W1, x) + b1
        a1 = np.where(z1>0, z1, 0)
        z2 = np.einsum('ij,j->i', W2, a1) + b2

        n = len(y_true)
        loss = (1/n) * np.sum((z2 - y_true)**2)

        dL_dz2 = (2/n) * (z2 - y_true)
        dL_da1 = np.einsum('i, ij -> j',dL_dz2, W2)
        dL_dz1 = dL_da1 * np.where(z1>0, 1, 0)
        # print(dL_dz1)

        dW1 = np.einsum('i,j -> ij', dL_dz1, x)
        db1 = dL_dz1
        dW2 = np.einsum('i,j -> ij', dL_dz2, a1)
        db2 = dL_dz2

        return dict(
            loss = np.round(loss,5),
            dW1 = np.round(dW1,5),
            db1 = np.round(db1,5),
            dW2 = np.round(dW2,5),
            db2 = np.round(db2,5),
        )

        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        
