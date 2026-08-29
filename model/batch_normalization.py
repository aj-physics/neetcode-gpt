import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        mu = np.mean(x, axis = 0)

        var = np.mean((x - mu[None, :])**2, axis = 0)

        # print('mu = ', mu)
        # print('running_mean = ',running_mean)
        # print('running_var = ', running_var)

        if training == False:

            normalize = (x - np.array(running_mean)[None, :]) / np.sqrt(np.array(running_var)[None, :] + eps)

            print(normalize)

        else:

            running_mean = (1-momentum) * np.array(running_mean) + momentum * mu

            running_var = (1-momentum) * np.array(running_var) + momentum * var            

            normalize = (x - mu[None, :]) / np.sqrt(var[None, :] + eps)

        y = np.array(gamma)[None, :] * normalize + beta


        return np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4)
