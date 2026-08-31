import numpy as np
import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.

        dead_fraction_array = []

        with torch.no_grad():

            a = x

            for layer in model:
                a = layer(a)

                if isinstance(layer, nn.ReLU):

                    dead = (a <= 0).all(dim=0)

                    dead_frac = round(dead.float().mean().item(), 4)

                    dead_fraction_array.append(dead_frac)

        return dead_fraction_array

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise

        dead_fractions = np.array(dead_fractions)

        if (dead_fractions>0.5).any():
            return 'use_leaky_relu'
        elif dead_fractions[0] > 0.3:
            return 'reinitialize'
        elif (np.diff(dead_fractions) > 0).all() and dead_fractions[-1] > 0.1 :
            return 'reduce_learning_rate'
        elif np.max(dead_fractions) < 0.1:
            return 'healthy'
        else:
            return 'healthy'
