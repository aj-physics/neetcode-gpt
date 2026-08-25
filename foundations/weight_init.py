import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        torch.manual_seed(0)

        std = math.sqrt(2 / (fan_in + fan_out))

        W = torch.randn(fan_out, fan_in) * std

        return [[round(ele, 4) for ele in row] for row in W.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        std = math.sqrt(2 / fan_in)

        W = torch.randn(fan_out, fan_in) * std

        return [[round(ele, 4) for ele in row] for row in W.tolist()]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.

        torch.manual_seed(0)

        weights = []
        current_dim = input_dim

        # Generate all weights first
        for i in range(num_layers):

            if init_type == 'xavier':
                std = math.sqrt(2 / (current_dim + hidden_dim))

            elif init_type == 'kaiming':
                std = math.sqrt(2 / current_dim)

            elif init_type == 'random':
                std = 1

            W = torch.randn(hidden_dim, current_dim) * std
            weights.append(W)

            current_dim = hidden_dim

        # Generate input AFTER the weights
        x = torch.randn(input_dim)

        layer_stds = []

        for W in weights:

            x = torch.relu(W @ x)

            x_std = round(torch.std(x).item(), 2)

            layer_stds.append(x_std)

        return layer_stds




