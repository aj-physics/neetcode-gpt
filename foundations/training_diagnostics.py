import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        
        data_recorded = []

        with torch.no_grad():

            a = x

            for layer in model:
                a = layer(a)

                if isinstance(layer, nn.Linear):

                    mean_val = round((torch.mean(a)).item(),4)
                    std_val = round((torch.std(a)).item(),4)
                    dead = (a <= 0).all(dim=0)
                    dead_frac = round(dead.float().mean().item(), 4)

                    data_recorded.append({'mean': mean_val, 
                    'std': std_val, 
                    'dead_fraction': dead_frac})
        
        return data_recorded

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.

        model.zero_grad()

        pred = model(x)

        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, y)

        loss.backward()

        data_recorded = []

        for layer in model:

            if isinstance(layer, nn.Linear):

                grad = layer.weight.grad

                mean_val = round((torch.mean(grad)).item(),4)
                std_val = round((torch.std(grad)).item(),4)
                norm_val = round(torch.norm(grad).item(), 4)

                data_recorded.append({
                    'mean': mean_val,
                    'std': std_val,
                    'norm': norm_val
                })

        return data_recorded

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)


        dead_neurons = [ele['dead_fraction'] for ele in activation_stats]

        grad_stat_info = [ele['norm'] for ele in gradient_stats]

        act_stat_info = [ele['std'] for ele in activation_stats]

        for dead_frac in dead_neurons:
            if dead_frac > 0.5:
                return 'dead_neurons'

        for i, norm in enumerate(grad_stat_info):
            if norm > 1000:
                return 'exploding_gradients'

            if i == len(grad_stat_info)-1:
                if norm < 1e-5:
                    return 'vanishing_gradients'

        for std in act_stat_info:
            if std > 10.0:
                return 'exploding_gradients'
            if std < 0.1:
                return 'vanishing_gradients'

        return 'healthy'