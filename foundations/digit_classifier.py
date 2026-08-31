import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid

        self.model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(512, 10), nn.Sigmoid())


    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places

        out = self.model(images)

        rounded_list = [
        [round(value, 4) for value in row]
        for row in out.tolist()]

        print(rounded_list)


        return rounded_list
