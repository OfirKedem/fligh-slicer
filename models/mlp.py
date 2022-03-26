import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_size: int):
        super().__init__()
        self.l1 = nn.Linear(in_size, 500)
        self.l2 = nn.Linear(500, 100)
        self.l3 = nn.Linear(100, 2)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)

        return x
