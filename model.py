
#!/usr/bin/env python
# coding: utf-8
"""
A 4-layer fully-connected ReLU-activated network
"""

import torch
from torch import nn

class fcNet(nn.Module):

    def __init__(self, dim_features, dim_labels, n_tasks):
        """
        Args:
             dim_features (int): dimension of input feature
             dim_labels (int): dimension of output label
             n_tasks (int): number of tasks
        """
        super(fcNet, self).__init__()
        # main layers
        self.main = nn.Sequential(
            nn.Linear(dim_features, dim_labels),
            nn.ReLU(),
            nn.Linear(dim_labels, dim_labels),
            nn.ReLU(),
            nn.Linear(dim_labels, dim_labels),
            nn.ReLU()
        )
        # heads for multitask
        self.heads = [nn.Linear(dim_labels, dim_labels) for _ in range(n_tasks)]
        self.mse = nn.MSELoss()

    def forward(self, x, ys):
        # prediction
        h = self.main(x)
        # mse loss per task
        loss = []
        for i, head in enumerate(self.heads):
            yp = head(h)
            loss.append(self.mse(ys[:,i], yp))
        return torch.stack(loss)
