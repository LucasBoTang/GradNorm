
#!/usr/bin/env python
# coding: utf-8
"""
A 4-layer fully-connected ReLU-activated network
"""

import torch
from torch import nn
import torch.nn.functional as F

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
        self.fc1 = nn.Linear(dim_features, dim_labels)
        self.fc2 = nn.Linear(dim_labels, dim_labels)
        self.fc3 = nn.Linear(dim_labels, dim_labels)
        self.fc4 = nn.Linear(dim_labels, dim_labels)
        # heads for multitask
        self.heads = nn.ModuleList([nn.Linear(dim_labels, dim_labels) for _ in range(n_tasks)])
        self.mse = nn.MSELoss()

    def forward(self, x, ys):
        # prediction
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        # mse loss per task
        loss = []
        for i, head in enumerate(self.heads):
            yp = head(h)
            loss.append(self.mse(ys[:,i], yp))
        return torch.stack(loss)
