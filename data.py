#!/usr/bin/env python
# coding: utf-8
"""
A toy example dataset
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class toyDataset(Dataset):
    """
    This class is Torch Dataset for multitask toy example
    """

    def __init__(self, num_data, dim_features, dim_labels, scalars):
        """
        Args:
            num_data (int): number of data points
            dim_features (int): dimension of features
            dim_labels (int): dimension of labels
            scalars (list(float)): set the scales of the outputs
        """
        self.n = num_data
        self.p = dim_features
        self.d = dim_labels
        self.sigmas = np.array(scalars)
        # constant matrices B from a normal distribution N(0,10)
        self.B = np.random.normal(scale=10, size=(self.p, self.d))
        # matrices epsilon for task-specific information
        self.epsilons = np.random.normal(scale=3.5, size=(len(self.sigmas), self.p, self.d))
        # normalized features
        self.x = np.random.uniform(0, 1, size=(self.n,self.p))
        # labels per task
        ys = []
        for i in range(len(self.sigmas)):
            y = self.sigmas[i] * np.tanh(self.x @ (self.B + self.epsilons[i]))
            ys.append(y)
        self.ys = np.stack(ys, axis=1)

    def __len__(self):
        return self.n

    def __getitem__(self, ind):
        return torch.FloatTensor(self.x[ind]), torch.FloatTensor(self.ys[ind])
