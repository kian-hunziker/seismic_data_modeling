import torch
import torch.nn as nn


class DummyEncoder(nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()

    def forward(self, x):
        return x


class LinearEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
