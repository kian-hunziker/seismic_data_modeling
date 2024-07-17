import torch
import torch.nn as nn


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, x, state=None):
        return x


class LinearDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearDecoder, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, state=None):
        return self.linear(x)