import torch
import torch.nn as nn


class DummyEncoder(nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()

    def forward(self, x):
        return x
