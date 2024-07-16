import torch
import torch.nn as nn


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, x, state=None):

        return x
