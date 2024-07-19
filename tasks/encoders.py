import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LayerNormClassEncoder(nn.Module):
    def __init__(self, in_features, num_classes, out_features):
        super(LayerNormClassEncoder, self).__init__()
        self.lin_1 = nn.Linear(in_features, num_classes)
        self.lin_2 = nn.Linear(num_classes, out_features)
        self.input_norm = nn.LayerNorm(in_features)
        self.class_norm = nn.LayerNorm(num_classes)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.lin_1(x)
        x = F.relu(x)
        x = self.class_norm(x)
        x = self.lin_2(x)
        return x
