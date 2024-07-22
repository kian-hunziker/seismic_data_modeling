import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config_utils import instantiate
from dataloaders.base import SequenceDataset


class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x


class DummyEncoder(Encoder):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def forward(self, x):
        return x


class LinearEncoder(Encoder):
    def __init__(self, in_features, out_features):
        super().__init__(in_features=in_features, out_features=out_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class LayerNormClassEncoder(Encoder):
    def __init__(self, in_features, num_classes, out_features):
        super().__init__(in_features, out_features)
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


enc_registry = {
    'dummy': DummyEncoder,
    'linear': LinearEncoder,
    'layernorm': LayerNormClassEncoder
}


def instantiate_encoder(encoder, dataset: SequenceDataset = None, model=None):
    if encoder is None:
        return None

    if dataset is None:
        print('Please specify dataset to instantiate encoder')
        return None

    if model is None:
        print('Please specify model to instantiate encoder')
        return None

    in_features = dataset.d_data
    out_features = model.d_model

    obj = instantiate(enc_registry, encoder, in_features=in_features, out_features=out_features)

    return obj
