import torch
import torch.nn as nn

from utils.config_utils import instantiate
from dataloaders.base import SequenceDataset


class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Decoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, state=None):
        return x


class DummyDecoder(Decoder):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def forward(self, x, state=None):
        return x


class LinearDecoder(Decoder):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, state=None):
        return self.linear(x)


dec_registry = {
    'dummy': DummyDecoder,
    'linear': LinearDecoder
}


def instantiate_decoder(decoder, dataset: SequenceDataset = None, model: nn.Module = None):
    if decoder is None:
        return None

    if dataset is None:
        print('Please specify dataset to instantiate encoder')
        return None

    if model is None:
        print('Please specify model to instantiate encoder')
        return None

    in_features = model.d_model
    out_features = dataset.d_data

    obj = instantiate(dec_registry, decoder, in_features=in_features, out_features=out_features)

    return obj
