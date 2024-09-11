import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config_utils import instantiate
from dataloaders.base import SequenceDataset
from tasks.positional_encoding import PositionalEncoding


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


class SigDecoder(nn.Module):
    def __init__(
            self,
            d_model=64,
            seq_len=1024,
            latent_dim=64,
            regression=False,
            vocab_size=256,
            nhead=4,
            dim_feedforward=128,
            num_layers=2
    ):
        super(SigDecoder, self).__init__()
        self.regression = regression
        self.seq_proj = nn.Linear(in_features=latent_dim, out_features=seq_len)
        self.dim_proj = nn.Linear(in_features=1, out_features=d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        if self.regression:
            self.dim_reduction = nn.Linear(in_features=d_model, out_features=1)
        else:
            self.dim_reduction = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x, state=None):
        # x: [batch_size, latent_dim]
        x = F.relu(self.seq_proj(x))
        # x: [batch_size, seq_len]
        x = F.relu(self.dim_proj(x.unsqueeze(-1)))
        # x: [batch_size, seq_len, d_model]
        x = self.pe(x.transpose(0, 1)).transpose(0, 1)
        # x: [batch_size, seq_len, d_model]
        x = self.encoder(x)
        # x: [batch_size, seq_len, d_model]
        x = self.dim_reduction(x)
        # x: [batch_size, seq_len, 1]
        return x


dec_registry = {
    'dummy': DummyDecoder,
    'linear': LinearDecoder,
    'transformer': SigDecoder
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

    if decoder._name_ == 'transformer':
        obj = instantiate(dec_registry, decoder)
        return obj

    in_features = model.d_model
    if dataset.num_classes is not None:
        out_features = dataset.num_classes
    else:
        out_features = dataset.d_data

    obj = instantiate(dec_registry, decoder, in_features=in_features, out_features=out_features)

    return obj
