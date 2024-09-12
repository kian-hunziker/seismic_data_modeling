import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config_utils import instantiate
from dataloaders.base import SequenceDataset
from tasks.positional_encoding import PositionalEncoding

from models.sashimi.s4_standalone import LinearActivation, S4Block as S4
from models.sashimi.sashimi_standalone import UpPool, FFBlock, ResidualBlock


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


class LayerNormLinearEncoder(Encoder):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.input_norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.linear(x)
        return x


class MLPEncoder(Encoder):
    def __init__(self, in_features, out_features, hidden_features=256):
        super().__init__(in_features, out_features)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class EmbeddingEncoder(Encoder):
    def __init__(self, in_features, out_features, num_classes=256):
        super().__init__(in_features, out_features)
        self.embedding = nn.Embedding(num_classes, out_features)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.embedding(x)
        return x


class SigEncoder(nn.Module):
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
        super(SigEncoder, self).__init__()
        self.regression = regression

        if self.regression:
            self.embedding = nn.Linear(1, d_model)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
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
        self.dim_reduction = nn.Linear(in_features=d_model, out_features=1)
        self.out_proj = nn.Linear(in_features=seq_len, out_features=latent_dim)

    def forward(self, x, state=None):
        # x: [batch_size, seq_len, 1]
        if self.regression:
            x = self.embedding(x)
        else:
            x = self.embedding(x.squeeze())
        # x: [batch_size, seq_len, d_model]
        x = self.pe(x.transpose(0, 1)).transpose(0, 1)
        # x: [batch_size, seq_len, d_model]
        x = self.encoder(x)
        # x: [batch_size, seq_len, d_model]
        x = F.relu(self.dim_reduction(x)).squeeze(-1)
        # x: [batch_size, seq_len]
        x = self.out_proj(x)
        # x: [batch_size, latent_dim]
        return x


class S4Encoder(nn.Module):
    def __init__(self, d_model, n_blocks, bidirectional=False):
        super(S4Encoder, self).__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.bidirectional = bidirectional

        # self.input_linear = nn.Linear(1, d_model)
        self.input_linear = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=5, stride=1, padding=2)

        def s4_block(dim, bidirectional=False, dropout=0.0, **s4_args):
            layer = S4(
                d_model=dim,
                d_state=64,
                bidirectional=bidirectional,
                dropout=dropout,
                transposed=True,
                **s4_args,
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
            )

        def ff_block(dim, ff=2, dropout=0.0):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=dropout,
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
            )

        blocks = []
        for i in range(n_blocks):
            blocks.append(s4_block(dim=d_model))
            blocks.append(ff_block(dim=d_model))

        self.blocks = nn.ModuleList(blocks)

    def _forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_linear(x)
        for block in self.blocks:
            x, _ = block(x)
        return x[:, :, -1]

    def forward(self, x, state=None):
        if self.bidirectional:
            x_rev = torch.flip(x, dims=[1])
            out_forward = self._forward(x)
            out_rev = self._forward(x_rev)
            return out_forward + out_rev
        else:
            return self._forward(x)


class DownPoolEncoder(nn.Module):
    def __init__(self, hidden_dim, pool=16):
        super(DownPoolEncoder, self).__init__()
        self.pool = pool

        self.linear1 = nn.Linear(pool, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        # self.out_proj = nn.Linear(64 * pool, out_channels)

    def forward(self, x, state=None):
        x = x.reshape(x.shape[0], -1, self.pool)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        # x = self.out_proj(x.flatten(1, -1))
        return x.squeeze(-1)


enc_registry = {
    'dummy': DummyEncoder,
    'linear': LinearEncoder,
    'layernorm': LayerNormLinearEncoder,
    'mlp': MLPEncoder,
    'embedding': EmbeddingEncoder,
    'transformer': SigEncoder,
    's4-encoder': S4Encoder,
    'pool': DownPoolEncoder,
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

    if encoder._name_ == 'transformer' or encoder._name_ == 's4-encoder' or encoder._name_ == 'pool':
        obj = instantiate(enc_registry, encoder)
        return obj

    in_features = dataset.d_data
    out_features = model.d_model

    if dataset.num_classes is not None:
        obj = instantiate(
            enc_registry,
            encoder,
            in_features=in_features,
            out_features=out_features,
            num_classes=dataset.num_classes
        )
    else:
        obj = instantiate(
            enc_registry,
            encoder,
            in_features=in_features,
            out_features=out_features
        )

    return obj
