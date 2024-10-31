import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config_utils import instantiate
from dataloaders.base import SequenceDataset
from tasks.positional_encoding import PositionalEncoding

from models.sashimi.s4_standalone import LinearActivation, S4Block as S4
from models.sashimi.sashimi_standalone import UpPool, FFBlock, ResidualBlock
from einops import rearrange

from omegaconf import OmegaConf


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
    def __init__(self, in_features, out_features, num_classes=None):
        super().__init__(in_features, out_features)
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight._no_weight_decay = True

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


class S4Decoder(nn.Module):
    def __init__(self, d_model, n_blocks, bidirectional=False, add_mean=False):
        super(S4Decoder, self).__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.bidirectional = bidirectional
        self.add_mean = add_mean

        # if self.add_mean:
        #    self.in_proj = nn.Linear(in_features=d_model, out_features=d_model)

        self.conv_t = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=16,
            stride=16,
            padding=0,
        )
        self.out_proj = nn.Linear(d_model, 1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(s4_block(dim=d_model))
            blocks.append(ff_block(dim=d_model))
        self.blocks = nn.ModuleList(blocks)

    def _forward(self, x, state=None):
        if x.dim == 3:
            x = x.squeeze(-1)
        if self.add_mean:
            means = x[:, 0]
            # x = self.in_proj(x)
            x[:, 1:] = x[:, 1:] + means.unsqueeze(-1)
        x = self.conv_t(x.unsqueeze(1))
        for block in self.blocks:
            x, _ = block(x)
        x = self.out_proj(x.transpose(1, 2))
        # if self.add_mean:
        #    x = x + means.unsqueeze(-1).unsqueeze(-1)
        return x

    def forward(self, x, state=None):
        if self.bidirectional:
            x_rev = torch.flip(x, dims=[1])
            out_forward = self._forward(x)
            out_rev = self._forward(x_rev)
            return out_forward + out_rev
        else:
            return self._forward(x)


class UpPoolDecoder(nn.Module):
    def __init__(self, d_model, pool):
        super(UpPoolDecoder, self).__init__()
        self.d_model = d_model
        self.pool = pool
        self.conv_t = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=self.pool,
            stride=self.pool,
            padding=0,
        )
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x, state=None):
        if x.dim == 3:
            x = x.squeeze(-1)
        x = self.conv_t(x.unsqueeze(1))
        x = self.out_proj(x.transpose(1, 2))
        return x


class EmbeddingDecoder(nn.Module):
    def __init__(self, num_classes, output_dim, d_model=64, n_blocks=0):
        super(EmbeddingDecoder, self).__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_blocks = n_blocks

        self.embedding = nn.Embedding(self.num_classes, self.output_dim)

        self.conv_t = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=16,
            stride=16,
            padding=0,
        )
        self.out_proj = nn.Linear(d_model, 1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(s4_block(dim=d_model))
            blocks.append(ff_block(dim=d_model))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, state=None):
        if self.n_blocks == 0:
            x = self.embedding(x).transpose(1, 2)
            return x
        else:
            x = self.embedding(x)
            x = self.conv_t(x)
            for block in self.blocks:
                x, _ = block(x)
            x = self.out_proj(x.transpose(1, 2))

        return x


class PhasePickDecoder(nn.Module):
    def __init__(self, d_model, output_dim=3, convolutional=False, kernel_size=33, dropout=0.0):
        """
        Decoder for phase picking tasks
        :param d_model: dimension of model backbone
        :param output_dim: dimension of output, for phase picking this is usually 3
        :param convolutional: If true, use a Conv1d with kernel size 'kernel_size'. Else, use a linear layer.
        :param kernel_size: Size of the convolutional kernel. Must be uneven!
        :param dropout: input dropout rate
        """
        super(PhasePickDecoder, self).__init__()
        self.convolutional = convolutional

        if self.convolutional:
            assert kernel_size % 2 == 1, 'Kernel size must be uneven'

            self.conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=output_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=int(kernel_size // 2),
            )
        else:
            self.linear = nn.Linear(d_model, output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, state=None):
        x = self.dropout(x)
        if self.convolutional:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)
        else:
            return self.linear(x)


class SequenceClassifier(nn.Module):
    def __init__(self, in_features, out_features, num_classes, mode='avg'):
        super(SequenceClassifier, self).__init__()
        self.mode = mode
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x, state=None):
        if self.mode == 'avg':
            x = torch.mean(x, dim=1)
            x = self.linear(x)
        elif self.mode == 'last':
            x = x[:, -1, :]
            x = self.linear(x)
        else:
            print(f'Unknown mode: {self.mode}')
        return x


class CausalDecoder(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=16):
        super(CausalDecoder, self).__init__()
        self.kernel_size = kernel_size

        # Define 1D convolution with 1 input channel, 1 output channel, and kernel_size
        self.conv = nn.Conv1d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=kernel_size - 1,
            bias=False
        )

    def forward(self, x, state=None):
        # x: [B, L, D_in] then output: [B, L, D_out] (for training)
        # at inference time: x is [B, D_in], state [B, L, D_in], output [B, D_out]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add the channel dimension: [batch_size, 1, seq_len]

        if state is not None:
            x = torch.cat((state, x), dim=1)

        # Apply the causal convolution
        x = x.transpose(1, 2)
        out = self.conv(x)[:, :, :-(self.kernel_size - 1)]
        out = out.transpose(1, 2)

        if state is not None:
            out = out[:, -1, :]
        # Remove the extra channel dimension: [batch_size, seq_len]
        return out.squeeze(1)


class ConvNetDecoder(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=5, dim=128):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=dim,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, x, state=None):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x.transpose(1, 2)


class Conv1dUpsampling(nn.Module):
    def __init__(self, hidden_dim: int, reduce_time_layers: int = 2):
        super(Conv1dUpsampling, self).__init__()

        # Upsample only in the time dimension, increase time dimensions of the hidden_states tensor
        layers = []
        for _ in range(reduce_time_layers):
            layers.extend([
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GELU()
            ])
        self.time_upsample = nn.Sequential(*layers)

        # Reduce the potential effects of padded artifacts introduced by the upsampling
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, dim, time)
        x = self.time_upsample(x)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # Revert shape to (batch_size, time, dim)
        return x


class BidirAutoregDecoder(nn.Module):
    def __init__(self, in_features, out_features, upsample=False):
        super(BidirAutoregDecoder, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upSample = Conv1dUpsampling(hidden_dim=in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, state=None):
        x_ntk, x_ptk, x_tokens = x
        if self.upsample:
            x_tokens = None
            x_ntk = self.upSample(x_ntk)
            x_ptk = self.upSample(x_ptk)
        out_ntk = self.linear(x_ntk)
        out_ptk = self.linear(x_ptk)
        return (out_ntk, out_ptk, x_tokens)


class BidirPhasePickDecoder(nn.Module):
    def __init__(self, in_features, out_features, upsample=True, kernel_size=33):
        super(BidirPhasePickDecoder, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upSample = Conv1dUpsampling(hidden_dim=in_features)

        self.linear = nn.Linear(in_features, in_features)
        self.conv_1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.out_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=1,
            padding=int(kernel_size // 2),
        )

    def forward(self, x, state=None):
        x_ntk, x_ptk, _ = x
        if self.upsample:
            # x_ntk = self.upSample(x_ntk)
            out = self.upSample(x_ptk)
        # out = torch.cat([x_ntk, x_ptk], dim=-1)
        out = F.gelu(self.conv_1(out.transpose(1, 2)))
        out = self.out_conv(out).transpose(1, 2)[:, 4:-4, :]
        return out


class BidirPhasePickDecoderSmall(nn.Module):
    def __init__(self, in_features, out_features, upsample=True, kernel_size=33):
        super(BidirPhasePickDecoderSmall, self).__init__()
        self.upsample = upsample

        self.out_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=1,
            padding=int(kernel_size // 2),
        )
        if self.upsample:
            self.upSample = Conv1dUpsampling(hidden_dim=out_features)
        self.out_proj = nn.Conv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, state=None):
        if len(x) == 2:
            x_ptk = x[0]
        elif len(x) == 3:
            x_ptk = x[1]
        else:
            x_ptk = x
        out = self.out_conv(x_ptk.transpose(1, 2)).transpose(1, 2)
        if self.upsample:
            out = self.upSample(out)
        out = self.out_proj(out.transpose(1, 2)).transpose(1, 2)
        return out


class UpsamplingDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpsamplingDecoder, self).__init__()
        self.upSample = Conv1dUpsampling(hidden_dim=in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, state=None):
        out = self.upSample(x)
        out = self.linear(out)
        return out


class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool, transposed=True):
        """
        if transposed is True, the input is [batch_size, H, seq_len]
        else: [batch_size, seq_len, H]
        """
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
        )

    def forward(self, x, skip=None):
        if not self.transposed:
            x = x.transpose(1, 2)
        x = self.linear(x)

        x = F.pad(x[..., :-1], (1, 0))  # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        if skip is not None:
            x = x + skip
        if not self.transposed:
            x = x.transpose(1, 2)
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            squeeze_idx = -1 if self.transposed else 1
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            state = list(torch.unbind(x, dim=-1))
        else:
            assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device)  # (batch, h, s)
        state = list(torch.unbind(state, dim=-1))  # List of (..., H)
        return state


class CausalUpsamplingDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(CausalUpsamplingDecoder, self).__init__()
        self.upSample = UpPool(
            d_input=in_features,
            expand=4,
            pool=4,
            transposed=False,
        )
        self.linear = nn.Linear(in_features // 4, out_features)

    def forward(self, x, state=None):
        x, _ = self.upSample(x)
        x = self.linear(x)
        return x


class CausalBidirAutoregDecoder(nn.Module):
    def __init__(self, in_features, out_features, upsample=False):
        super(CausalBidirAutoregDecoder, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upPool = UpPool(
                d_input=in_features,
                expand=4,
                pool=4,
                transposed=False,
            )
            self.linear = nn.Linear(in_features // 4, out_features)
        else:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, state=None):
        x_ntk, x_ptk, x_tokens = x
        if self.upsample:
            x_tokens = None
            x_ntk, _ = self.upPool(x_ntk)
            x_ptk, _ = self.upPool(x_ptk)
        out_ntk = self.linear(x_ntk)
        out_ptk = self.linear(x_ptk)
        return (out_ntk, out_ptk, x_tokens)


class SanityCheckPhasePicker(nn.Module):
    def __init__(self, in_features, out_features, upsample: bool = False, output_len: int = 4096):
        super().__init__()
        self.upsample = upsample
        self.output_len = output_len

        self.linear1 = nn.Linear(in_features, 4 * out_features)
        self.linear2 = nn.Linear(4 * out_features, 4 * out_features)
        self.linear3 = nn.Linear(4 * out_features, out_features)

        self.conv = nn.Conv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=7,
            stride=1,
            padding=3
        )

    def forward(self, x, state=None):
        if isinstance(x, tuple) or isinstance(x, list):
            if len(x) == 3:
                x = x[1]
            elif len(x) == 2:
                x = x[0]
        seq_len = x.shape[1]

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = x.transpose(1, 2)
        if self.upsample:
            x = F.interpolate(x, size=4 * seq_len, mode='linear')
        x = self.conv(x).transpose(1, 2)
        len_diff = x.shape[1] - self.output_len
        if len_diff > 0:
            x = x[:, len_diff // 2: - len_diff // 2, :]
        return x


dec_registry = {
    'dummy': DummyDecoder,
    'linear': LinearDecoder,
    'transformer': SigDecoder,
    's4-decoder': S4Decoder,
    'pool': UpPoolDecoder,
    'embedding': EmbeddingDecoder,
    'phase-pick': PhasePickDecoder,
    'sequence-classifier': SequenceClassifier,
    'causal-decoder': CausalDecoder,
    'convnet-decoder': ConvNetDecoder,
    'bidir-autoreg-decoder': BidirAutoregDecoder,
    'bidir-phasepick-decoder': BidirPhasePickDecoder,
    'bidir-phasepick-decoder-small': BidirPhasePickDecoderSmall,
    'upsampling-decoder': UpsamplingDecoder,
    'causal-upsampling-decoder': CausalUpsamplingDecoder,
    'causal-bidir-autoreg-decoder': CausalBidirAutoregDecoder,
    'sanity-check-decoder': SanityCheckPhasePicker,
}

pretrain_decoders = ['transformer', 's4-decoder', 'pool', 'embedding']
phasepick_decoders = ['phase-pick']


def instantiate_decoder(decoder, dataset: SequenceDataset = None, model: nn.Module = None):
    if decoder is None:
        return None

    if decoder._name_ in pretrain_decoders:
        obj = instantiate(dec_registry, decoder)
        return obj

    if dataset is None:
        print('Please specify dataset to instantiate encoder')
        return None

    if model is None:
        print('Please specify model to instantiate encoder')
        return None

    in_features = model.d_model
    if dataset.num_classes is not None:
        out_features = dataset.num_classes
    else:
        out_features = dataset.d_data

    if decoder._name_ in phasepick_decoders:
        obj = instantiate(dec_registry, decoder, d_model=in_features)
        return obj

    obj = instantiate(dec_registry, decoder, in_features=in_features, out_features=out_features)

    return obj


def instantiate_decoder_simple(decoder, d_data, d_model):
    if decoder._name_ in phasepick_decoders:
        obj = instantiate(dec_registry, decoder, d_model=d_model)
        return obj
    obj = instantiate(dec_registry, decoder, in_features=d_model, out_features=d_data)
    return obj


def load_decoder_from_file(decoder_file, dataset: SequenceDataset = None, model=None):
    decoder_state_dict, hparams = torch.load(decoder_file, weights_only=False)
    dec_config = OmegaConf.create(hparams['decoder'])
    decoder = instantiate_decoder(dec_config, dataset=dataset, model=model)
    decoder.load_state_dict(decoder_state_dict)

    # freeze parameters
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    return decoder, hparams
