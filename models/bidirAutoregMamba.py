import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from models.mamba_complex import MambaComplex, MambaBidirectional
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    print('import error')
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from einops import rearrange
from models.sashimi.s4_standalone import LinearActivation


class dummyBlock(nn.Module):
    def __init__(self, dim):
        super(dummyBlock, self).__init__()
        self.dim = dim

    def forward(self, x, residual):
        return x, x


class BidirAutoregMamba(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_layers: int,
                 dropout: float = 0.0,
                 expand=2,
                 ff=2,
                 is_complex=False,
                 d_conv=4,
                 d_state=64,
                 ):
        super(BidirAutoregMamba, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.d_conv = d_conv
        self.is_complex = is_complex

        self.sos = torch.nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.sos)
        self.eos = torch.nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.eos)

        def mamba_block(dim):
            # make sure, the dt_rank is divisible by 2 which is required to make complex step function work
            dt_rank = math.ceil(dim / 16)
            if self.is_complex and dt_rank % 2 != 0:
                dt_rank += 1

            mamba_args = {
                'd_conv': self.d_conv,
                'expand': 2,
                'is_complex': self.is_complex,
                'dropout': self.dropout,
                'dt_rank': dt_rank
            }

            mamba_args['d_state'] = d_state

            mixer_cls = partial(
                MambaComplex,
                **mamba_args
            )

            #mixer_cls = nn.Identity
            norm_cls = partial(RMSNorm, eps=1e-5)
            if ff == 0:
                mlp_cls = nn.Identity
            else:
                mlp_cls = partial(
                    GatedMLP, hidden_features=ff * d_model, bias=True,
                )
            block = Block(
                dim,
                mixer_cls,
                mlp_cls,
                norm_cls=norm_cls,
                fused_add_norm=False,
                residual_in_fp32=False,
            )
            # not sure if we need this
            return block


        self.blocks = nn.ModuleList([mamba_block(self.d_model) for _ in range(self.n_layers)])

        self.ln_f = nn.LayerNorm(self.d_model)

    def forward(self, hidden_states, state=None):
        if isinstance(hidden_states, tuple):
            hidden_states, x_tokens = hidden_states
        else:
            x_tokens = None
        batch_size, seq_len, dim = hidden_states.shape

        # Add the SOS and EOS token to the input sequence
        sos_token = self.sos.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape [batch_size, 1, d_model]
        # Add the SOS token for shifted right,  so the sequence length will be seq_len + 1
        hidden_states = torch.cat([sos_token, hidden_states], dim=1)
        eos_token = self.eos.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape [batch_size, 1, d_model]
        # Add the SOS token for shifted left,  so the sequence length will be seq_len + 2
        hidden_states = torch.cat([hidden_states, eos_token], dim=1)

        residual = None

        for l, block in enumerate(self.blocks):
            if l > 0:
                # reverse the sequence of hidden states
                hidden_states = torch.flip(hidden_states, [1])
                residual = torch.flip(residual, [1])

            # print('hidden_states', hidden_states[0, :10, 0])
            # if residual is not None:
            #    print('residual', residual[0, :10, 0])
            hidden_states, residual = block(hidden_states, residual)

            if (
                    l + 1) == self.n_layers - 1:  # use the hidden states from the second last layer for next token prediction
                #print('hidden_states_ntp', hidden_states[0, :10, 0])
                # if residual is not None:
                #print('residual', residual[0, :10, 0])
                hidden_states_NTP = hidden_states + residual
            elif (l + 1) == self.n_layers:  # use the hidden states from the last layer for previous token prediction
                #print('hidden_states_ptk', hidden_states[0, :10, 0])
                # if residual is not None:
                #print('residual', residual[0, :10, 0])
                hidden_states_PTP = hidden_states + residual

        X_NTK = self.ln_f(hidden_states_NTP)
        X_PTK = self.ln_f(hidden_states_PTP)

        return (X_NTK, X_PTK, x_tokens), None
