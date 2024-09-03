import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from models.mamba_complex import MambaComplex
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    print('import error')
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from einops import rearrange
from models.sashimi.s4_standalone import LinearActivation, S4Block as S4


class PureMamba(nn.Module):
    def __init__(
            self,
            n_layers: int = 16,
            d_model: int = 64,
            is_complex: bool = True,

    ):
        super().__init__()
        self.is_complex = is_complex
        self.d_model = d_model
        self.n_layers = n_layers

        def mamba_block(dim, layer_idx):
            # make sure, the dt_rank is divisible by 2 which is required to make complex step function work
            dt_rank = math.ceil(dim / 16)
            if self.is_complex and dt_rank % 2 != 0:
                dt_rank += 1
            mixer_cls = partial(MambaComplex,
                                d_state=64,
                                d_conv=4,
                                expand=2,
                                layer_idx=layer_idx,
                                complex=self.is_complex,
                                dt_rank=dt_rank)
            norm_cls = partial(RMSNorm, eps=1e-5)
            mlp_cls = nn.Identity
            block = Block(
                dim,
                mixer_cls,
                mlp_cls,
                norm_cls=norm_cls,
                fused_add_norm=False,
                residual_in_fp32=False,
            )
            # not sure if we need this
            block.layer_idx = layer_idx
            return block

        self.layers = nn.ModuleList(
            [
                mamba_block(d_model, i)
                for i in range(n_layers)
            ]
        )

    def forward(self, hidden_states, state=None, inference_params=None, **mixer_kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        hidden_states = hidden_states + residual
        return hidden_states, None  # return output and 'state'

    def step(self, x, inference_params, state=None, **mixer_kwargs):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.forward(x, state=state, inference_params=inference_params, **mixer_kwargs)[0].squeeze(1), None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def default_state(self, device=None):
        return []
