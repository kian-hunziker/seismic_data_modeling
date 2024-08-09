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

class DownPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input * pool,
            self.d_output,
            transposed=True,
        )

    def forward(self, x):
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.pool:
            x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []


class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
        )

    def forward(self, x, skip=None):
        x = self.linear(x)

        x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        if skip is not None:
            x = x + skip
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            state = list(torch.unbind(x, dim=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device) # (batch, h, s)
        state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        return state

class MambaSashimi(nn.Module):
    def __init__(
        self,
        d_model=64,
        n_layers=8,
        pool=[4, 4],
        expand=2,
        ff=2,
        bidirectional=False,
        unet=False,
        dropout=0.0,
        complex=False,
        **s4_args,
    ):
        """
        SaShiMi model backbone.

        Args:
            d_model: dimension of the model. We generally use 64 for all our experiments.
            n_layers: number of (Residual (S4) --> Residual (FF)) blocks at each pooling level.
                We use 8 layers for our experiments, although we found that increasing layers even further generally
                improves performance at the expense of training / inference speed.
            pool: pooling factor at each level. Pooling shrinks the sequence length at lower levels.
                We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
                It's possible that a different combination of pooling factors and number of tiers may perform better.
            expand: expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the architecture.
                We generally found 2 to perform best (among 2, 4).
            ff: expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4).
            bidirectional: use bidirectional S4 layers. Bidirectional layers are suitable for use with non-causal models
                such as diffusion models like DiffWave.
            unet: use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling.
                All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
                We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
                for non-autoregressive models.
            dropout: dropout rate. Default to 0.0, since we haven't found settings where SaShiMi overfits.
        """
        super().__init__()
        self.d_model = H = d_model
        self.d_output = H
        self.unet = unet
        self.complex = complex

        def mamba_block(dim, layer_idx):
            # make sure, the dt_rank is divisible by 2 which is required to make complex step funciton work
            dt_rank = math.ceil(dim / 16)
            if self.complex and dt_rank % 2 != 0:
                dt_rank += 1
            mixer_cls = partial(MambaComplex, d_state=64, d_conv=4, expand=2, layer_idx=layer_idx, complex=self.complex, dt_rank=dt_rank)
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

        layer_idx = 0

        # Down blocks
        d_layers = []
        for p in pool:
            if unet:
                # Add blocks in the down layers
                for _ in range(n_layers):
                    d_layers.append(mamba_block(H, layer_idx))
                    layer_idx += 1
                    #if ff > 0: d_layers.append(ff_block(H))

            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p))
            H *= expand

        # Center block
        c_layers = []
        for _ in range(n_layers):
            c_layers.append(mamba_block(H, layer_idx))
            layer_idx += 1
            #if ff > 0: c_layers.append(ff_block(H))

        # Up blocks
        u_layers = []
        for p in pool[::-1]:
            block = []
            H //= expand
            block.append(UpPool(H * expand, expand, p))

            for _ in range(n_layers):
                block.append(mamba_block(H, layer_idx))
                layer_idx += 1
                #if ff > 0: block.append(ff_block(H))

            u_layers.append(nn.ModuleList(block))

        self.d_layers = nn.ModuleList(d_layers)
        self.c_layers = nn.ModuleList(c_layers)
        self.u_layers = nn.ModuleList(u_layers)
        self.norm = nn.LayerNorm(H)

        assert H == d_model

    def forward(self, x, state=None, inference_params=None):
        """
        input: (batch, length, d_input)
        output: (batch, length, d_output)
        """
        x = x.transpose(1, 2)

        # Down blocks
        outputs = []
        outputs.append(x)
        residual = None
        for layer in self.d_layers:
            if isinstance(layer, DownPool):
                x, _ = layer(x)
                outputs.append(x)
                residual = None
            if isinstance(layer, Block):
                x = x.transpose(1, 2)
                x, residual = layer(x, residual, inference_params)
                x = x.transpose(1, 2)
                outputs.append(x)

        # Center block
        x = x.transpose(1, 2)
        residual = None
        for layer in self.c_layers:
            x, residual = layer(x, residual, inference_params)
        

        x = x.transpose(1, 2)
        x = x + outputs.pop() # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                # TODO: add unet support
                #for layer in block:
                #    x, _ = layer(x)
                #    x = x + outputs.pop() # skip connection
                residual = None
                for layer in block:
                    if isinstance(layer, UpPool):
                        x, _ = layer(x)
                        x = x + outputs.pop()
                        
                    if isinstance(layer, Block):
                        x = x.transpose(1, 2)
                        x, residual = layer(x, residual, inference_params)
                        x = x.transpose(1, 2)
                        x = x + outputs.pop()
            else:
                # not unet
                residual = None
                for layer in block:
                    if isinstance(layer, UpPool):
                        x, _ = layer(x)
                        x = x + outputs.pop()
                        outputs.append(x)
                        x = x.transpose(1, 2)
                    if isinstance(layer, Block):
                        x, residual = layer(x, residual, inference_params)

                x = x.transpose(1, 2)
                x = x + outputs.pop() # add a skip connection from the input of the modeling part of this up block

        # feature projection
        x = x.transpose(1, 2) # (batch, length, expand)
        x = self.norm(x)

        return x, None # required to return a state

    def default_state(self, *args, **kwargs):
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        states = []
        for layer in layers:
            if isinstance(layer, Block):
                states.append([])
            else:
                states.append(layer.default_state(*args, **kwargs))
        return states
        #return [layer.default_state(*args, **kwargs) for layer in layers if not isinstance(layer, Block)]

    def allocate_inference_cache(self, batch_size, max_seqlen):
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        return [layer.allocate_inference_cach(batch_size, max_seqlen) for layer in layers if isinstance(layer, Block)]

    def step(self, x, state, inference_params, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = [] # Store all layers for SaShiMi
        next_state = []
        residual = None
        for layer in self.d_layers:
            if isinstance(layer, DownPool):
                outputs.append(x)
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
                residual = None
            if isinstance(layer, Block):
                outputs.append(x)    
                x = x.unsqueeze(0)
                state.pop()
                x, residual = layer(x, residual, inference_params)
                next_state.append([])
                x = x.squeeze(0)
            #outputs.append(x)
            #x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            #next_state.append(_next_state)
            if x is None: break

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            if self.unet:
                # TODO: add unet support
                for i in range(skipped):
                    next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped//3:]
            else:
                for i in range(skipped):
                    for _ in range(len(self.u_layers[i])):
                        next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            residual = None
            for layer in self.c_layers:
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                state.pop()
                x, residual = layer(x, residual, inference_params)
                next_state.append([])
            x = x.squeeze(0)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for block in u_layers:
            if self.unet:

                residual = None
                for layer in block:
                    if isinstance(layer, Block):
                        if x.dim() == 2:
                            x = x.unsqueeze(0)
                        state.pop()
                        x, residual = layer(x, residual, inference_params)
                        next_state.append([])
                        x = x.squeeze(0) + outputs.pop()
                    if isinstance(layer, UpPool):
                        x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                        next_state.append(_next_state)
                        x = x + outputs.pop()
            else:
                residual = None
                for layer in block:
                    if isinstance(layer, Block):
                        if x.dim() == 2:
                            x = x.unsqueeze(0)
                        state.pop()
                        x, residual = layer(x, residual, inference_params)
                        next_state.append([])
                    
                    if isinstance(layer, UpPool):
                        x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                        next_state.append(_next_state)
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x.squeeze(0)
                x = x + outputs.pop()

        # feature projection
        x = self.norm(x)
        return x, next_state
                