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
    def __init__(self, d_input, expand, pool, transposed=True):
        """
        if transposed is True, the input is [batch_size, H, seq_len]
        else: [batch_size, seq_len, H]
        """
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input * pool,
            self.d_output,
            transposed=transposed,
        )

    def forward(self, x):
        if self.transposed:
            x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        else:
            x = rearrange(x, '... (l s) h -> ... l (h s)', s=self.pool)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.pool:
            squeeze_idx = -1 if self.transposed else 1

            x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            x = x.unsqueeze(squeeze_idx)
            x = self.linear(x)
            x = x.squeeze(squeeze_idx)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []


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


class FFBlock(nn.Module):

    def __init__(self, d_model, expand=2, dropout=0.0, transposed=True):
        """
        Feed-forward block.

        Args:
            d_model: dimension of input
            expand: expansion factor for inverted bottleneck
            dropout: dropout rate
            transposed: if True, the input is [B, H, L]
        """
        super().__init__()
        self.transposed = transposed

        input_linear = LinearActivation(
            d_model,
            d_model * expand,
            transposed=transposed,
            activation='gelu',
            activate=True,
        )
        dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        output_linear = LinearActivation(
            d_model * expand,
            d_model,
            transposed=transposed,
            activation=None,
            activate=False,
        )

        self.ff = nn.Sequential(
            input_linear,
            dropout,
            output_linear,
        )

    def forward(self, x):
        return self.ff(x), None

    def default_state(self, *args, **kwargs):
        return None

    def step(self, x, state, **kwargs):
        # expects: (B, D, L)
        # print('ff', x.shape)
        if self.transposed:
            return self.ff(x.unsqueeze(-1)).squeeze(-1), state
        else:
            return self.ff(x.unsqueeze(1)).squeeze(1), state


class ResidualBlock(nn.Module):

    def __init__(
            self,
            d_model,
            layer,
            dropout=0.0,
            transposed=True
    ):
        """
        Residual S4 block.

        Args:
            d_model: dimension of the model
            layer: a layer config
            dropout: dropout rate
            transposed: if True, input is [B, H, L], else [B, L, H]
        """
        super().__init__()

        self.layer = layer
        self.norm = nn.LayerNorm(d_model)
        self.transposed = transposed
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        """
        if transposed is True, input x is shape (B, d_input, L)
        if transposed is False, input x is shape (B, L, d_input)
        """
        z = x

        # Prenorm
        if self.transposed:
            z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        else:
            z = self.norm(z)

        # Apply layer: we ignore the state input and output for training
        z, _ = self.layer(z)

        # Dropout on the output of the layer
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x, None

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x
        # Prenorm
        z = self.norm(z)
        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)
        # Residual connection
        x = z + x
        return x, state


'''
# Dummy Block for testing with double precision
class Block(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, hidden_states, residual = None, inference_params=None, **mixer_kwargs
    ):
        return hidden_states, 0.5*hidden_states
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
'''


class HybridSashimi(nn.Module):
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
            is_complex=False,
            d_conv=4,
            outermost_s4=False,
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
        self.d_conv = d_conv
        self.unet = unet
        self.is_complex = is_complex
        self.outermost_s4 = outermost_s4
        self.n_layers = n_layers

        def mamba_block(dim, layer_idx):
            # return Block() #just for dummy testing with double precision

            # make sure, the dt_rank is divisible by 2 which is required to make complex step function work
            dt_rank = math.ceil(dim / 16)
            if self.is_complex and dt_rank % 2 != 0:
                dt_rank += 1

            mixer_cls = partial(
                MambaComplex,
                d_state=64,
                d_conv=self.d_conv,
                expand=2,
                layer_idx=layer_idx,
                is_complex=self.is_complex,
                dropout=dropout,
                dt_rank=dt_rank,
            )

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
            block.layer_idx = layer_idx
            return block

        def s4_block(dim):
            layer = S4(
                d_model=dim,
                d_state=64,
                bidirectional=bidirectional,
                dropout=dropout,
                transposed=False,
                **s4_args,
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                transposed=False
            )

        def ff_block(dim):
            layer = FFBlock(
                d_model=dim,
                # TODO: expand=ff, doesnt work since we used ff in mamba block
                expand=2,
                dropout=dropout,
                transposed=False
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                transposed=False
            )

        layer_idx = 0

        # Down blocks
        d_layers = []
        for p in pool:
            if unet:
                # Add blocks in the down layers
                for i in range(n_layers):
                    if self.outermost_s4 and (i == 0 or i == n_layers - 1):
                        # make outer layers S4
                        d_layers.append(s4_block(H))
                        d_layers.append(ff_block(H))
                    else:
                        d_layers.append(mamba_block(H, layer_idx))
                    layer_idx += 1
                    # if ff > 0: d_layers.append(ff_block(H))

            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p, transposed=False))
            H *= expand

        # Center block
        c_layers = []
        for i in range(n_layers):
            if self.outermost_s4 and (i == 0 or i == n_layers - 1):
                # make outer layers S4
                c_layers.append(s4_block(H))
                c_layers.append(ff_block(H))
            else:
                c_layers.append(mamba_block(H, layer_idx))
            layer_idx += 1
            # activated per default
            # if ff > 0: c_layers.append(ff_block(H))

        # Up blocks
        u_layers = []
        for p in pool[::-1]:
            block = []
            H //= expand

            block.append(UpPool(H * expand, expand, p, transposed=False))

            for i in range(n_layers):
                if self.outermost_s4 and (i == 0 or i == n_layers - 1):
                    # make outer layers S4
                    block.append(s4_block(H))
                    block.append(ff_block(H))
                else:
                    block.append(mamba_block(H, layer_idx))

                layer_idx += 1
                # if ff > 0: block.append(ff_block(H))

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

        # Down blocks
        outputs = []
        outputs.append(x)
        residual = None
        for layer in self.d_layers:
            if not isinstance(layer, Block):
                if residual is not None:
                    x = x + residual
                x, _ = layer(x)
                outputs.append(x)
                residual = None
            else:
                # isinstance(layer, Block):
                x, residual = layer(x, residual, inference_params)
                outputs.append(x)

        if residual is not None:
            x = x + residual

        # Center block
        residual = None
        for i, layer in enumerate(self.c_layers):
            if self.outermost_s4 and i == len(self.c_layers) - 2:
                x = x + residual
            if not isinstance(layer, Block):
                x, _ = layer(x)
            else:
                x, residual = layer(x, residual, inference_params)

        if not self.outermost_s4:
            x = x + residual

        x = x + outputs.pop()  # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                residual = None
                for i, layer in enumerate(block):
                    if self.outermost_s4 and i == len(block) - 2:
                        x = x + residual

                    if not isinstance(layer, Block):
                        x, _ = layer(x)
                        x = x + outputs.pop()

                    if isinstance(layer, Block):
                        x, residual = layer(x, residual, inference_params)
                        x = x + outputs.pop()

                if not self.outermost_s4:
                    x = x + residual
            else:
                # not unet
                residual = None
                for i, layer in enumerate(block):
                    if self.outermost_s4 and i == len(block) - 2:
                        x = x + residual
                    if not isinstance(layer, Block):
                        x, _ = layer(x)
                        if isinstance(layer, UpPool):
                            x = x + outputs.pop()
                            outputs.append(x)
                    if isinstance(layer, Block):
                        x, residual = layer(x, residual, inference_params)

                if not self.outermost_s4:
                    x = x + residual

                x = x + outputs.pop()  # add a skip connection from the input of the modeling part of this up block

        # feature projection
        x = self.norm(x)
        return x, None  # required to return a state

    def default_state(self, *args, **kwargs):
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        states = []
        for layer in layers:
            if isinstance(layer, Block):
                states.append([])
            else:
                states.append(layer.default_state(*args, **kwargs))
        return states

    def allocate_inference_cache(self, batch_size, max_seqlen):
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        return [layer.allocate_inference_cach(batch_size, max_seqlen) for layer in layers if isinstance(layer, Block)]

    def setup_rnn(self, mode='dense'):
        """
        Convert the SaShiMi model to a RNN for autoregressive generation.

        Args:
            mode: S4 recurrence mode. Using `diagonal` can speed up generation by 10-20%.
                `linear` should be faster theoretically but is slow in practice since it
                dispatches more operations (could benefit from fused operations).
                Note that `diagonal` could potentially be unstable if the diagonalization is numerically unstable
                (although we haven't encountered this case in practice), while `dense` should always be stable.
        """
        assert mode in ['dense', 'diagonal', 'linear']
        for module in self.modules():
            if hasattr(module, '_setup_step'): module._setup_step()  # module._setup_step(mode=mode)

    def step(self, x, state, inference_params, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = []  # Store all layers for SaShiMi
        next_state = []
        residual = None

        for i, layer in enumerate(self.d_layers):
            if not isinstance(layer, Block):
                outputs.append(x)
                if residual is not None:
                    x = x + residual.squeeze(1)

                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
                residual = None
            if isinstance(layer, Block):
                outputs.append(x)
                x = x.unsqueeze(1)
                state.pop()
                x, residual = layer(x, residual, inference_params)
                next_state.append([])
                x = x.squeeze(1)

            if x is None: break
        if residual is not None:
            x = x + residual.squeeze(1)

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            if self.unet:
                for i in range(skipped):
                    next_state.append(state.pop())
                # TODO: double check this! used to be skipped // 3 but didnt work for 5 layer unet
                u_layers = list(self.u_layers)[skipped // (len(self.u_layers[0]) - 1):]
            else:
                for i in range(skipped):
                    for _ in range(len(self.u_layers[i])):
                        next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            residual = None
            for i, layer in enumerate(self.c_layers):
                if self.outermost_s4 and i == len(self.c_layers) - 2:
                    x = x + residual
                    x = x.squeeze(1)
                if not isinstance(layer, Block):
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                else:
                    if x.dim() == 2:
                        x = x.unsqueeze(1)
                    state.pop()
                    x, residual = layer(x, residual, inference_params)
                    next_state.append([])

            if not self.outermost_s4:
                x = x + residual
                x = x.squeeze(1)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for block in u_layers:
            if self.unet:
                residual = None
                for i, layer in enumerate(block):
                    if self.outermost_s4 and i == len(block) - 2:
                        x = x + residual.squeeze(1)
                    if isinstance(layer, Block):
                        if x.dim() == 2:
                            x = x.unsqueeze(1)
                        state.pop()
                        x, residual = layer(x, residual, inference_params)
                        next_state.append([])
                        x = x.squeeze(1) + outputs.pop()
                    else:
                        x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                        next_state.append(_next_state)
                        x = x + outputs.pop()

                if not self.outermost_s4:
                    x = x + residual.squeeze(1)
            else:
                residual = None
                for i, layer in enumerate(block):
                    if self.outermost_s4 and i == len(block) - 2:
                        x = x + residual
                        x = x.squeeze(1)
                    if isinstance(layer, Block):
                        if x.dim() == 2:
                            x = x.unsqueeze(1)
                        state.pop()
                        x, residual = layer(x, residual, inference_params)
                        next_state.append([])

                    else:
                        x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                        next_state.append(_next_state)
                        if isinstance(layer, UpPool):
                            x = x + outputs.pop()
                            outputs.append(x)

                if not self.outermost_s4:
                    x = x + residual
                    x = x.squeeze(1)
                x = x + outputs.pop()

        # feature projection
        x = self.norm(x)
        return x, next_state
