# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    print('no causal_conv_1d')
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MambaComplex(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            is_complex=False,
            dropout=0.0,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.is_complex = is_complex

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        if self.is_complex:
            projection_dim = 4
        else:
            projection_dim = 2
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * projection_dim, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        if self.is_complex:
            # S4D lin initialization for complex case
            A_real = repeat(
                torch.ones(self.d_state, dtype=torch.float32, device=device) * 0.5,
                "n -> d n",
                d=self.d_inner
            ).contiguous()
            # TODO: double check initialization. In the mamba paper they say the imaginary part is i * n
            # but in the paper they reference (on the parametrization and initialization of DSS) it is i * pi * n
            A_imag = repeat(
                torch.arange(0, self.d_state, dtype=torch.float32, device=device) * torch.pi,
                "n -> d n",
                d=self.d_inner
            ).contiguous()
            A_log_real = torch.log(A_real)
            self.A_log_real = nn.Parameter(A_log_real)
            self.A_log_real._no_weight_decay = True
            self.A_imag = nn.Parameter(A_imag)
        else:
            # S4D real initialization
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_log = torch.log(A)  # Keep A_log in fp32
            self.A_log = nn.Parameter(A_log)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        At inference time (with inference_params is not None and inference_params.seqlen_offset > 0),
        the step function is called. The hidden states must then have dimension [B, 1, D] / [batch_size, 1, d_state].

        inference_params are updated inplace.
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        if self.is_complex:
            # COMPLEX:
            A = -torch.exp(self.A_log_real.float()) + 1j * self.A_imag.float()
        else:
            # REAL:
            A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None and self.d_conv <= 4:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None or self.d_conv > 4:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)

            if self.is_complex:
                dt, B, C = torch.split(x_dbl, [self.dt_rank, 2 * self.d_state, 2 * self.d_state], dim=-1)
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=seqlen, two=2).contiguous()
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=seqlen, two=2).contiguous()
            else:
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")

            out = self.out_proj(y)

            # dropout only in forward, as step function is only used during inference
            out = self.dropout(out)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """
        Compute step
        :param hidden_states: input of shape [batch_size, 1, d_state]
        :param conv_state: state of convolution
        :param ssm_state: state of ssm
        :return: output of shape [batch_size, 1, d_state]
        """
        # hidden states dim: [batch_size, 1, data_dim]
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None or self.d_conv > 4:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)

        if self.is_complex:
            dt, B, C = torch.split(x_db, [self.dt_rank, 2 * self.d_state, 2 * self.d_state], dim=-1)
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)

        if self.is_complex:
            # COMPLEX:
            A = -torch.exp(self.A_log_real.float()) + 1j * self.A_imag.float()
        else:
            # REAL:
            A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if self.is_complex or selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)

            if self.is_complex:
                comp_type = torch.complex64
            else:
                comp_type = dtype

            y = torch.einsum("bdn,bn->bd", ssm_state.to(comp_type), C)

            if y.is_complex():
                y = y.real * 2

            y = y + self.D.to(dtype) * x

            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        # ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        if self.is_complex:
            ssm_dtype = torch.complex64
        else:
            ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype

        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=torch.complex64 if self.is_complex else self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class MambaBidirectional(nn.Module):
    def __init__(self, d_model: int, mode: str = 'attention', **mamba_args):
        """
        Bidirectional Mamba module. The input is passed through the forward Mamba block and the
        reversed input is passed through the backward Mamba block. Mode controls how the two outputs
        are combined.
        :param d_model: d_model for both Mamba blocks
        :param mode: one of ['attention', 'linear', 'weight', 'scalar', 'sum', 'avg']. Default is 'attention'.
        :param mamba_args: Arguments passed to the Mamba blocks
        """
        super(MambaBidirectional, self).__init__()
        self.mode = mode
        self.mamba_forward = MambaComplex(d_model=d_model, **mamba_args)
        self.mamba_backward = MambaComplex(d_model=d_model, **mamba_args)

        if self.mode == 'attention':
            self.weight_projection = nn.Linear(2 * d_model, 1)
        elif self.mode == 'linear':
            self.linear = nn.Linear(2 * d_model, d_model)
        elif self.mode == 'weight':
            self.weight = nn.Parameter(torch.randn(1, 1, d_model))
        elif self.mode == 'scalar':
            self.weight = nn.Parameter(torch.tensor(0.5))
        elif self.mode == 'sum' or self.mode == 'avg':
            pass
        else:
            print(f'Unknown mode: {self.mode}')

    def forward(self, hidden_states, inference_params=None):
        rev = hidden_states.flip(dims=(1,))
        x_forward = self.mamba_forward(hidden_states)
        x_backward = self.mamba_backward(rev)

        if self.mode == 'attention':
            weights = torch.sigmoid(self.weight_projection(torch.cat((x_forward, x_backward), dim=-1)))
            combined_output = weights * x_forward + (1 - weights) * x_backward
        elif self.mode == 'linear':
            combined_output = self.linear(torch.cat((x_forward, x_backward), dim=-1))
        elif self.mode == 'weight' or self.mode == 'scalar':
            w = torch.sigmoid(self.weight)
            combined_output = w * x_forward + (1 - w) * x_backward
        elif self.mode == 'sum':
            combined_output = x_forward + x_backward
        elif self.mode == 'avg':
            combined_output = x_forward + x_backward / 2.0

        return combined_output
