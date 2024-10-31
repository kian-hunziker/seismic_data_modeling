import torch
import torch.nn as nn
from hydra_m.hydra_m import Hydra

import math
from functools import partial
from einops import rearrange


def LinearActivation(
        d_input, d_output, bias=True,
        transposed=False,
        activation=None,
        activate=False, # Apply activation as part of this module
        **kwargs,
    ):
    """Returns a linear nn.Module with control over axes order, initialization, and activation."""

    # Construct core module
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    if activation is not None and activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


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

        #x = F.pad(x[..., :-1], (1, 0))  # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        if skip is not None:
            x = x + skip
        if not self.transposed:
            x = x.transpose(1, 2)
        return x, None


class HydraBlock(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        d_state=64, 
        d_conv=7, 
        expand=2, 
        headdim=64, 
        use_mem_eff_path=False,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.mixer = Hydra(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            use_mem_eff_path=use_mem_eff_path,
            **kwargs
        )
        self.norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, state=None):
        residual = x
        # pre-norm
        out = self.norm(x.to(dtype=self.norm.weight.dtype))
        # apply Hydra layer
        out = self.mixer(out)
        # dropout
        out = self.dropout(out)
        # residual connection
        out = out + residual
        #out = self.norm(out.to(dtype=self.norm.weight.dtype))
        return out, None


class HydraUnet(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 5,
        pool: list = [4, 4],
        expand: int = 2,
        dropout: float = 0.0,
        skip_first_res=True,
        **hydra_args,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.pool = pool
        self.skip_first_res = skip_first_res

        hydra_args.update({'dropout': dropout})

        H = d_model
        try:
            d_head = hydra_args['headdim']
        except:
            print('headdim not in hydra args')
            d_head = d_model

        d_layers = []
        for p in pool:
            for i in range(n_layers):
                hydra_args.update({'headdim': d_head})
                d_layers.append(HydraBlock(hidden_size=H, **hydra_args))
            d_layers.append(DownPool(H, expand, p, transposed=False))
            H *= expand
            d_head *= expand

        c_layers = []
        for i in range(n_layers):
            c_layers.append(HydraBlock(hidden_size=H, **hydra_args))

        u_layers = []
        for p in pool[::-1]:
            block = []
            H //= expand
            d_head //= expand
            block.append(UpPool(H * expand, expand, p, transposed=False))

            for i in range(n_layers):
                hydra_args.update({'headdim': d_head})
                block.append(HydraBlock(hidden_size=H, **hydra_args))

            u_layers.append(nn.ModuleList(block))
            
        self.d_layers = nn.ModuleList(d_layers)
        self.c_layers = nn.ModuleList(c_layers)
        self.u_layers = nn.ModuleList(u_layers)
        self.norm = nn.LayerNorm(H)

    def forward(self, x, state=None):
        outputs = []
        if self.skip_first_res:
            outputs.append(torch.zeros_like(x))
        else:
            outputs.append(x)

        for layer in self.d_layers:
            x, _ = layer(x)
            outputs.append(x)

        for layer in self.c_layers:
            x, _ = layer(x)

        x = x + outputs.pop()

        for block in self.u_layers:
            for layer in block:
                x, _ = layer(x)
                x = x + outputs.pop()

        x = self.norm(x)
        
        return x, None
        