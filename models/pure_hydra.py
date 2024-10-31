import torch
import torch.nn as nn
from hydra_m.hydra_m import Hydra


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
        return out


class PureHydra(nn.Module):
    def __init__(
        self, 
        d_model, 
        n_layers, 
        d_state=64,
        d_conv=7,
        expand=2,
        headdim=64,
        use_mem_eff_path=False,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.Sequential(
            *[
                HydraBlock(
                    hidden_size=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    use_mem_eff_path=use_mem_eff_path,
                    **kwargs
                ) for _ in range(n_layers)
            ]
        )

    def forward(self, x, state=None):
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]
        return self.layers(x), None # also return a 'state'
