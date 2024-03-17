from dataclasses import dataclass

import torch
from torch import nn

from fields.encodings import NeRFEncoding


@dataclass(frozen=True)
class CollisionNetConfig:
    """Configuration for the Color network."""

    d_hidden: int = 256
    """Number of hidden units in the MLP."""
    n_layers: int = 4
    """Number of hidden layers in the MLP."""
    weight_norm: bool = True
    """Whether to use weight normalization."""
    multi_res: int = 4
    """Number of frequencies to use in the input embedding."""
    squeeze_out: bool = True
    """Whether to squeeze the output to [0, 1]."""


class CollisionNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 config: CollisionNetConfig,
                 ):
        super().__init__()

        self.squeeze_out = config.squeeze_out
        
        dims = [d_in] + [config.d_hidden for _ in range(config.n_layers)] + [d_out]

        self.embed_view_fn = NeRFEncoding(
            in_dim=3,
            num_frequencies=config.multi_res,
            include_input=True
        )
        dims[0] += (self.embed_view_fn.get_out_dim() - 3) * 2 

        self.num_layers = len(dims)

        for layer_idx in range(0, self.num_layers - 1):
            out_dim = dims[layer_idx + 1]
            lin = nn.Linear(dims[layer_idx], out_dim)

            if config.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(layer_idx), lin)

        self.relu = nn.ReLU()

    def forward(self, view_o , view_dirs):
        if self.embed_view_fn is not None:
            view_dirs = self.embed_view_fn(view_dirs)
            view_o = self.embed_view_fn(view_o)

        rendering_input = [view_o, view_dirs]


        x = torch.cat(rendering_input, dim=-1)

        for layer_idx in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer_idx))

            x = lin(x)

            if layer_idx < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x
