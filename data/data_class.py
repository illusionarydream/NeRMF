from dataclasses import dataclass
from enum import Enum
from typing import Optional , Dict

import numpy as np
import torch
from jaxtyping import Float, Int

from utils.tensor_dataclass import TensorDataclass


# RawPixelBundle
@dataclass
class RawPixelBundle(TensorDataclass):
    img_indices: Optional[Int[torch.Tensor, "*bs 1"]]
    h_indices: Float[torch.Tensor, "*bs 1"]
    w_indices: Float[torch.Tensor, "*bs 1"]
    poses: Float[torch.Tensor, "*bs 4 4"]
    mask1s: Float[torch.Tensor, "*bs 3"]
    mask2s: Float[torch.Tensor, "*bs 3"]
    rgb_gt: Optional[Float[torch.Tensor, "*bs 3"]]

    _field_custom_dimensions = {"poses": 2}
    
# RayBundle from nerf_studio
# below is a part copy of the RayBundle from ./camera/ray_utils.py

#@dataclass
#class RayBundle(TensorDataclass):

    
# RenderOutput dataclass
@dataclass
class RenderOutput(TensorDataclass):
    rgb: Float[torch.Tensor, "bs 3"]
    #depth: Float[torch.Tensor, "bs 1"]
    weights: Float[torch.Tensor, "bs n_samples"]
    #s_val: Float[torch.Tensor, "bs 1"]
    analytic_normals: Float[torch.Tensor, "bs n_samples 3"]
    #normalized_analytic_normals: Float[torch.Tensor, "bs n_samples 3"]
    mask1: Float[torch.Tensor, "bs 3"]
    mask2: Float[torch.Tensor, "bs 3"]
    rgb_neus: Float[torch.Tensor, "bs 3"]
    smoothed_normals: Optional[Float[torch.Tensor, "bs n_samples 3"]]
    

    _field_custom_dimensions = {
        "analytic_normals": 2,
        "normalized_analytic_normals": 2,
        "smoothed_normals": 2,
        "dirs": 2,
    }

