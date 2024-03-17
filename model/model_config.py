from dataclasses import dataclass, field
from typing import Optional, Union

from fields.sdf_field import SDFNetConfig, SingleVarianceNetConfig
from fields.color_network import ColorNetConfig
from fields.refract_network import RefractNetConfig
from fields.nerf_density_field import NeRFConfig
from fields.collision_network import CollisionNetConfig

from model.render_config import RendererConfig

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the whole model."""
    
    sdf_network : SDFNetConfig = SDFNetConfig()
    deviation_network : SingleVarianceNetConfig = SingleVarianceNetConfig()
    color_network : ColorNetConfig = ColorNetConfig()
    nerf_network : NeRFConfig = NeRFConfig()
    refract_network : RefractNetConfig = RefractNetConfig()
    collision_network : CollisionNetConfig = CollisionNetConfig()
    
    renderer: RendererConfig = RendererConfig()
    
    
    # loss related
    igr_weight: float = 0.01
    """Weight for the igr / eikonal loss."""
    mask_weight : float = 0.5
    """Weight for the mask loss."""
    second_rgb_loss : float = 0.1
    """Weight for the  second rgb loss."""
    smoothed_loss: float = 0.1
    """Weight for the  smoothed_loss."""
    normal_reg_loss: float = 0.001
    """normal_reg_loss for the normal """

    # lr scheduling related
    lr: float = 5e-4
    """Learning rate."""
    lr_alpha: float = 0.05
    """Learning rate hp: alpha."""
    warm_up_end: int = 5_000
    """Number of steps to warm up."""
    end_iter: int = 1_000_00
    """Number of steps to train."""
    anneal_end: int = 50_000
    """Number of steps to anneal."""
    geometry_warmup_end: int = 20000
    """Number of steps to warm up geometry, during warm up, all hints are set to 0."""

    # chunk sizes
    batch_size: int = 512
    """Batch size."""
    shadow_mini_chunk_size: int = 2048
    """Mini chunk size for batched shadow hint calculation."""
    training_chunk_size: int = 512
    """Chunk size for training."""
    inference_chunk_size: int = 512
    """Chunk size for inference (testing)."""