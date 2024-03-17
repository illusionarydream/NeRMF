from dataclasses import dataclass
from typing import Optional
from enum import Enum

class PixelSamplingStrategy(Enum):
    """
    Strategy for sampling batch of pixels from images.
    """

    ALL_IMAGES = 'all_images'
    """Pixels in a batch are sampled from different images, like NeRF."""
    SAME_IMAGE = 'same_image'
    """Pixels in a batch are sampled from the same image, like NeuS."""

@dataclass(frozen=True)
class DataManagerConfig:
    """Configuration for the data manager."""

    path: str = ''
    """Path to the dataset."""
    white_background: bool = False
    """Whether to use white background."""
    half_res: bool = False
    """Whether to down-sample images to half resolution."""
    view_num_limit: Optional[int] = None
    """Limit the number of training views."""
    testset_skip: int = 8
    """Skip every N views in the test set."""
    video_frame_num: int = 60
    """Number of frames in the each video clip (rotate light & rotate view)."""
    #is_z_up: bool = False
    #"""Whether the scene is z-up. (otherwise y-up)"""
    pixel_sampling_strategy: PixelSamplingStrategy = PixelSamplingStrategy.ALL_IMAGES
    """Pixel sampling strategy for training."""


