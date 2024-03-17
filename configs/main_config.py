from dataclasses import dataclass, field
from typing import Optional, Union
import tyro

from camera.ray_generator import RayGeneratorConfig
from data.data_config import DataManagerConfig
from model.model_config import ModelConfig


@dataclass(frozen=True)
class IntervalsConfig:
    """Configuration for intervals, like saving checkpoints and testing."""

    log_metrics: int = 200
    """Log metrics every N iterations."""
    save_ckpt: int = 5_000
    """Save checkpoint every N iterations."""
    render_test_views: int = 250_000
    """Test every N iterations."""
    render_video: int = 1_000_000
    """Render video every N iterations."""
    dump_mesh: int = 500_000
    """Dump mesh every N iterations."""


@dataclass(frozen=True)
class SystemConfig:
    """Configuration for the whole system."""

    model: ModelConfig = ModelConfig()
    data: DataManagerConfig = DataManagerConfig()
    ray_generator: RayGeneratorConfig = RayGeneratorConfig()
    intervals: IntervalsConfig = IntervalsConfig()

    ckpt_path: Optional[str] = None
    """Path to the checkpoint to load."""
    base_dir: str = 'outputs'
    """Base directory for saving outputs."""
    exp_name: str = 'baseline'
    """Name of the experiment setup."""
    scene_name: str = 'scene'
    """Name of the scene, can be used for other remarks."""

    seed: int = 3407
    """Random seed for reproducibility."""
    serialized_shm_info: Optional[str] = None
    """Serialized SHM info as srt arg."""
    evaluation_only: bool = False
    """Whether to only run evaluation."""
    cuda_number: int = 0  #这里是测试参数，写完以后记得删掉
    """cuda index"""


# SubCommands
@dataclass(frozen=True)
class NeRMF(SystemConfig):
    """NeRMF Config"""
    
    pass

# MainCommands
@dataclass
class MainArgs:
    """MainArgs"""
    
    config: tyro.conf.FlagConversionOff[NeRMF]
    
    # Turn off flag conversion for booleans with default values.

# function to get config
def get_config():
    return tyro.cli(MainArgs)


if __name__ == '__main__':
    config = get_config()
    print(config)
