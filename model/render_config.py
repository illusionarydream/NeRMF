from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class DepthComputationType(Enum):
    """Methods to compute depth."""

    AlphaBlend = 'alpha_blending'
    """Alpha blending a depth according to weights."""
    MaximalWeightPoint = 'maximum_point'
    """Use the point with maximal weight as depth, following NeuS's unbiasedness property."""
    SphereTracing = 'sphere_tracing'
    """Use sphere tracing to compute depth, slower."""


class NormalComputationType(Enum):
    """Methods to compute normal."""

    Analytic = 'analytic'
    """Analytic normal from derivative of sdf."""
    NormalizedAnalytic = 'normalized_analytic'
    """Analytic normal from derivative of sdf with normalization."""


@dataclass(frozen=True)
class RendererConfig:
    """Configuration for the NeuS renderer."""

    use_outside_nerf: bool = False
    """Whether to use outside nerf for background."""
    n_samples: int = 64
    """Number of stratified samples per ray."""
    n_importance_samples: int = 64
    """Number of importance samples per ray."""
    n_outside_samples: int = 32
    """Number of samples per ray for outside nerf."""
    normal_type: NormalComputationType = NormalComputationType.NormalizedAnalytic
    """Method to compute normal."""
    up_sample_steps: int = 4
    """Number of steps to up-sample during hierarchical sampling."""
    depth_type: DepthComputationType = DepthComputationType.AlphaBlend
    """Method to compute depth."""
    refract_index : float = 1.470
    """refract_index"""
