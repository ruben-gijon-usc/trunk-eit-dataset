"""
Trunk EIT Dataset - ML dataset generation for EIT applied to tree trunks.

Structure:
- models: Domain models (Trunk, Anomaly, Pos)
- simulation: Grid generation (generate_grid, grid2png)
- eidors: EIDORS integration via Octave
- pipeline: Dataset orchestration (run_pipeline, PipelineConfig)
"""

from .models import Anomaly, Pos, Trunk
from .pipeline import DatasetSample, PipelineConfig, run_pipeline
from .simulation import generate_grid, grid2png

__all__ = [
    "Trunk",
    "Anomaly",
    "Pos",
    "generate_grid",
    "grid2png",
    "run_pipeline",
    "PipelineConfig",
    "DatasetSample",
]
