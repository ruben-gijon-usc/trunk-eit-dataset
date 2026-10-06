"""
Trunk EIT Dataset - ML dataset generation for EIT applied to tree trunks.

Structure:
- models: Domain models (Trunk, Anomaly, Pos)
- simulation: Grid generation (generate_grid, grid2png)
- eidors: EIDORS integration via Octave
- pipeline: Dataset orchestration (run_pipeline, PipelineConfig)
"""

from .data_representations import generate_grid
from .models import Anomaly, Pos, Trunk

__all__ = [
    "Trunk",
    "Anomaly",
    "Pos",
    "generate_grid",
]
