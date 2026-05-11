"""
Trunk EIT Dataset - ML dataset generation for EIT applied to tree trunks.

Structure:
- models: Domain models (Trunk, Anomaly, Pos)
- simulation: Forward solvers (generate_grid, eit_simulation)
- eidors: EIDORS integration via Octave
- pipeline: Dataset orchestration (run_pipeline, PipelineConfig)
"""

from .models import Trunk, Anomaly, Pos
from .simulation import generate_grid, grid2png, eit_simulation, EITProtocol
from .pipeline import run_pipeline, PipelineConfig, DatasetSample

__all__ = [
    'Trunk',
    'Anomaly',
    'Pos',
    'generate_grid',
    'grid2png',
    'eit_simulation',
    'EITProtocol',
    'run_pipeline',
    'PipelineConfig',
    'DatasetSample',
]