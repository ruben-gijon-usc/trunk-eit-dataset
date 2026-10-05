"""
Simulation utilities — public re-export shim.

The canonical implementation lives in ``src.data_representations.grid``.
This module re-exports ``generate_grid`` and ``grid2png`` so that the
existing import path ``from src.simulation import ...`` keeps working.
"""

from .data_representations.grid import generate_grid, grid2png

__all__ = ["generate_grid", "grid2png"]
