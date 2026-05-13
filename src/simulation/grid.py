"""
Grid generation for EIT simulation.
Re-exports from src.data_representations.grid for backward compatibility.
"""

from ..data_representations.grid import generate_grid, grid2png

__all__ = ["generate_grid", "grid2png"]