"""
EIDORS integration module.

Provides Python interface to EIDORS (via Octave) for EIT forward simulations.
"""

from .bridge import run_eidors_simulation, EIDORSResult

__all__ = ['run_eidors_simulation', 'EIDORSResult']