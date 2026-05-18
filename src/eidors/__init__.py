"""
EIDORS integration module.

Provides Python interface to EIDORS (via Octave) for EIT forward simulations.
"""

from .bridge import EIDORSResult, run_eidors_simulation

__all__ = ["run_eidors_simulation", "EIDORSResult"]
