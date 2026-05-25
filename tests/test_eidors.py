"""
Tests for EIDORS integration.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eidors.bridge import EIDORSResult, get_eidors_mesh, run_eidors_simulation
from src.models import Anomaly, Circle, Pos, SimpleTrunk


class TestEIDORSBridge:
    """Test EIDORS bridge functions."""

    def test_get_eidors_mesh(self):
        """Test getting mesh from EIDORS."""
        mesh = get_eidors_mesh(n_electrodes=16)
        assert mesh is not None
        assert mesh.nodes is not None
        assert mesh.elems is not None
        assert len(mesh.nodes) > 0
        assert len(mesh.elems) > 0

    def test_run_eidors_simulation_homogeneous(self):
        """Test EIDORS with homogeneous trunk (no anomalies)."""
        trunk = SimpleTrunk.create(radius=1.0, base_conductivity=1.0)

        result = run_eidors_simulation(trunk, n_electrodes=16)

        assert result is not None
        assert isinstance(result, EIDORSResult)
        assert result.voltages.shape[0] > 0
        assert result.mesh.nodes is not None
        assert result.mesh.elems is not None

    def test_run_eidors_simulation_with_anomaly(self):
        """Test EIDORS with a single anomaly."""
        trunk = SimpleTrunk.create(
            radius=1.0,
            base_conductivity=1.0,
            anomalies=[
                Anomaly(shape=Circle(center=Pos(r=0.3, phi=0.0), radius=0.15), conductivity=0.5)
            ],
        )

        result = run_eidors_simulation(trunk, n_electrodes=16)

        assert result is not None
        assert result.voltages.shape[0] > 0
        unique_cond = np.unique(result.elem_data)
        assert len(unique_cond) == 2  # base + anomaly

    def test_run_eidors_simulation_multiple_anomalies(self):
        """Test EIDORS with multiple anomalies."""
        trunk = SimpleTrunk.create(
            radius=1.0,
            base_conductivity=1.0,
            anomalies=[
                Anomaly(shape=Circle(center=Pos(r=0.2, phi=0.0), radius=0.1), conductivity=0.3),
                Anomaly(
                    shape=Circle(center=Pos(r=0.4, phi=np.pi / 2), radius=0.15), conductivity=0.6
                ),
            ],
        )

        result = run_eidors_simulation(trunk, n_electrodes=16)

        assert result is not None
        unique_cond = np.unique(result.elem_data)
        assert len(unique_cond) == 3  # base + 2 anomalies

    def test_voltage_measurements_count(self):
        """Test that we get expected number of voltage measurements."""
        trunk = SimpleTrunk.create(radius=1.0, base_conductivity=1.0)
        result = run_eidors_simulation(trunk, n_electrodes=16)

        # 16 electrodes, adjacent pattern should give 208 measurements
        assert result.voltages.shape[0] == 208


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
