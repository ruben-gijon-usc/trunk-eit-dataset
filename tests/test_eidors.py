"""
Tests for EIDORS integration.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import Trunk, Anomaly, Circle
from src.eidors.bridge import run_eidors_simulation, get_eidors_mesh, EIDORSResult


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
        trunk = Trunk(
            radius=1.0,
            base_conductivity=1.0,
            anomalies=[]
        )

        result = run_eidors_simulation(trunk, n_electrodes=16)

        assert result is not None
        assert isinstance(result, EIDORSResult)
        assert result.voltages.shape[0] > 0
        assert result.mesh.nodes is not None
        assert result.mesh.elems is not None

    def test_run_eidors_simulation_with_anomaly(self):
        """Test EIDORS with a single anomaly."""
        trunk = Trunk(
            radius=1.0,
            base_conductivity=1.0,
            anomalies=[
                Anomaly(
                    shape=Circle(cx=0.3, cy=0.0, radius=0.15),
                    conductivity=0.5
                )
            ]
        )

        result = run_eidors_simulation(trunk, n_electrodes=16)

        assert result is not None
        assert result.voltages.shape[0] > 0
        unique_cond = np.unique(result.elem_data)
        assert len(unique_cond) == 2  # base + anomaly

    def test_run_eidors_simulation_multiple_anomalies(self):
        """Test EIDORS with multiple anomalies."""
        trunk = Trunk(
            radius=1.0,
            base_conductivity=1.0,
            anomalies=[
                Anomaly(shape=Circle(cx=0.2, cy=0.0, radius=0.1), conductivity=0.3),
                Anomaly(shape=Circle(cx=0.0, cy=0.4, radius=0.15), conductivity=0.6),
            ]
        )

        result = run_eidors_simulation(trunk, n_electrodes=16)

        assert result is not None
        unique_cond = np.unique(result.elem_data)
        assert len(unique_cond) == 3  # base + 2 anomalies

    def test_voltage_measurements_count(self):
        """Test that we get expected number of voltage measurements."""
        trunk = Trunk(radius=1.0, base_conductivity=1.0, anomalies=[])
        result = run_eidors_simulation(trunk, n_electrodes=16)

        # 16 electrodes, adjacent pattern should give 208 measurements
        assert result.voltages.shape[0] == 208


if __name__ == "__main__":
    pytest.main([__file__, "-v"])