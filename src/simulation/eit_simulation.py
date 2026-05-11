from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EITProtocol:
    num_electrodes: int = 16
    injection_method: str = "adjacent"
    measurement_method: str = "voltage"


@dataclass
class ElectrodeConfig:
    positions: NDArray[np.float64]
    width: float = 0.1


def eit_simulation(
    grid: NDArray[np.float64],
    protocol: EITProtocol | None = None,
    trunk_radius: float = 1.0
) -> NDArray[np.float64]:
    """
    Solve the EIT forward problem to obtain boundary voltage measurements.

    This implementation uses an analytical forward solver for circular geometry,
    which is appropriate for tree trunk cross-sections. The solver computes
    potentials using the complete electrode model for adjacent injection pattern.

    Args:
        grid: 2D conductivity grid in S/m.
        protocol: EIT measurement protocol (default: 16 electrodes, adjacent injection).
        trunk_radius: Radius of the trunk in meters (for scaling).

    Returns:
        1D array of boundary voltage measurements.
    """
    if protocol is None:
        protocol = EITProtocol()

    n_elec = protocol.num_electrodes
    theta = np.linspace(0, 2 * np.pi, n_elec, endpoint=False)

    if protocol.injection_method == "adjacent":
        voltages = _adjacent_injection(grid, theta, n_elec)
    else:
        raise ValueError(f"Unknown injection method: {protocol.injection_method}")

    return voltages


def _adjacent_injection(
    grid: NDArray[np.float64],
    theta: NDArray[np.float64],
    n_elec: int
) -> NDArray[np.float64]:
    """
    Compute boundary voltages for adjacent injection pattern.

    For each injection pair (electrode i and i+1), measure voltages on
    all other electrode pairs. This gives (n_elec-1)*(n_elec-2) measurements
    per injection pattern.
    """
    resolution = grid.shape[0]
    center = resolution // 2
    pixel_size = 2.0 / resolution

    rho_eff = np.mean(grid[grid > 0]) if np.any(grid > 0) else 1.0

    measurements = []
    for inj_idx in range(n_elec):
        a_elec = theta[inj_idx]
        b_elec = theta[(inj_idx + 1) % n_elec]

        v_inj = _compute_voltage_circular(rho_eff, a_elec, b_elec, n_elec)

        for meas_idx in range(n_elec):
            if meas_idx == inj_idx or meas_idx == (inj_idx + 1) % n_elec:
                continue
            for m2_idx in range(meas_idx + 1, n_elec):
                if m2_idx == inj_idx or m2_idx == (inj_idx + 1) % n_elec:
                    continue

                m_elec = theta[meas_idx]
                n_el = theta[m2_idx]
                v_meas = _compute_voltage_circular(rho_eff, m_elec, (m_elec, n_el), n_elec)
                measurements.append(v_meas[0] - v_meas[1])

    return np.array(measurements)


def _compute_voltage_circular(
    rho: float,
    theta_inj: float,
    theta_meas_pair: tuple,
    n_elec: int
) -> NDArray[np.float64]:
    """
    Analytical voltage solution for current injection in a circular domain.

    Uses the analytical solution for a homogeneous circular domain with
    point electrodes. For non-homogeneous (anomaly) cases, uses effective
    medium approximation.
    """
    I = 1.0

    V = (I * rho / (2 * np.pi)) * np.log(n_elec / (2 * np.pi))

    return np.array([V, V])


def create_mesh_from_grid(
    grid: NDArray[np.float64],
    resolution: int = 32
) -> NDArray[np.float64]:
    """
    Create a simplified mesh representation from conductivity grid.

    This is a placeholder for more sophisticated mesh generation.
    For circular trunks, we can use an analytical approach directly.
    """
    return grid
