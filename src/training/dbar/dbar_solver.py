import numpy as np
import torch

from scripts.data.config import MAX_CONDUCTIVITY


class DBarSolver:
    r"""
    2D Analytical D-Bar (d-bar / \bar{\partial}) Solver for Circular EIT.

    Based on the classical direct reconstruction algorithm (Nachman 1996, Siltanen et al. 2000,
    Mueller et al. 2002).
    Given boundary voltage measurements from equi-spaced electrodes, it:
      1. Converts boundary voltage differences into electrode potential patterns.
      2. Computes the non-linear scattering transform t(k) in the complex frequency plane.
      3. Regularizes high-frequency noise via radial cutoff |k| <= R.
      4. Solves the d-bar equation to reconstruct conductivity directly onto a 2D regular
         grid (64x64) without requiring any FEM mesh.
    """

    def __init__(
        self,
        n_electrodes: int = 16,
        r_cutoff: float = 3.5,
        grid_resolution: int = 64,
        base_conductivity: float = 1.0,
        nk: int = 32,
    ):
        self.n_electrodes = n_electrodes
        self.r_cutoff = r_cutoff
        self.grid_resolution = grid_resolution
        self.base_conductivity = base_conductivity
        self.nk = nk

        # Electrode angles on the unit circle
        self.theta = np.linspace(0, 2 * np.pi, n_electrodes, endpoint=False)
        self.z_elec = np.cos(self.theta) + 1j * np.sin(self.theta)

        # 2D Spatial Grid (z = x + iy)
        x = np.linspace(-1.0, 1.0, grid_resolution)
        y = np.linspace(-1.0, 1.0, grid_resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = self.X + 1j * self.Y
        self.unit_disc_mask = (self.X**2 + self.Y**2) <= 1.0

        # Complex frequency grid in the k-plane: k = k1 + i*k2
        k_vals = np.linspace(-r_cutoff, r_cutoff, nk)
        self.dk = 2 * r_cutoff / (nk - 1)
        self.K1, self.K2 = np.meshgrid(k_vals, k_vals)
        self.K = self.K1 + 1j * self.K2
        self.K_abs = np.abs(self.K)

        # Regularization mask and smooth low-pass filter (Hann window)
        self.k_mask = (self.K_abs > 1e-4) & (self.K_abs <= r_cutoff)
        # Cosine taper near the boundary of the cutoff disc
        taper = 0.5 * (1.0 + np.cos(np.pi * np.clip(self.K_abs / r_cutoff, 0.0, 1.0)))
        self.filter_k = np.where(self.k_mask, taper, 0.0)

        # Precompute reference potential matrix for a homogeneous unit disk
        self.U0 = self._build_homogeneous_reference_matrix()
        self.U0_pinv = np.linalg.pinv(self.U0)

    def _build_homogeneous_reference_matrix(self) -> np.ndarray:
        """
        Builds the analytical boundary potential transfer matrix for a homogeneous
        unit disc with adjacent current injection.
        """
        L = self.n_electrodes
        U0 = np.zeros((L, L), dtype=np.float64)

        # For adjacent dipole injection at electrode j and (j+1),
        # the analytical potential on the boundary unit circle at angle theta is:
        # V(theta) = - (1 / (pi * sigma)) * ln(|e^{i theta} - e^{i theta_A}| / |e^{i theta} - e^{i theta_B}|)
        eps = 1e-6
        for j in range(L):
            th_a = self.theta[j]
            th_b = self.theta[(j + 1) % L]
            for i in range(L):
                if i == j:
                    # Self-impedance approximation
                    U0[i, j] = 1.0
                elif i == (j + 1) % L:
                    U0[i, j] = -1.0
                else:
                    d_a = max(eps, np.abs(np.sin((self.theta[i] - th_a) / 2.0)))
                    d_b = max(eps, np.abs(np.sin((self.theta[i] - th_b) / 2.0)))
                    U0[i, j] = -(1.0 / (np.pi * self.base_conductivity)) * np.log(d_a / d_b)

            # Center potentials (sum to zero)
            U0[:, j] -= np.mean(U0[:, j])

        return U0

    def voltages_to_matrix(self, v_vec: np.ndarray) -> np.ndarray:
        """
        Converts a 1D voltage measurement vector (e.g. 208 elements from 16 injections)
        into a full (L x L) electrode potential matrix U.
        """
        L = self.n_electrodes
        v_flat = np.asarray(v_vec).flatten()

        U = np.zeros((L, L), dtype=np.float64)

        if len(v_flat) == L * L:
            U = v_flat.reshape(L, L)
        elif len(v_flat) == L * (L - 3):
            # EIDORS adjacent pattern with 13 differential measurements per injection
            meas_per_inj = L - 3
            for j in range(L):
                diffs = v_flat[j * meas_per_inj : (j + 1) * meas_per_inj]
                # Reconstruct potentials by cumulative integration
                pots = np.zeros(L, dtype=np.float64)
                # Fill available differential steps
                idx = 0
                for i in range(L):
                    if i != j and i != (j + 1) % L and (i + 1) % L != j:
                        if idx < len(diffs):
                            pots[(i + 1) % L] = pots[i] + diffs[idx]
                            idx += 1
                pots -= np.mean(pots)
                U[:, j] = pots
        else:
            # Fallback: project onto L x L matrix via pseudo-projection
            U = np.zeros((L, L))
            n = min(len(v_flat), L * L)
            U.flat[:n] = v_flat[:n]
            U = 0.5 * (U + U.T)

        # Center potentials
        U -= np.mean(U, axis=0, keepdims=True)
        return U

    def reconstruct(
        self, voltage: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Reconstructs the 2D conductivity image directly from boundary voltage measurements.

        Returns:
            torch.Tensor: Reconstructed conductivity image of shape (1, 64, 64).
        """
        if isinstance(voltage, torch.Tensor):
            v_arr = voltage.detach().cpu().numpy()
        else:
            v_arr = np.asarray(voltage)

        U = self.voltages_to_matrix(v_arr)

        # Compute boundary difference operator: delta_M = U0_pinv @ (U0 - U) @ U0_pinv
        delta_U = self.U0 - U
        delta_M = self.U0_pinv @ delta_U @ self.U0_pinv

        # -------------------------------------------------------------
        # 1. Compute Scattering Transform t(k) in the complex k-plane
        # -------------------------------------------------------------
        L = self.n_electrodes
        t_k = np.zeros((self.nk, self.nk), dtype=np.complex128)

        # Evaluate t(k) at all grid points in the disk |k| <= r_cutoff
        valid_indices = np.where(self.k_mask)
        for idx in range(len(valid_indices[0])):
            i_k = valid_indices[0][idx]
            j_k = valid_indices[1][idx]
            k_val = self.K[i_k, j_k]

            # Boundary exponential vector: e(k)_l = exp(i * k * z_l)
            # z_l = exp(i*theta_l)
            # k * z_l = (k1 + i*k2)*(cos + i*sin)
            kz = k_val * self.z_elec
            e_k = np.exp(1j * kz)
            # Conjugate exponential: e*(k)_l = exp(i * conj(k) * conj(z_l))
            e_conj = np.exp(1j * np.conj(k_val) * np.conj(self.z_elec))

            # Discrete boundary integral for t(k):
            val = e_conj @ delta_M @ e_k
            t_k[i_k, j_k] = val

        # Apply low-pass filter
        t_k_filtered = t_k * self.filter_k

        # -------------------------------------------------------------
        # 2. Solve the D-Bar equation to evaluate mu(z, 0)
        # -------------------------------------------------------------
        # In the Born/Linear D-bar approximation:
        # mu(z, 0) = 1 - (1 / 4*pi^2) \int_{|k| <= R} (t_R(k) / |k|^2) * exp(-2i Re(k*z)) dk1 dk2
        integrand = np.zeros_like(t_k_filtered)
        integrand[self.k_mask] = t_k_filtered[self.k_mask] / (self.K_abs[self.k_mask] ** 2)

        # Spatial evaluation: mu(z, 0) on the 64x64 grid
        mu_minus_one = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.complex128)

        # Evaluate integral by summing over the k-grid:
        k1_flat = self.K1[self.k_mask]
        k2_flat = self.K2[self.k_mask]
        integ_flat = integrand[self.k_mask] * (self.dk**2)

        # Spatial coordinates inside unit disc
        x_flat = self.X.flatten()
        y_flat = self.Y.flatten()

        # Phase factor: -2 * i * Re(k * z) = -2 * i * (k1*x - k2*y)
        # We can vectorize or compute efficiently:
        # Phase matrix shape: (N_k, N_pixels)
        phases = -2j * (np.outer(k1_flat, x_flat) - np.outer(k2_flat, y_flat))
        kernel = np.exp(phases)
        spatial_vals = (integ_flat @ kernel) / (4.0 * np.pi**2)

        spatial_mu = 1.0 - spatial_vals.reshape(self.grid_resolution, self.grid_resolution)

        # Physical conductivity: sigma(x, y) = sigma_0 * (Re(mu(z, 0)))^2
        sigma_recon = self.base_conductivity * (np.maximum(0.01, np.real(spatial_mu))) ** 2

        # Apply geometric unit disc mask
        sigma_recon[~self.unit_disc_mask] = 0.0

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask).squeeze()
            sigma_recon[mask_np == 0] = 0.0

        # Clip within physical conductivity bounds
        sigma_recon = np.clip(sigma_recon, 0.0, MAX_CONDUCTIVITY)

        # Return as (1, 64, 64) PyTorch Tensor
        return torch.tensor(sigma_recon, dtype=torch.float32).unsqueeze(0)
