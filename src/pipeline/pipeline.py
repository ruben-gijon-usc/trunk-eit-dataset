"""
EIT Dataset Pipeline — Config C (Dual-Task: Classification + Regression).

Dataset design:
- Healthy trunks (0 anomalies) constitute ~25 % of samples.
- Anomaly conductivity is drawn from two physically-motivated bands:
    - Resistive  (dry rot / cracks): [RESISTIVE_BAND] S/m
    - Conductive (moisture / wet rot): [CONDUCTIVE_BAND] S/m
  A gap around base_conductivity prevents ambiguous label boundaries.
- Spatial placement is ring-stratified (central / mid-wood / peripheral).
- Anomaly size is stratified (small / medium / large).

Labels stored per sample:
    anomaly_class  (int): 0=healthy, 1=resistive, 2=conductive, 3=mixed
    anomaly_params (ndarray, shape [n_anomalies, 4]): [r, phi, radius, sigma]
      per anomaly; padded to max_anomalies with NaN when saved to arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..models import Anomaly, Pos, Trunk
from ..eidors.bridge import run_eidors_simulation
from ..simulation import generate_grid


# ---------------------------------------------------------------------------
# Class-label constants
# ---------------------------------------------------------------------------
CLASS_HEALTHY = 0
CLASS_RESISTIVE = 1
CLASS_CONDUCTIVE = 2
CLASS_MIXED = 3

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DatasetSample:
    """One EIT forward-problem sample with all targets for ML.

    Attributes:
        voltages: Boundary voltage measurements from EIDORS (shape depends on
            injection protocol; typically 208 for 16-electrode adjacent).
        conductivity_map: Rasterised conductivity grid (resolution × resolution).
        trunk_config: Original geometric model that produced this sample.
        anomaly_class: Integer label for classification tasks.
            0 = healthy, 1 = resistive only, 2 = conductive only, 3 = mixed.
        anomaly_params: Per-anomaly regression targets, shape (n_anomalies, 4).
            Columns: [center_r (m), center_phi (rad), radius (m), sigma (S/m)].
            Shape is (0, 4) for healthy samples.
        mesh_voltages: Raw EIDORS voltage vector (same as voltages; kept for
            forward-compatibility with mesh-based reconstruction).
        mesh_elem_data: Per-element conductivity assigned by EIDORS (S/m).
        mesh_nodes: FEM node coordinates, shape (n_nodes, 2).
        mesh_elems: FEM triangle connectivity, shape (n_elems, 3), 1-indexed.
    """

    voltages: NDArray[np.float64]
    conductivity_map: NDArray[np.float64]
    trunk_config: Trunk
    anomaly_class: int = CLASS_HEALTHY
    anomaly_params: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float64)
    )
    mesh_voltages: Optional[NDArray[np.float64]] = None
    mesh_elem_data: Optional[NDArray[np.float64]] = None
    mesh_nodes: Optional[NDArray[np.float64]] = None
    mesh_elems: Optional[NDArray[np.int32]] = None


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the EIT dataset generation pipeline.

    Attributes:
        num_samples: Total number of samples to generate.
        grid_resolution: Side length of the square conductivity grid (pixels).
        trunk_radius: Physical radius of the trunk cross-section (m).
        base_conductivity: Background conductivity of healthy wood (S/m).
        num_anomalies_range: Inclusive (min, max) range for the number of
            anomalies per sample.  Set min=0 to include healthy trunks.
        anomaly_conductivity_bands: List of (low, high) conductivity bands (S/m)
            from which anomaly conductivities are drawn with equal probability.
            If None, falls back to ``anomaly_conductivity_range``.
            Default: resistive band (0.001, 0.05) and conductive band (0.15, 1.0),
            bracketing a gap around base_conductivity=0.08 S/m.
        anomaly_conductivity_range: Fallback flat uniform range used only when
            ``anomaly_conductivity_bands`` is None.
        anomaly_radius_range: Fallback flat uniform range for anomaly radius (m),
            used only when ``size_stratify=False``.
        size_stratify: If True, draw anomaly radii from three stratified size
            classes (small / medium / large) with configured probabilities.
        size_classes: List of (r_min, r_max, probability) tuples defining
            size-stratification buckets.  Probabilities must sum to 1.
        spatial_stratify: If True, place anomaly centres in one of three radial
            rings (central / mid-wood / peripheral) chosen uniformly.
        spatial_rings: List of (frac_min, frac_max) tuples representing the
            inner/outer radii of each ring as fractions of the available
            placement radius (trunk_radius - anomaly_radius).
        use_eidors: Whether to call the EIDORS forward solver.
        n_electrodes: Number of equally-spaced boundary electrodes.
        healthy_fraction: Fraction of total samples that should be healthy
            (anomaly_class=0).  The generator enforces this approximately via
            controlled sampling of num_anomalies=0.
    """

    num_samples: int = 8000
    grid_resolution: int = 64
    trunk_radius: float = 1.0
    base_conductivity: float = 0.08

    num_anomalies_range: tuple[int, int] = (0, 4)

    # Dual-band conductivity sampling (Config C defaults)
    anomaly_conductivity_bands: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.001, 0.05), (0.15, 1.0)]
    )
    # Legacy fallback (used when anomaly_conductivity_bands is None / empty)
    anomaly_conductivity_range: tuple[float, float] = (0.001, 1.0)

    # Size stratification: (r_min, r_max, probability)
    size_stratify: bool = True
    size_classes: list[tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.04, 0.12, 0.30),   # small  — 30 %
            (0.12, 0.25, 0.40),   # medium — 40 %
            (0.25, 0.40, 0.30),   # large  — 30 %
        ]
    )
    # Legacy fallback
    anomaly_radius_range: tuple[float, float] = (0.04, 0.40)

    # Spatial stratification: (frac_min, frac_max) of placement radius per ring
    spatial_stratify: bool = True
    spatial_rings: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.00, 0.33),   # central core
            (0.33, 0.66),   # mid-wood
            (0.66, 0.90),   # peripheral
        ]
    )

    use_eidors: bool = True
    n_electrodes: int = 16
    healthy_fraction: float = 0.25


# ---------------------------------------------------------------------------
# Helper: anomaly-class labelling
# ---------------------------------------------------------------------------

def compute_anomaly_class(trunk: Trunk, base_conductivity: float) -> int:
    """Derive the integer classification label from a Trunk configuration.

    The label is based solely on how each anomaly's conductivity compares to
    the trunk's base (background) conductivity:

    - 0  Healthy    — no anomalies present.
    - 1  Resistive  — all anomalies have σ < base_conductivity (dry rot, cracks).
    - 2  Conductive — all anomalies have σ > base_conductivity (moisture, wet rot).
    - 3  Mixed      — anomalies of both types present simultaneously.

    Args:
        trunk: Trunk geometric/physical model.
        base_conductivity: Background conductivity (S/m) used as decision boundary.

    Returns:
        Integer class label (0–3).
    """
    if not trunk.anomalies:
        return CLASS_HEALTHY

    flags = np.array([a.conductivity for a in trunk.anomalies])
    has_resistive = np.any(flags < base_conductivity)
    has_conductive = np.any(flags > base_conductivity)

    if has_resistive and has_conductive:
        return CLASS_MIXED
    if has_resistive:
        return CLASS_RESISTIVE
    return CLASS_CONDUCTIVE


def extract_anomaly_params(trunk: Trunk) -> NDArray[np.float64]:
    """Pack per-anomaly geometric and physical properties into a 2-D array.

    Each row represents one anomaly with columns:
        [center_r (m), center_phi (rad), radius (m), conductivity (S/m)]

    Args:
        trunk: Trunk model whose anomalies are to be extracted.

    Returns:
        Float array of shape (n_anomalies, 4).  Shape is (0, 4) for healthy
        trunks with no anomalies.
    """
    if not trunk.anomalies:
        return np.empty((0, 4), dtype=np.float64)

    rows = [
        [a.center.r, a.center.phi, a.radius, a.conductivity]
        for a in trunk.anomalies
    ]
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Random trunk generation
# ---------------------------------------------------------------------------

def _sample_conductivity(config: PipelineConfig) -> float:
    """Sample one anomaly conductivity from the configured band distribution.

    If ``anomaly_conductivity_bands`` is set, a band is chosen uniformly at
    random and a value is drawn uniformly within that band.  This keeps the
    two physically distinct anomaly types equally represented without manual
    oversampling.

    Args:
        config: Active pipeline configuration.

    Returns:
        Conductivity value in S/m.
    """
    if config.anomaly_conductivity_bands:
        band_idx = np.random.randint(len(config.anomaly_conductivity_bands))
        lo, hi = config.anomaly_conductivity_bands[band_idx]
        return float(np.random.uniform(lo, hi))
    return float(np.random.uniform(*config.anomaly_conductivity_range))


def _sample_radius(config: PipelineConfig) -> float:
    """Sample one anomaly radius using size-class stratification.

    Three buckets — small, medium, large — are sampled with configured
    probabilities via ``np.random.choice`` to guarantee size diversity even
    with a modest sample count.

    Args:
        config: Active pipeline configuration.

    Returns:
        Anomaly radius in metres.
    """
    if config.size_stratify and config.size_classes:
        probs = np.array([s[2] for s in config.size_classes])
        probs = probs / probs.sum()  # normalise in case of floating-point drift
        idx = int(np.random.choice(len(config.size_classes), p=probs))
        lo, hi, _ = config.size_classes[idx]
        return float(np.random.uniform(lo, hi))
    return float(np.random.uniform(*config.anomaly_radius_range))


def _sample_center_r(config: PipelineConfig, anomaly_radius: float) -> float:
    """Sample anomaly centre radial position using ring stratification.

    Uniform placement over [0, R_trunk - r_anomaly] biases toward deeper
    positions because a ring of radius r has area ∝ r.  Ring stratification
    corrects this by dividing the placement radius into three equal-probability
    zones and drawing uniformly within the chosen zone.

    Args:
        config: Active pipeline configuration.
        anomaly_radius: Radius of the anomaly being placed (m).

    Returns:
        Radial distance of anomaly centre from trunk axis (m).
    """
    available_r = config.trunk_radius - anomaly_radius

    if config.spatial_stratify and config.spatial_rings and available_r > 0:
        ring_idx = np.random.randint(len(config.spatial_rings))
        frac_lo, frac_hi = config.spatial_rings[ring_idx]
        r_lo = frac_lo * available_r
        r_hi = frac_hi * available_r
        return float(np.random.uniform(r_lo, r_hi))

    return float(np.random.uniform(0.0, max(available_r, 0.0)))


def generate_random_trunk(config: PipelineConfig) -> Trunk:
    """Generate a randomised trunk with the distribution defined by *config*.

    When ``num_anomalies_range`` starts at 0, healthy trunks are possible.
    The number of anomalies is drawn uniformly; the caller (``generate_dataset``)
    is responsible for enforcing ``healthy_fraction`` at the dataset level.

    Anomaly properties are sampled via:
    - ``_sample_conductivity``  — band-stratified σ
    - ``_sample_radius``        — size-class-stratified r_anomaly
    - ``_sample_center_r``      — ring-stratified radial position

    Args:
        config: Active pipeline configuration.

    Returns:
        Randomly configured Trunk object.
    """
    num_anomalies = int(np.random.randint(
        config.num_anomalies_range[0],
        config.num_anomalies_range[1] + 1
    ))

    anomalies: list[Anomaly] = []
    for _ in range(num_anomalies):
        anomaly_radius = _sample_radius(config)
        center_r = _sample_center_r(config, anomaly_radius)
        center_phi = float(np.random.uniform(0.0, 2.0 * np.pi))
        conductivity = _sample_conductivity(config)

        anomalies.append(Anomaly(
            radius=anomaly_radius,
            center=Pos(r=center_r, phi=center_phi),
            conductivity=conductivity,
        ))

    return Trunk(
        radius=config.trunk_radius,
        base_conductivity=config.base_conductivity,
        anomalies=anomalies,
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(config: PipelineConfig) -> list[DatasetSample]:
    """Generate the full EIT dataset according to *config*.

    Healthy-trunk quota is enforced by pre-computing a boolean mask
    ``force_healthy`` for which samples must have 0 anomalies.  The remaining
    samples draw ``num_anomalies`` freely from the configured range (excluding 0
    if the range permits it).  This guarantees the ``healthy_fraction`` target
    without rejection sampling.

    Each generated sample carries:
    - ``conductivity_map``  — 2-D conductivity grid (regression target, image).
    - ``anomaly_class``     — integer label (classification target).
    - ``anomaly_params``    — per-anomaly parameter array (structured regression).
    - ``voltages``          — boundary voltages from EIDORS (feature vector).

    Args:
        config: Active pipeline configuration.

    Returns:
        List of ``DatasetSample`` objects.
    """
    n = config.num_samples
    n_healthy = int(round(n * config.healthy_fraction))

    # Build a shuffled mask: True ↔ this sample must be healthy
    healthy_mask = np.zeros(n, dtype=bool)
    healthy_mask[:n_healthy] = True
    rng_state = np.random.get_state()
    np.random.shuffle(healthy_mask)
    np.random.set_state(rng_state)  # restore so user seeds are not perturbed
    np.random.shuffle(healthy_mask)

    samples: list[DatasetSample] = []

    for i in range(n):
        if healthy_mask[i]:
            # Force a healthy trunk (no anomalies)
            trunk = Trunk(
                radius=config.trunk_radius,
                base_conductivity=config.base_conductivity,
                anomalies=[],
            )
        else:
            # Draw with at least 1 anomaly
            trunk = generate_random_trunk(config)
            # If the random draw happened to produce 0 anomalies, retry once
            if not trunk.anomalies and config.num_anomalies_range[1] > 0:
                trunk = generate_random_trunk(config)

        grid = generate_grid(trunk, config.grid_resolution)
        label = compute_anomaly_class(trunk, config.base_conductivity)
        params = extract_anomaly_params(trunk)

        sample = DatasetSample(
            voltages=np.empty(0, dtype=np.float64),
            conductivity_map=grid,
            trunk_config=trunk,
            anomaly_class=label,
            anomaly_params=params,
        )

        if config.use_eidors:
            try:
                eidors_result = run_eidors_simulation(
                    trunk,
                    n_electrodes=config.n_electrodes,
                )
                if eidors_result is not None:
                    sample.voltages = eidors_result.voltages
                    sample.mesh_voltages = eidors_result.voltages
                    sample.mesh_elem_data = eidors_result.elem_data
                    sample.mesh_nodes = eidors_result.mesh.nodes
                    sample.mesh_elems = eidors_result.mesh.elems
            except Exception as exc:
                print(f"[WARN] EIDORS failed for sample {i}: {exc}")

        samples.append(sample)

        if (i + 1) % 100 == 0:
            n_healthy_so_far = sum(1 for s in samples if s.anomaly_class == CLASS_HEALTHY)
            print(
                f"Generated {i + 1}/{n} samples "
                f"| healthy={n_healthy_so_far} "
                f"| resistive={sum(1 for s in samples if s.anomaly_class == CLASS_RESISTIVE)} "
                f"| conductive={sum(1 for s in samples if s.anomaly_class == CLASS_CONDUCTIVE)} "
                f"| mixed={sum(1 for s in samples if s.anomaly_class == CLASS_MIXED)}"
            )

    return samples


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def _pad_anomaly_params(
    samples: list[DatasetSample],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Stack per-sample anomaly_params into a padded (N, max_anomalies, 4) array.

    Padding value is NaN so downstream code can mask with ``np.isnan``.

    Args:
        samples: List of dataset samples.

    Returns:
        Tuple of:
            params_arr — shape (N, max_anomalies, 4), float64, NaN-padded.
            counts_arr — shape (N,), int32, number of anomalies per sample.
    """
    counts = np.array([len(s.anomaly_params) for s in samples], dtype=np.int32)
    max_k = int(counts.max()) if counts.max() > 0 else 1

    params_arr = np.full((len(samples), max_k, 4), np.nan, dtype=np.float64)
    for i, s in enumerate(samples):
        k = len(s.anomaly_params)
        if k > 0:
            params_arr[i, :k, :] = s.anomaly_params

    return params_arr, counts


def save_dataset(
    samples: list[DatasetSample],
    output_path: str,
    format: str = "hdf5",
) -> None:
    """Persist dataset to disk in the requested format.

    Saved arrays / fields:
    - ``voltages``          — boundary voltage measurements (features).
    - ``conductivity_maps`` — conductivity grid ground truth (regression).
    - ``anomaly_classes``   — integer class labels (classification).
    - ``anomaly_params``    — per-anomaly [r, phi, radius, sigma], NaN-padded.
    - ``anomaly_counts``    — number of anomalies per sample (unpadding key).
    - ``trunk_radii``       — physical trunk radius per sample (m).
    - ``base_conductivities`` — background conductivity per sample (S/m).
    - Mesh arrays (HDF5/JSON only, when EIDORS was used).

    Args:
        samples: List of generated dataset samples.
        output_path: Destination file path (extension should match format).
        format: One of ``"hdf5"``, ``"numpy"``, or ``"json"``.

    Raises:
        ValueError: If *format* is unrecognised.
    """
    params_arr, counts_arr = _pad_anomaly_params(samples)
    classes_arr = np.array([s.anomaly_class for s in samples], dtype=np.int32)

    if format == "hdf5":
        import h5py
        with h5py.File(output_path, "w") as f:
            # --- features ---
            f.create_dataset("voltages", data=[s.voltages for s in samples])
            # --- regression target: image ---
            f.create_dataset("conductivity_maps", data=[s.conductivity_map for s in samples])
            # --- classification target ---
            f.create_dataset("anomaly_classes", data=classes_arr)
            # --- structured regression targets ---
            f.create_dataset("anomaly_params", data=params_arr)
            f.create_dataset("anomaly_counts", data=counts_arr)
            # --- metadata ---
            f.create_dataset("trunk_radii", data=[s.trunk_config.radius for s in samples])
            f.create_dataset(
                "base_conductivities",
                data=[s.trunk_config.base_conductivity for s in samples],
            )
            # --- mesh data (optional) ---
            has_mesh = samples[0].mesh_voltages is not None
            if has_mesh:
                valid_mesh = [s for s in samples if s.mesh_voltages is not None]
                f.create_dataset("mesh_voltages", data=[s.mesh_voltages for s in valid_mesh])
                f.create_dataset("mesh_elem_data", data=[s.mesh_elem_data for s in valid_mesh])

                max_nodes = max(len(s.mesh_nodes) for s in valid_mesh)
                nodes_arr = np.zeros((len(samples), max_nodes, 2), dtype=np.float64)
                for i, s in enumerate(samples):
                    if s.mesh_nodes is not None:
                        nodes_arr[i, : len(s.mesh_nodes)] = s.mesh_nodes
                f.create_dataset("mesh_nodes", data=nodes_arr)

                max_elems = max(len(s.mesh_elems) for s in valid_mesh)
                elems_arr = np.zeros((len(samples), max_elems, 3), dtype=np.int32)
                for i, s in enumerate(samples):
                    if s.mesh_elems is not None:
                        elems_arr[i, : len(s.mesh_elems)] = s.mesh_elems
                f.create_dataset("mesh_elems", data=elems_arr)

            # Metadata attributes
            f.attrs["format_version"] = "2.0"
            f.attrs["task"] = "dual (classification + regression)"
            f.attrs["class_labels"] = "0=healthy,1=resistive,2=conductive,3=mixed"
            f.attrs["anomaly_params_columns"] = "center_r,center_phi,radius,sigma"
            f.attrs["anomaly_params_padding"] = "NaN"

    elif format == "numpy":
        np.savez(
            output_path,
            voltages=np.array([s.voltages for s in samples]),
            conductivity_maps=np.array([s.conductivity_map for s in samples]),
            anomaly_classes=classes_arr,
            anomaly_params=params_arr,
            anomaly_counts=counts_arr,
            trunk_radii=np.array([s.trunk_config.radius for s in samples]),
            base_conductivities=np.array(
                [s.trunk_config.base_conductivity for s in samples]
            ),
        )

    elif format == "json":
        import json

        data: dict = {
            "format_version": "2.0",
            "task": "dual (classification + regression)",
            "class_labels": "0=healthy,1=resistive,2=conductive,3=mixed",
            "anomaly_params_columns": "center_r,center_phi,radius,sigma",
            "voltages": [s.voltages.tolist() for s in samples],
            "conductivity_maps": [s.conductivity_map.tolist() for s in samples],
            "anomaly_classes": classes_arr.tolist(),
            "anomaly_params": params_arr.tolist(),
            "anomaly_counts": counts_arr.tolist(),
            "trunk_radii": [s.trunk_config.radius for s in samples],
            "base_conductivities": [s.trunk_config.base_conductivity for s in samples],
        }
        has_mesh = samples[0].mesh_voltages is not None
        if has_mesh:
            data["mesh_voltages"] = [
                s.mesh_voltages.tolist() if s.mesh_voltages is not None else []
                for s in samples
            ]
            data["mesh_elem_data"] = [
                s.mesh_elem_data.tolist() if s.mesh_elem_data is not None else []
                for s in samples
            ]
            data["mesh_nodes"] = [
                s.mesh_nodes.tolist() if s.mesh_nodes is not None else []
                for s in samples
            ]
            data["mesh_elems"] = [
                s.mesh_elems.tolist() if s.mesh_elems is not None else []
                for s in samples
            ]
        with open(output_path, "w") as fh:
            json.dump(data, fh)

    else:
        raise ValueError(f"Unknown format: {format!r}. Choose 'hdf5', 'numpy', or 'json'.")

    # Class distribution summary
    unique, counts = np.unique(classes_arr, return_counts=True)
    label_names = {0: "healthy", 1: "resistive", 2: "conductive", 3: "mixed"}
    dist_str = ", ".join(
        f"{label_names.get(int(u), u)}={c} ({100*c/len(samples):.1f}%)"
        for u, c in zip(unique, counts)
    )
    print(f"Saved {len(samples)} samples to {output_path}  [{dist_str}]")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    output_path: str,
    config: Optional[PipelineConfig] = None,
    format: str = "hdf5",
) -> list[DatasetSample]:
    """Run the complete EIT dataset generation pipeline (Config C).

    Generates random trunk configurations → rasterises conductivity grids →
    runs EIDORS forward simulation → labels samples → saves to disk.

    Args:
        output_path: Destination file path.
        config: Pipeline configuration.  Uses Config C defaults when None.
        format: Output format: ``"hdf5"``, ``"numpy"``, or ``"json"``.

    Returns:
        List of generated ``DatasetSample`` objects.
    """
    if config is None:
        config = PipelineConfig()

    print(f"=== EIT Dataset Pipeline (Config C — Dual-Task) ===")
    print(f"  Samples         : {config.num_samples}")
    print(f"  Grid resolution : {config.grid_resolution}×{config.grid_resolution}")
    print(f"  Healthy fraction: {config.healthy_fraction:.0%}")
    print(f"  Conductivity bands: {config.anomaly_conductivity_bands}")
    print(f"  Size stratify   : {config.size_stratify}")
    print(f"  Spatial stratify: {config.spatial_stratify}")
    print(f"  EIDORS          : {config.use_eidors}")
    print(f"  Output          : {output_path} [{format}]")
    print()

    samples = generate_dataset(config)
    save_dataset(samples, output_path, format)

    return samples
