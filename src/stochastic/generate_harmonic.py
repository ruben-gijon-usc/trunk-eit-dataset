import math
import random
from typing import Literal

from ..models import Anomaly, Harmonic, Pos, SimpleTrunk, Trunk


class StochasticTrunkFactory:
    """
    Highly abstracted Monte Carlo generator for Tree Trunks.
    Uses diverse statistical distributions to simulate biological realism.
    """

    def __init__(
        self,
        # Trunk Dimensions (Log-normal distribution)
        trunk_radius_mu: float = 0.5,     # Mean radius (m)
        trunk_radius_sigma: float = 0.1,  # Standard deviation

        # Anomaly Counts (Poisson distribution)
        expected_anomalies: float = 4.0,  # Average number of anomalies per tree (lambda)

        # Anomaly Placement (Beta distribution for radius)
        # alpha=1, beta=1 -> Uniform area. alpha>1, beta=1 -> Near bark. alpha=1, beta>1 -> Near pith.
        pos_alpha: float = 1.5,
        pos_beta: float = 1.5,

        # Anomaly Size (Exponential distribution)
        anomaly_scale: float = 0.08,      # Average size of an anomaly (m)

        # Fourier/Harmonic Complexity
        max_harmonic_degree: int = 5,
        harmonic_sigma_base: float = 0.25,

        # Physical Properties (Gaussian distribution)
        base_cond_mu: float = 0.01,
        base_cond_sigma: float = 0.002,
        wet_cond_mu: float = 0.1,
        wet_cond_sigma: float = 0.02,
    ):
        self.trunk_radius_mu = trunk_radius_mu
        self.trunk_radius_sigma = trunk_radius_sigma

        self.expected_anomalies = expected_anomalies

        self.pos_alpha = pos_alpha
        self.pos_beta = pos_beta
        self.anomaly_scale = anomaly_scale

        self.max_harmonic_degree = max_harmonic_degree
        self.harmonic_sigma_base = harmonic_sigma_base

        self.base_cond_mu = base_cond_mu
        self.base_cond_sigma = base_cond_sigma
        self.wet_cond_mu = wet_cond_mu
        self.wet_cond_sigma = wet_cond_sigma

        self.propagation_modes: list[Literal["Constant", "Linear", "Cos"]] = ["Constant", "Linear", "Cos"]

    def _poisson_sample(self, lam: float) -> int:
        """Generates a random integer from a Poisson distribution (Knuth's algorithm)."""
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def _get_random_trunk_radius(self) -> float:
        """Lognormal distribution for strictly positive, natural biological sizing."""
        # Convert mean/var to lognormal mu/sigma
        variance = self.trunk_radius_sigma ** 2
        mean = self.trunk_radius_mu
        mu = math.log(mean ** 2 / math.sqrt(variance + mean ** 2))
        sigma = math.sqrt(math.log(1 + (variance / mean ** 2)))
        return random.lognormvariate(mu, sigma)

    def _get_random_pos(self, trunk_radius: float) -> Pos:
        """Beta distribution allows clustering anomalies near the edge, center, or evenly."""
        # Beta returns 0.0 to 1.0. We use sqrt to ensure uniform area density baseline.
        r_normalized = math.sqrt(random.betavariate(self.pos_alpha, self.pos_beta))
        r = trunk_radius * r_normalized
        phi = random.uniform(0, 2 * math.pi)
        return Pos(r=r, phi=phi)

    def _get_random_harmonic(self, trunk_radius: float, pos: Pos, max_attempts: int = 50) -> Harmonic:
        """Generates an organic shape with Rejection Sampling."""
        # Exponential distribution for size: lots of small anomalies, rarely massive ones.
        r0 = random.expovariate(1.0 / self.anomaly_scale)
        r0 = max(0.01, min(r0, trunk_radius * 0.8)) # Clamp to sane values

        # Pick a random degree of complexity for this specific anomaly
        degree = random.randint(1, self.max_harmonic_degree)

        for _ in range(max_attempts):
            harmonics: list[tuple[float, float]] = []
            for k in range(1, degree + 1):
                # Higher frequencies get smaller amplitudes to keep it looking like wood
                sigma = (r0 * self.harmonic_sigma_base) / k
                a = random.gauss(0, sigma)
                b = random.gauss(0, sigma)
                harmonics.append((a, b))

            shape = Harmonic(center=pos, base_radius=r0, harmonics=harmonics)

            # Rejection sampling check
            if shape.is_valid(n_points=180):
                return shape

        # Fallback to a perfect circle if the harmonic generation fails too many times
        return Harmonic(center=pos, base_radius=r0, harmonics=[(0.0, 0.0)])

    def _get_random_anomaly(self, trunk_radius: float) -> Anomaly:
        """Assembles a single physical anomaly."""
        pos = self._get_random_pos(trunk_radius)
        shape = self._get_random_harmonic(trunk_radius, pos)

        # Conductivity normally distributed around the wet average
        conductivity = max(0.0, random.gauss(self.wet_cond_mu, self.wet_cond_sigma))
        propagation_fn = random.choice(self.propagation_modes)

        return Anomaly(shape, conductivity, propagation_fn)

    def generate(self) -> Trunk:
        """
        Generates a completely randomized Trunk based on the factory's initialized distributions.
        Returns:
            Trunk: A fully assembled, structurally valid tree trunk domain.
        """
        # 1. Determine base macro-properties
        trunk_radius = self._get_random_trunk_radius()
        base_conductivity = max(0.0, random.gauss(self.base_cond_mu, self.base_cond_sigma))

        # 2. Determine how many anomalies this tree has (Poisson)
        n_anomalies = self._poisson_sample(self.expected_anomalies)

        # 3. Generate the internal defects
        anomalies = [
            self._get_random_anomaly(trunk_radius)
            for _ in range(n_anomalies)
        ]

        # 4. Assemble
        return SimpleTrunk.create(
            radius=trunk_radius,
            base_conductivity=base_conductivity,
            anomalies=anomalies
        )
