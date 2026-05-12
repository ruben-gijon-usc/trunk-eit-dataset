import math as m
import random
from typing import List, Tuple
from ..models import Trunk, HarmonicAnomaly, Pos

class RandomTreeFactory:
    def __init__(self, trunk_radius: float, base_conductivity: float, wet_conductivity: float):
        if trunk_radius <= 0:
            raise ValueError("El radio del tronco debe ser mayor que 0.")
        if base_conductivity <= 0:
            raise ValueError("La conductividad base debe ser mayor que 0.")
        if wet_conductivity <= 0:
            raise ValueError("La conductividad base debe ser mayor que 0.")
        
        self.trunk_radius = trunk_radius
        self.base_conductivity = base_conductivity
        self.wet_conductivity = wet_conductivity


    def get_random_anomaly(self, harmonic_degree: int = 3, harmonic_sigma: float = 0.2, mu_r0_multipliyer: float = 0.15, sigma_r0: float = 0.05, max_radius: float = None) -> HarmonicAnomaly:
        if harmonic_degree < 0:
            raise ValueError("El grado armónico no puede ser negativo.")

        # Muestreo de rechazo para asegurar coherencia física
        pos = self.get_random_pos()
        # Asignamos un r0 aleatorio basado en el radio del tronco
        r0 = self.get_random_gaussian(mu=self.trunk_radius * mu_r0_multipliyer, sigma=sigma_r0)
        if max_radius is not None and isinstance(max_radius, float):
            r0 = max(r0, max_radius)
            
        harmonics: List[Tuple[float, float]] = []
        for k in range(1, harmonic_degree + 1):
            sigma = (r0 * harmonic_sigma) / k
            a, b = self.get_random_gaussian(0, sigma), self.get_random_gaussian(0, sigma)
            harmonics.append((a, b))
        
        # Instanciamos la anomalía con una ligera variación en la conductividad húmeda
        anomaly = HarmonicAnomaly(
            r0=r0,
            pos=pos,
            harmonics=harmonics,
            conductivity=self.wet_conductivity * random.uniform(0.8, 1.2)
        )
        return anomaly

    def get_random_trunk(self, n_anomalies: int, harmonic_degree: int = 3, base_sigma: float = 0.2) -> Trunk:
        if n_anomalies < 0:
            raise ValueError("El número de anomalías no puede ser negativo.")
            
        anomalies = [self.get_random_anomaly(harmonic_degree, base_sigma) for _ in range(n_anomalies)]
        
        trunk = Trunk(
            radius=self.trunk_radius,
            base_conductivity=self.base_conductivity,
            anomalies=anomalies
        )
        return trunk

    def get_random_pos(self) -> Pos:
        # Uniformidad por área usando la raíz cuadrada
        r = self.trunk_radius * m.sqrt(random.random())
        phi = random.uniform(0, 2 * m.pi)
        return Pos(r=r, phi=phi)
    
    def get_random_gaussian(self, mu: float = 0., sigma: float = 1., scale: float = 1.) -> float:
        return scale * random.gauss(mu, sigma)
