"""
Configuration constants for the Dataset Generation Pipeline.
Modify these uniform ranges to configure the Monte Carlo distribution.
"""

TRUNK_RADIUS_MIN = 1.0
TRUNK_RADIUS_MAX = 1.0

ANOMALIES_MIN = 1
ANOMALIES_MAX = 3

ANOMALY_RADIUS_MIN = 0.05
ANOMALY_RADIUS_MAX = 0.35

COND_BASE_MIN = 0.5
COND_BASE_MAX = 2.0

COND_ANOMALY_MIN = 10.0
COND_ANOMALY_MAX = 200.0

# -------------------------------------------------------------------
# ML Evaluation Constants (Derived from generation boundaries)
# -------------------------------------------------------------------
# El punto medio entre la madera más conductiva y la anomalía menos conductiva.
GROUND_TRUTH_THRESHOLD = (COND_BASE_MAX + COND_ANOMALY_MIN) / 2.0

# Valor máximo para escalar algoritmos de error estructural (como SSIM)
MAX_CONDUCTIVITY = COND_ANOMALY_MAX
