import json
import sys
from pathlib import Path

# Resolver la ruta raíz del proyecto para encontrar la carpeta 'dataset'
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import matplotlib.pyplot as plt
import numpy as np


def analyze_dataset(dataset_dir: str = None):
    if dataset_dir is None:
        dataset_dir = str(PROJECT_ROOT / "dataset")
        
    base_path = Path(dataset_dir)
    json_dir = base_path / "json"
    
    if not json_dir.exists():
        print(f"Error: No se encontró el directorio {json_dir}")
        print("¿Ya ejecutaste generate_dataset.py?")
        return

    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No hay archivos JSON en {json_dir}")
        return

    print(f"Analizando {len(json_files)} muestras en el dataset...")

    num_anomalies_list = []
    base_conductivities = []
    
    anomaly_radii = []
    anomaly_conds = []
    anomaly_x = []
    anomaly_y = []

    # Extraer datos de los JSON
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Base del tronco
        base = data.get("base", {})
        base_conductivities.append(base.get("conductivity", 0))

        # Anomalías
        anomalies = data.get("anomalies", [])
        num_anomalies_list.append(len(anomalies))

        for anom in anomalies:
            anomaly_conds.append(anom.get("conductivity", 0))
            shape = anom.get("shape", {})
            
            # Asumimos que la forma puede ser un círculo u otra figura con radio
            radius = shape.get("radius", 0)
            if radius:
                anomaly_radii.append(radius)
            
            # Centro en coordenadas polares (Pos)
            center = shape.get("center", {})
            r = center.get("r", 0)
            phi = center.get("phi", 0)
            
            # Convertir a coordenadas cartesianas para graficar
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            anomaly_x.append(x)
            anomaly_y.append(y)

    # ---------------------------------------------------------
    # Generar visualizaciones
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Análisis de la Distribución del Espacio de Imágenes (Anomalías)', fontsize=16)

    # 1. Distribución del número de anomalías por imagen
    max_anom = max(num_anomalies_list) if num_anomalies_list else 0
    axs[0, 0].hist(num_anomalies_list, bins=np.arange(max_anom + 2) - 0.5, 
                   color='skyblue', edgecolor='black', rwidth=0.8)
    axs[0, 0].set_title('Cantidad de anomalías por tronco')
    axs[0, 0].set_xlabel('Número de anomalías')
    axs[0, 0].set_ylabel('Frecuencia (Imágenes)')
    axs[0, 0].set_xticks(range(max_anom + 1))

    # 2. Distribución de conductividades de las anomalías
    if anomaly_conds:
        axs[0, 1].hist(anomaly_conds, bins=30, color='salmon', edgecolor='black')
    axs[0, 1].set_title('Distribución de Conductividad (Anomalías)')
    axs[0, 1].set_xlabel('Conductividad')
    axs[0, 1].set_ylabel('Frecuencia (Anomalías)')

    # 3. Distribución del tamaño (Radio) de las anomalías
    if anomaly_radii:
        axs[1, 0].hist(anomaly_radii, bins=30, color='lightgreen', edgecolor='black')
    axs[1, 0].set_title('Distribución de Tamaño (Radio)')
    axs[1, 0].set_xlabel('Radio de la anomalía')
    axs[1, 0].set_ylabel('Frecuencia')

    # 4. Distribución Espacial de las anomalías (Heatmap / Scatter)
    if anomaly_x and anomaly_y:
        hb = axs[1, 1].hexbin(anomaly_x, anomaly_y, gridsize=20, cmap='inferno', extent=[-1, 1, -1, 1])
        axs[1, 1].set_title('Distribución Espacial de Anomalías (Centros)')
        axs[1, 1].set_xlabel('Posición X')
        axs[1, 1].set_ylabel('Posición Y')
        cb = fig.colorbar(hb, ax=axs[1, 1])
        cb.set_label('Densidad')
        
        # Dibujar el contorno del tronco (radio 1.0 aproximado)
        circle = plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', linewidth=2)
        axs[1, 1].add_patch(circle)
        axs[1, 1].set_xlim([-1.1, 1.1])
        axs[1, 1].set_ylim([-1.1, 1.1])
        axs[1, 1].set_aspect('equal')

    plt.tight_layout()
    output_img = str(PROJECT_ROOT / "dataset_distribution.png")
    plt.savefig(output_img, dpi=300)
    print(f"\n✅ Análisis completado. Gráficas guardadas en: {output_img}")


if __name__ == "__main__":
    analyze_dataset()
