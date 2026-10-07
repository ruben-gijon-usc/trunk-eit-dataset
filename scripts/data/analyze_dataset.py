import json
import sys
import argparse
from pathlib import Path

# Resolver la ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np


def analyze_dataset(dataset_dir: str):
    base_path = Path(dataset_dir)
    json_dir = base_path / "metadata"
    
    if not json_dir.exists():
        # Fallback para datasets antiguos
        json_dir = base_path / "json"
        if not json_dir.exists():
            print(f"Error: No se encontró el directorio de metadatos en {base_path}")
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

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = data.get("base", {})
        base_conductivities.append(base.get("conductivity", 0))

        anomalies = data.get("anomalies", [])
        num_anomalies_list.append(len(anomalies))

        for anom in anomalies:
            anomaly_conds.append(anom.get("conductivity", 0))
            shape = anom.get("shape", {})
            
            radius = shape.get("base_radius", shape.get("radius", 0))
            if radius:
                anomaly_radii.append(radius)
            
            center = shape.get("center", {})
            r = center.get("r", 0)
            phi = center.get("phi", 0)
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            anomaly_x.append(x)
            anomaly_y.append(y)

    # ---------------------------------------------------------
    # Generar Reporte de Texto (Log)
    # ---------------------------------------------------------
    log_path = base_path / "dataset_analysis_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f"REPORTE DE ANÁLISIS DEL DATASET\n")
        f.write("="*50 + "\n")
        f.write(f"Total de troncos generados: {len(json_files)}\n")
        f.write(f"Total de anomalías generadas: {sum(num_anomalies_list)}\n\n")

        f.write("1. CONDUCTIVIDAD DE LA MADERA SANA (Fondo)\n")
        f.write(f"   Min: {np.min(base_conductivities):.4f} S/m\n")
        f.write(f"   Max: {np.max(base_conductivities):.4f} S/m\n")
        f.write(f"   Media: {np.mean(base_conductivities):.4f} S/m\n")
        f.write(f"   Std: {np.std(base_conductivities):.4f}\n\n")

        f.write("2. CANTIDAD DE ANOMALÍAS POR TRONCO\n")
        f.write(f"   Media: {np.mean(num_anomalies_list):.2f}\n")
        f.write(f"   Max: {np.max(num_anomalies_list)}\n\n")

        if anomaly_conds:
            f.write("3. CONDUCTIVIDAD DE LAS ANOMALÍAS\n")
            f.write(f"   Min: {np.min(anomaly_conds):.4f} S/m\n")
            f.write(f"   Max: {np.max(anomaly_conds):.4f} S/m\n")
            f.write(f"   Media: {np.mean(anomaly_conds):.4f} S/m\n")
            f.write(f"   Std: {np.std(anomaly_conds):.4f}\n\n")

            f.write("4. RADIOS DE LAS ANOMALÍAS\n")
            f.write(f"   Min: {np.min(anomaly_radii):.4f}\n")
            f.write(f"   Max: {np.max(anomaly_radii):.4f}\n")
            f.write(f"   Media: {np.mean(anomaly_radii):.4f}\n")
            f.write(f"   Std: {np.std(anomaly_radii):.4f}\n\n")
            
            f.write("5. DISTANCIA ESPACIAL AL CENTRO\n")
            distances = np.sqrt(np.array(anomaly_x)**2 + np.array(anomaly_y)**2)
            f.write(f"   Distancia Mínima: {np.min(distances):.4f}\n")
            f.write(f"   Distancia Máxima: {np.max(distances):.4f}\n")
            f.write(f"   Distancia Media: {np.mean(distances):.4f}\n")

    print(f"📄 Reporte estadístico guardado en: {log_path}")

    # ---------------------------------------------------------
    # Generar visualizaciones (PDF Priorizado)
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Análisis de la Distribución de Montecarlo', fontsize=16)

    max_anom = max(num_anomalies_list) if num_anomalies_list else 0
    axs[0, 0].hist(num_anomalies_list, bins=np.arange(max_anom + 2) - 0.5, 
                   color='skyblue', edgecolor='black', rwidth=0.8)
    axs[0, 0].set_title('Cantidad de anomalías por tronco')
    axs[0, 0].set_xlabel('Número de anomalías')
    axs[0, 0].set_ylabel('Frecuencia (Imágenes)')
    axs[0, 0].set_xticks(range(max_anom + 1))

    if anomaly_conds:
        axs[0, 1].hist(anomaly_conds, bins=30, color='salmon', edgecolor='black')
    axs[0, 1].set_title('Distribución de Conductividad (Anomalías)')
    axs[0, 1].set_xlabel('Conductividad')
    axs[0, 1].set_ylabel('Frecuencia (Anomalías)')

    if anomaly_radii:
        axs[1, 0].hist(anomaly_radii, bins=30, color='lightgreen', edgecolor='black')
    axs[1, 0].set_title('Distribución de Tamaño (Radio Base)')
    axs[1, 0].set_xlabel('Radio de la anomalía')
    axs[1, 0].set_ylabel('Frecuencia')

    if anomaly_x and anomaly_y:
        hb = axs[1, 1].hexbin(anomaly_x, anomaly_y, gridsize=20, cmap='inferno', extent=[-1, 1, -1, 1])
        axs[1, 1].set_title('Distribución Espacial de Anomalías (Centros)')
        axs[1, 1].set_xlabel('Posición X')
        axs[1, 1].set_ylabel('Posición Y')
        cb = fig.colorbar(hb, ax=axs[1, 1])
        cb.set_label('Densidad')
        
        circle = plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', linewidth=2)
        axs[1, 1].add_patch(circle)
        axs[1, 1].set_xlim([-1.1, 1.1])
        axs[1, 1].set_ylim([-1.1, 1.1])
        axs[1, 1].set_aspect('equal')

    plt.tight_layout()
    output_pdf = str(base_path / "dataset_distribution.pdf")
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    print(f"📊 Gráficas vectoriales guardadas en: {output_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza la distribución estocástica del dataset generado.")
    parser.add_argument("--dataset_dir", type=str, default="dataset/dataset_16e_all", help="Ruta al directorio del dataset")
    args = parser.parse_args()
    
    analyze_dataset(PROJECT_ROOT / args.dataset_dir)
