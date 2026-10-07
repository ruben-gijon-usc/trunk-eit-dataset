# 🌲 Trunk EIT Dataset & Inverse Problem Models

Pipeline completo de simulación estocástica y Machine Learning para resolver el Problema Inverso de Tomografía de Impedancia Eléctrica (EIT) en troncos de madera con pudriciones internas.

Este repositorio está arquitecturado siguiendo **Domain-Driven Design (DDD)** separando la generación de datos físicos simulados (`src/data`) del proceso puro de Deep Learning (`src/training`).

---

## 🛠️ Instalación

Se utiliza `uv` como gestor de paquetes y dependencias ultra-rápido de Python:

```bash
uv sync                 # Instala dependencias del proyecto
```

---

## 💾 1. Generación de Datos (Fase de Simulación)

El generador crea geometrías estocásticas de madera y hongos a partir de armónicos y resuelve el problema físico (*Forward Problem*) conectando con EIDORS vía GNU Octave.

### 1.1 Generar el Dataset de Entrenamiento (Train/Val)
Genera el conjunto de desarrollo que utilizarán los modelos.
```bash
uv run python scripts/data/generate_dataset.py --samples 5000 --electrodes 16 --pattern all
```

### 1.2 Generar los Datasets de Testeo (Benchmarking)
Para demostrar la robustez física de las redes, generamos dos datasets de prueba independientes:
* **Test ID (In-Distribution):** Sigue las mismas reglas geométricas que el entrenamiento.
* **Test OOD (Out-Of-Distribution):** Multi-anomalías extremas, bordes puntiagudos (altos armónicos).
```bash
# Generar Test ID (Misma distribución)
uv run python scripts/data/generate_test_dataset.py --mode id --samples 1000 --electrodes 16

# Generar Test OOD (Stress Test Geométrico)
uv run python scripts/data/generate_test_dataset.py --mode ood --samples 1000 --electrodes 16
```

---

## 🤖 2. Entrenamiento Deep Neural Networks (DNN)

La arquitectura de ejecución consta de 2 pasos estrictos para evitar *Data Leakage*. (Nota: los voltajes de entrada se **normalizan automáticamente (Z-Score)** por defecto al cargar el dataset para asegurar convergencia estable).

### Paso 1: Búsqueda de Hiperparámetros (Optuna)
Busca la mejor estructura (capas paramétricas) y learning rates optimizando una función de pérdida específica.
```bash
# Tuning usando MSE clásico
uv run python scripts/training/dnn/tune_dnn.py --dataset_dir dataset/dataset_16e_adjacent --loss mse --trials 20

# Tuning usando Loss Híbrida (RMSE Espacial + SSIM)
uv run python scripts/training/dnn/tune_dnn.py --dataset_dir dataset/dataset_16e_adjacent --loss hybrid --trials 20
```

### Paso 2: Validación 5-Fold y Modelo Final
Lee el JSON generado en el Paso 1, ejecuta una validación cruzada y entrena el modelo final con el 100% de los datos guardando el `.pt`.
```bash
uv run python scripts/training/dnn/evaluate_dnn.py --dataset_dir dataset/dataset_16e_adjacent --params_file scripts/training/dnn/best_params_dnn_hybrid.json
```

---

## 🌳 3. Entrenamiento XGBoost (Baseline)

Modelo base de Machine Learning tradicional. Predecir los 4096 elementos del *Grid* es intensivo en CPU, por lo que su `tune` restringe el número de ramas lógicas para lograr entrenamientos resolubles.

### Paso 1: Búsqueda de Hiperparámetros (Optuna)
```bash
uv run python scripts/training/xgboost/tune_xgboost.py --dataset_dir dataset/dataset_16e_adjacent --trials 10
```

### Paso 2: Validación 5-Fold y Modelo Final
Evalúa el árbol usando las mismas métricas que las DNNs (convirtiendo internamente a tensores 2D).
```bash
uv run python scripts/training/xgboost/evaluate_xgboost.py --dataset_dir dataset/dataset_16e_adjacent --params_file scripts/training/xgboost/best_params_xgboost.json
```

---

## 📊 Arquitectura del Repositorio

* `src/data/` — Lógica de generación, máscaras 2D y conexión con EIDORS (Octave).
* `src/training/` — Clases puras de ML (PyTorch Dataset, Métricas, `criterion/` para Losses, EarlyStopping).
* `scripts/` — Ejecutables orquestadores (*Entrypoints*). Donde lanzas los comandos.
* `dataset/` — Carpeta autogenerada con los tensores `.npy` y los *Ground Truth*.
