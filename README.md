# 🌲 Trunk EIT Dataset & Inverse Problem Models

Pipeline completo de simulación física estocástica y Deep Learning para la resolución del **Problema Inverso de Tomografía de Impedancia Eléctrica (EIT)** en secciones transversales de árboles con pudriciones internas.

El repositorio está diseñado siguiendo los principios de **Domain-Driven Design (DDD)**, separando estrictamente la simulación y generación de datos físicos (`src/data`) de las arquitecturas de redes neuronales, funciones de pérdida y evaluación (`src/training`).

---

## 🛠️ Instalación y Requisitos

1. **Requisitos del sistema**:
   - Python 3.10 o superior
   - **GNU Octave** (para la resolución directa del problema forward con EIDORS vía FEM):
     ```bash
     sudo apt update && sudo apt install octave
     ```
   *(La librería EIDORS ya viene preconfigurada en `vendor/eidors`).*

2. **Gestor de dependencias `uv`**:
   ```bash
   uv sync                 # Instala dependencias y crea el virtualenv
   ```

3. **Verificación de la Suite de Tests**:
   ```bash
   uv run pytest -v
   ```

---

## 💾 1. Generación y Análisis de Datos (Simulación)

El generador procedural modela troncos estocásticos con expansiones armónicas y anomalías internas, resolviendo la ecuación en derivadas parciales de Laplace/Calderón $\nabla \cdot (\sigma \nabla V) = 0$ mediante mallas de elementos finitos (FEM) en EIDORS.

### 1.1 Dataset de Entrenamiento Principal (Train/Val)
Genera el conjunto de desarrollo que utilizarán las redes neuronales ($N=10.000$ muestras por defecto):
```bash
uv run python scripts/data/generate_dataset.py --samples 10000 --electrodes 16 --pattern all

# (Opcional para PINNs): Guardar mapas 2D del potencial eléctrico interno V(x,y)
uv run python scripts/data/generate_dataset.py --samples 1000 --electrodes 16 --pattern adjacent --save_voltage_maps
```

### 1.2 Datasets de Testeo Independientes (Benchmarking)
Para evaluar la generalización de las redes sin sesgos, se generan dos conjuntos de prueba independientes:
* **Test ID (In-Distribution):** Sigue la misma distribución anatómica del entrenamiento (~1.000 muestras).
* **Test OOD (Out-Of-Distribution):** Estrés geométrico con múltiples anomalías y altos armónicos (~1.000 muestras).
```bash
# Test ID (Misma distribución)
uv run python scripts/data/generate_test_dataset.py --mode id --samples 1000 --electrodes 16

# Test OOD (Topologías complejas y desafiantes)
uv run python scripts/data/generate_test_dataset.py --mode ood --samples 1000 --electrodes 16
```

### 1.3 Análisis y Distribución del Dataset
Genera gráficos estadísticos sobre las propiedades morfológicas y conductividades de los troncos generados:
```bash
uv run python scripts/data/analyze_dataset.py --dataset_dir dataset/dataset_16e_all
```

### 1.4 Herramientas Interactivas de Visualización
El repositorio incluye visualizadores HTML interactivos en `src/data/interactive_tools/` para inspeccionar y ajustar los parámetros de generación:
* `src/data/interactive_tools/trunk_simulation.html`: Simulación interactiva del tronco y electrodos.
* `src/data/interactive_tools/armonic_shape.html`: Explorador de modos armónicos y deformaciones de corteza.

---

## 🔬 Protocolo de Evaluación y Métricas del Paper

Siguiendo las mejores prácticas de Deep Learning para datasets de $N \ge 10.000$ muestras:
1. **Entrenamiento y Validación:** Split fijo 80% Train / 20% Val con `EarlyStopping(patience=10)` monitorizando `val_loss`.
2. **Evaluación de Generalización:** Los modelos entrenados se evalúan muestra a muestra sobre los conjuntos retenidos independientes:
   * **Test ID** (`dataset/dataset_16e_adjacent_test_id`, ~1.000 muestras).
   * **Test OOD** (`dataset/dataset_16e_adjacent_test_ood`, ~1.000 muestras con estrés geométrico).
3. **Rigor Estadístico:** Las métricas de test se reportan como **media ± desviación típica** ($\text{mean} \pm \text{std}$) a nivel de muestra sobre los conjuntos de prueba:
   * **RMSE Enmascarado** (fidelidad cuantitativa de conductividad $\sigma$).
   * **SSIM Enmascarado** (coherencia estructural y nitidez de bordes).
   * **Dice / IoU** (precisión de segmentación de anomalías).
   * **Position Error (PE)** (error euclidiano en píxeles del centro de masa de la pudrición).
4. **Registro Centralizado (`experiments_results.csv`):** Todas las evaluaciones registran automáticamente una fila con el hash del commit git (`git_commit`), hiperparámetros, métricas en validación y estadísticas en Test ID y Test OOD.

---

## 🤖 2. Modelos Deep Neural Networks (DNN)

Modelos densos con capas dinámicas (`DynamicDNN`), Batch Normalization, Dropout y activaciones ReLU.

### 2.1 Variante A: DNN con Masked RMSE (`scripts/training/dnn/dnn_rmse/`)
```bash
# 1. Búsqueda de Hiperparámetros (Optuna)
uv run python scripts/training/dnn/dnn_rmse/tune_dnn_rmse.py --dataset_dir dataset/dataset_16e_all --trials 20

# 2. Entrenamiento y Evaluación en Test ID / OOD
uv run python scripts/training/dnn/dnn_rmse/evaluate_dnn_rmse.py
```

### 2.2 Variante B: DNN con Hybrid Loss (`scripts/training/dnn/dnn_hybrid/`)
Combina precisión física y coherencia geométrica: $\mathcal{L} = \alpha \cdot \text{RMSE} + \beta \cdot (1 - \text{SSIM})$.
```bash
# 1. Búsqueda de Hiperparámetros (Optuna)
uv run python scripts/training/dnn/dnn_hybrid/tune_dnn_hybrid.py --dataset_dir dataset/dataset_16e_all --trials 20

# 2. Entrenamiento y Evaluación en Test ID / OOD
uv run python scripts/training/dnn/dnn_hybrid/evaluate_dnn_hybrid.py
```

---

## 👁️ 3. Reconstrucción Convolucional 2D (CNN Decoder)

Red convolucional de deconvolución progresiva (`DynamicCNN`) que mapea el vector 1D de voltajes a un volumen latente y reconstruye el plano espacial $64 \times 64$ mediante convoluciones transpuestas con regularización.

### 3.1 Variante A: CNN con Masked RMSE (`scripts/training/cnn/cnn_rmse/`)
```bash
# 1. Búsqueda de Hiperparámetros (Optuna)
uv run python scripts/training/cnn/cnn_rmse/tune_cnn_rmse.py --dataset_dir dataset/dataset_16e_all --trials 20

# 2. Entrenamiento y Evaluación en Test ID / OOD
uv run python scripts/training/cnn/cnn_rmse/evaluate_cnn_rmse.py
```

### 3.2 Variante B: CNN con Hybrid Loss (`scripts/training/cnn/cnn_hybrid/`)
```bash
# 1. Búsqueda de Hiperparámetros (Optuna)
uv run python scripts/training/cnn/cnn_hybrid/tune_cnn_hybrid.py --dataset_dir dataset/dataset_16e_all --trials 20

# 2. Entrenamiento y Evaluación en Test ID / OOD
uv run python scripts/training/cnn/cnn_hybrid/evaluate_cnn_hybrid.py
```

---

## 🔄 4. Autoencoder Física-Inverso (`EITAutoencoder`)

Consta de:
- **`ConvEncoder`**: Modelo sustituto directo (*forward surrogate*) que aprende a predecir los voltajes de frontera $\sigma(x,y) \to \hat{V}$ supervisado con RMSE.
- **`ConvDecoder`**: Modelo inverso que mapea los voltajes a conductividad $\hat{V} \to \hat{\sigma}$. Durante la inferencia / testeo, el decodificador reconstruye directamente a partir de las mediciones reales de frontera.

### 4.1 Variante A: Autoencoder con RMSE (`scripts/training/autoencoder/autoencoder_rmse/`)
```bash
# 1. Búsqueda de Hiperparámetros (Optuna)
uv run python scripts/training/autoencoder/autoencoder_rmse/tune_autoencoder_rmse.py --dataset_dir dataset/dataset_16e_all --trials 20

# 2. Entrenamiento y Evaluación en Test ID / OOD
uv run python scripts/training/autoencoder/autoencoder_rmse/evaluate_autoencoder_rmse.py
```

### 4.2 Variante B: Autoencoder con Hybrid Loss (`scripts/training/autoencoder/autoencoder_hybrid/`)
```bash
# 1. Búsqueda de Hiperparámetros (Optuna)
uv run python scripts/training/autoencoder/autoencoder_hybrid/tune_autoencoder_hybrid.py --dataset_dir dataset/dataset_16e_all --trials 20

# 2. Entrenamiento y Evaluación en Test ID / OOD
uv run python scripts/training/autoencoder/autoencoder_hybrid/evaluate_autoencoder_hybrid.py
```

---

## 🌳 5. Baseline Machine Learning: XGBoost

Línea base clásica con árboles de decisión potenciados por gradiente (MultiOutputRegressor sobre píxeles).

```bash
# 1. Búsqueda de Hiperparámetros (Optuna)
uv run python scripts/training/xgboost/tune_xgboost.py --dataset_dir dataset/dataset_16e_all --trials 10

# 2. Entrenamiento y Evaluación en Test ID / OOD
uv run python scripts/training/xgboost/evaluate_xgboost.py
```

---

## 🧮 6. Baseline Analítico Clásico: Método D-Bar ($\bar{\partial}$)

Solución analítica no iterativa basada en dispersión compleja (método $\bar{\partial}$ de Calderón). **No requiere entrenamiento de pesos** y reconstruye conductividades continuas directamente en el plano 2D.

```bash
uv run python scripts/training/dbar/evaluate_dbar.py --train_dir dataset/dataset_16e_all --r_cutoff 3.5
```
*(Genera figuras vectoriales de alta fidelidad `.pdf` en `scripts/training/dbar/visualizations/` y registra sus estadísticas en `experiments_results.csv`).*

---

## 📁 Estructura del Repositorio

```text
trunk-eit-dataset/
├── experiments_results.csv        # Registro centralizado de resultados experimentales
├── README.md                      # Documentación completa del proyecto
├── AGENTS.md                      # Reglas, arquitectura y directrices para agentes IA
├── pyproject.toml                 # Configuración de dependencias (uv / pip)
├── dataset/                       # Datasets precalculados (.npy)
│   ├── dataset_16e_all/           # Conjunto de entrenamiento/validación (10.000 muestras)
│   ├── dataset_16e_adjacent_test_id/  # Conjunto de prueba In-Distribution
│   └── dataset_16e_adjacent_test_ood/ # Conjunto de prueba Out-Of-Distribution
├── scripts/
│   ├── data/                      # Generación de datos sintéticos, test sets y análisis
│   └── training/                  # Scripts de sintonización (tune) y evaluación
│       ├── dnn/                   # DNN RMSE y DNN Hybrid
│       ├── cnn/                   # CNN RMSE y CNN Hybrid
│       ├── autoencoder/           # Autoencoder RMSE y Autoencoder Hybrid
│       ├── xgboost/               # Baseline XGBoost
│       └── dbar/                  # Baseline analítico D-Bar
├── src/
│   ├── data/                      # Modelos de dominio geométrico y puente EIDORS (Octave)
│   │   └── interactive_tools/     # Visualizadores HTML interactivos
│   └── training/                  # Arquitecturas PyTorch, pérdidas, métricas y utilidades
│       ├── arquitectures/         # DNN, CNN y Autoencoder
│       ├── criterion/             # Masked RMSE, Masked SSIM, Hybrid y TV/PINN
│       ├── dbar/                  # Solver analítico D-Bar 2D
│       ├── experiment_logger.py   # Logger centralizado para experiments_results.csv
│       ├── metrics.py             # Métricas espaciales y evaluadores sample-wise
│       └── train_utils.py         # Bucle de entrenamiento y EarlyStopping
└── tests/                         # Suite completa de tests unitarios (pytest)
```
