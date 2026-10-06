# 🌲 Trunk EIT Dataset Generator

Un marco avanzado para la simulación y generación de datasets sintéticos de **Tomografía de Impedancia Eléctrica (EIT)** aplicados a la inspección no invasiva de troncos de madera. 

Este proyecto mejora el estado del arte (como el presentado en *Aller et al., 2022*) mediante la inyección de estocasticidad biológica, modelado armónico de anomalías (defectos de la madera) y generación rigurosa de máscaras y matrices espaciales para entrenar modelos de Deep Learning modernos.

## ✨ Características Principales
- **Anomalías Orgánicas:** Uso de coeficientes de series de Fourier para generar defectos de formas irregulares naturales, escapando de los círculos o elipses perfectos clásicos.
- **Variabilidad Estocástica:** Parametrización basada en distribuciones lógicas (Poisson para cantidad de anomalías, Log-Normal para su tamaño). Ningún tronco es idéntico a otro.
- **Simulador EIDORS Dinámico:** Comunicación ultra-rápida en formato binario (`.mat`) entre Python y Octave. Soporte dinámico para ajustar la densidad de la malla, cantidad de electrodos (8, 16, 32) y distintos patrones de estimulación (`adjacent`, `opposite`).
- **Deep Learning Ready:** Exportación de rejillas densas de conductividad (64x64) y extracción de máscaras físicas perfectas basadas puramente en la topología matemática (aislando el ruido de la conductividad o fallos por huecos).

---

## 🏗️ Arquitectura del Código (`src/`)

El núcleo del proyecto (`src/`) sigue principios de diseño orientados a dominio (DDD) separando estrictamente las matemáticas puras de su discretización visual.

* `src/models/`: **Dominio Matemático**. 
  Contiene la topología base independiente de la malla. Define objetos espaciales (`Pos`), topologías abstractas (`Circle`, `Ellipse`, `Harmonic`), y compone el árbol mediante las clases `Anomaly` y `Trunk`. Todo el módulo implementa la interfaz `Serializable` para exportar el modelo a `.json` de forma nativa e invertible.
* `src/data_representations/`: **Discretización e Imágenes**. 
  Convierte los modelos matemáticos puros en matrices tensoriales utilizables por ML. El módulo `grid.py` rasteriza la conductividad evaluando funciones relativas, y extrae la máscara geométrica booleana (`generate_mask`) calculando los límites sub-pixel del tronco.
* `src/forward_process/`: **Puente EIDORS / Octave**.
  Orquesta la simulación del *Forward Problem*. El script `simulator.py` se encarga de crear subprocesos asíncronos aislados en carpetas temporales, pasando datos binariamente mediante SciPy a Octave y devolviendo resultados unificados en un `ForwardResult`.
* `src/stochastic/`: **Generación Procedural**.
  Fábricas (*Factories*) estadísticas. Modulan matemáticamente las distribuciones de los defectos naturales para conformar repositorios de miles de muestras sin sesgos lógicos.

---

## 🚀 Generación de Datasets

El script principal de orquestación se encuentra en `scripts/dataset/generate_dataset.py`. Este script lee los parámetros interactivos de la terminal, inicializa la fábrica estocástica, coordina las simulaciones EIDORS y consolida los resultados eficientemente en disco.

### Uso y Parámetros
Puedes invocar el generador desde la terminal apoyándote en `uv run` para que gestione las dependencias (PyTorch, SciPy, Numpy, Matplotlib):

```bash
uv run python scripts/dataset/generate_dataset.py [OPCIONES]
```

| Parámetro | Tipo | Default | Descripción |
| :--- | :--- | :--- | :--- |
| `--samples` | `int` | `10220` | Cantidad total de troncos (simulaciones) a generar. |
| `--electrodes` | `str` | `"16"` | Número de electrodos alrededor de la corteza. Acepta `"8"`, `"16"`, `"32"` o `"all"`. Si se usa `"all"`, evaluará los 3 conjuntos de electrodos consecutivamente para los mismos troncos, creando datasets directamente comparables. |
| `--pattern` | `str` | `"all"` | Patrón de inyección/medición. Acepta `"adjacent"`, `"opposite"` o `"all"`. Si se elige `"all"`, la simulación calculará ambos espectros eléctricos sobre el *mismo* tronco y configuración espacial. |
| `--seed` | `int` | `42` | Semilla de aleatoriedad. Garantiza la repetibilidad bit a bit de todo el dataset generado en distintos equipos. |

### Ejemplo de Ejecución
```bash
uv run python scripts/dataset/generate_dataset.py --samples 500 --electrodes 32 --pattern all
```
*Este comando generará 500 troncos únicos. Octave generará una malla densa para 32 electrodos y evaluará las corrientes inyectadas usando tanto un patrón adyacente como uno opuesto, todo dentro del mismo ciclo.*

### Estructura de Salida
Para ahorrar un inmenso espacio en disco y mantener la integridad comparativa en los experimentos, el pipeline comparte las matrices espaciales principales y genera subcarpetas exclusivas *solamente* para las variaciones de las mediciones eléctricas:

```text
dataset/dataset_32e_all/
├── json/               # Topología matemática pura de los troncos (para metadatos)
├── grid/               # Imágenes PNG representativas y matrices NumPy base
├── mesh/               # FEM Nodes y Elementos (exportación unificada compartida)
├── voltages_adjacent/  # Tensores 1D con las mediciones en S/m (Patrón Adyacente)
├── elem_data_adjacent/ # Respuestas crudas mapeadas del solver EIDORS 
├── voltages_opposite/  # Tensores 1D con las mediciones en S/m (Patrón Opuesto)
└── elem_data_opposite/ 
```

---

## 🧠 Entrenamiento y Modelos (`training/`)

La carpeta `training/` incluye un robusto pipeline de ML/PyTorch listo para ingerir estos datasets.
* **`eit_dataset.py`**: DataLoader dinámico. Recrea las matrices de conductividad a demanda, extrae las máscaras geométricas y permite cargar un patrón eléctrico específico con `pattern="adjacent"`.
* **`train_utils.py`**: Bucle de entrenamiento agnóstico y universal, inyectando auto-evaluación métrica de 2D (SSIM, Error de Posición, DICE). Excluye automáticamente el "aire" durante el cálculo gracias a la inyección de la máscara del Dataset, evitando que el fondo infle artificialmente las precisiones del modelo.
