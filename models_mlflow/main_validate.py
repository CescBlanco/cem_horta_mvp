from src.models.validate import *
from src.rutas import *

import mlflow

"""
Este script configura el entorno de MLflow y evalúa la validación externa de tres experimentos usando diferentes conjuntos de características.

1. **Configuración de MLflow**:
   - Se establece la URI del servidor de seguimiento de MLflow mediante la función `mlflow.set_tracking_uri()`, lo que permite registrar los experimentos, los modelos y sus métricas en el directorio local especificado (`mlruns`).

2. **Evaluación de validación externa**:
   - Se ejecutan tres evaluaciones de validación externa con la función `evaluar_validacion_externa()`, pasando tres conjuntos distintos de características:
     - `FEATURES_1`: Conjunto completo de características para el primer experimento.
     - `FEATURES_2`: Subconjunto de características que excluye "TotalPagadoEconomia".
     - `FEATURES_3`: Subconjunto de características que excluye tanto "VidaGymMeses" como "TotalPagadoEconomia".
   
   En cada ejecución de `evaluar_validacion_externa()`, se realiza la evaluación del modelo sobre un dataset externo, se calculan métricas como **accuracy**, **recall**, **f1**, **AUC**, se obtienen las importancias globales y por persona, y se generan artefactos como predicciones y la matriz de confusión.

3. **MLflow**:
   - Se configura para que el seguimiento de experimentos se realice en el directorio especificado por `mlruns`, donde se almacenarán los artefactos generados y los registros del modelo.
"""

# Configuración de la URI para el seguimiento de MLflow
mlflow.set_tracking_uri("file:./mlruns")

# Evaluación de validación externa de tres experimentos con diferentes conjuntos de características
evaluar_validacion_externa("Experimento_v1", FEATURES_1)
evaluar_validacion_externa("Experimento_v2", FEATURES_2)
evaluar_validacion_externa("Experimento_v3", FEATURES_3)