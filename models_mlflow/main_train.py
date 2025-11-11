
import matplotlib.pyplot as plt

from src.models.train import entrenar_modelos
from src.rutas import *


import mlflow

"""
Este script configura el entorno de MLflow y entrena tres modelos con diferentes conjuntos de características.

1. **Configuración de MLflow**:
   - Se establece la URI del servidor de MLflow mediante la función `mlflow.set_tracking_uri()`, lo que permite registrar los experimentos, los modelos y sus métricas en el directorio local especificado (`mlruns`).

2. **Entrenamiento de modelos**:
   - Se entrena tres modelos diferentes utilizando la función `entrenar_modelos()`, pasando tres conjuntos distintos de características:
     - `FEATURES_1`: Conjunto completo de características para el primer experimento.
     - `FEATURES_2`: Subconjunto de características que excluye "TotalPagadoEconomia".
     - `FEATURES_3`: Subconjunto de características que excluye tanto "VidaGymMeses" como "TotalPagadoEconomia".
   
   Cada ejecución de `entrenar_modelos()` entrena un modelo, realiza una búsqueda en cuadrícula para encontrar los mejores hiperparámetros, registra el modelo entrenado y sus métricas en MLflow y genera artefactos como matrices de confusión y el scaler.

3. **MLflow**:
   - Se configura para que el seguimiento de experimentos se realice en el directorio especificado por `mlruns`, donde se almacenarán los artefactos generados y los registros del modelo.
"""




# Configuración de la URI para el seguimiento de MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Entrenamiento de modelos con diferentes conjuntos de características
entrenar_modelos("Experimento_v1", FEATURES_1)
entrenar_modelos("Experimento_v2", FEATURES_2)
entrenar_modelos("Experimento_v3", FEATURES_3)