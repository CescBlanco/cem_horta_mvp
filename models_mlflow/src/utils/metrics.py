from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from typing import List, Dict

def calcular_metricas(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
    """
    Calcula las métricas de rendimiento más comunes para un conjunto de predicciones.

    Parámetros:
        y_true (List[int]): Lista de las etiquetas verdaderas (valores reales de la clase).
        y_pred (List[int]): Lista de las etiquetas predichas por el modelo (valores predichos).
        y_prob (List[float]): Lista de las probabilidades predichas por el modelo para la clase positiva.

    Retorna:
        Dict[str, float]: Un diccionario con las métricas calculadas:
            - "accuracy": Exactitud del modelo.
            - "recall": Recall (sensibilidad) del modelo.
            - "f1_score": Puntuación F1 del modelo.
            - "auc": Área bajo la curva ROC (AUC).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }

