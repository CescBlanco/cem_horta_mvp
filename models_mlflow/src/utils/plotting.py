import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from typing import List

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], path: str) -> None:
    """
    Dibuja y guarda la matriz de confusión como una imagen.

    Parámetros:
        y_true (List[int]): Lista de las etiquetas verdaderas (valores reales de la clase).
        y_pred (List[int]): Lista de las etiquetas predichas por el modelo (valores predichos).
        path (str): Ruta donde se guardará la imagen de la matriz de confusión.

    Retorna:
        None: La función guarda la matriz de confusión como una imagen en la ruta especificada.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()

def plot_roc_curve(y_test, y_prob, model_name, output_path):
    """
    Genera y guarda la curva ROC para el modelo entrenado.

    Parámetros:
        y_test (array): Etiquetas reales del conjunto de test.
        y_prob (array): Probabilidades predichas por el modelo para la clase positiva.
        model_name (str): Nombre del modelo (para usarlo en el título de la gráfica).
        output_path (str): Ruta donde se guardará la imagen de la curva ROC.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    # Crear la figura de la curva ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Guardar la gráfica
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(model, features, output_path):
    """
    Genera y guarda el gráfico de importancia de las variables para el modelo entrenado.

    Parámetros:
        model: El modelo entrenado (que debe tener el atributo `feature_importances_` o `coef_`).
        features (list): Lista con los nombres de las características.
        output_path (str): Ruta donde se guardará la imagen de la importancia de las variables.
    """
    # Verificar si el modelo tiene el atributo `feature_importances_` o `coef_`
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])  # En caso de ser un modelo lineal como la regresión logística
    else:
        print(f"⚠️ El modelo {model} no tiene información sobre la importancia de las características.")
        return  # Si el modelo no tiene el atributo de importancia, no hacemos nada

    # Ordenar las importancias de mayor a menor
    indices = np.argsort(importances)[::-1]

    # Crear el gráfico
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title(f'Importancia de las Características - {model.__class__.__name__}')
    plt.tight_layout()

    # Guardar la gráfica
    plt.savefig(output_path)
    plt.close()
