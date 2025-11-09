import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from typing import Tuple

def guardar_scaler(scaler, nombre_modelo: str) -> str:
    """
    Guarda un scaler en un archivo .pkl dentro del directorio `tmp_artifacts`.

    Parámetros:
        scaler (object): El scaler que se desea guardar (por ejemplo, StandardScaler o MinMaxScaler).
        nombre_modelo (str): El nombre del modelo al que pertenece el scaler, usado para crear el nombre del archivo.

    Retorna:
        str: La ruta del archivo donde se guarda el scaler (por ejemplo, "tmp_artifacts/scaler_nombre_modelo.pkl").
    """

    os.makedirs("tmp_artifacts", exist_ok=True)
    path = f"tmp_artifacts/scaler_{nombre_modelo}.pkl"
    joblib.dump(scaler, path)
    return path

def cargar_modelo_y_scaler(experimento: str, metric: str = "auc") -> Tuple[object, object, str]:

    """
    Carga el mejor modelo y su scaler asociado de un experimento en MLflow, basado en una métrica especificada.

    Parámetros:
        experimento (str): El nombre del experimento en MLflow del cual se va a cargar el modelo.
        metric (str, opcional): La métrica que se utilizará para seleccionar el mejor modelo (por defecto "auc").

    Retorna:
        tuple: Una tupla que contiene:
            - model (sklearn model): El modelo cargado.
            - scaler (object): El scaler asociado al modelo.
            - run_id (str): El ID del run de MLflow asociado al modelo.

    Lanza:
        ValueError: Si no se encuentra el experimento en MLflow.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experimento)
    if not experiment:
        raise ValueError(f"No se encontró el experimento {experimento} en MLflow")

    best_run = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )[0]
    run_id = best_run.info.run_id
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # Buscar y descargar scaler
    artifacts = client.list_artifacts(run_id)
    scaler_artifact = next(a.path for a in artifacts if "scaler" in a.path.lower())
    scaler_path = client.download_artifacts(run_id, scaler_artifact, "./tmp_artifacts")
    scaler = joblib.load(scaler_path)

    return model, scaler, run_id