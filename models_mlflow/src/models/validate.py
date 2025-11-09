import os
import pandas as pd
import mlflow
import joblib
import shap
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.utils.metrics import calcular_metricas
from src.rutas import VALIDATION_OUTPUT_PATH

def evaluar_validacion_externa(experiment_name: str, features: list) -> None:
    """
    Evalúa el modelo del experimento MLflow sobre un conjunto de datos externo,
    registra los resultados y genera artefactos, incluyendo trazabilidad completa.
    """
    print(f"\n🧪 Validando experimento '{experiment_name}' externamente...")

    # --- Cargar conjunto de validación ---
    val_path = os.path.join(VALIDATION_OUTPUT_PATH, f"df_validacion_{experiment_name}.csv")
    df_val = pd.read_csv(val_path)
    X_val = df_val[features]
    y_val = df_val['Abandono']
    ids_persona = df_val['IdPersona']

    # --- Cliente MLflow ---
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"❌ El experimento '{experiment_name}' no existe en MLflow")

    # --- Mejor run por AUC ---
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.auc DESC"]
    )[0]
    run_id = best_run.info.run_id

    # --- Cargar modelo y scaler ---
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    scaler_artifact_name = next((a.path for a in client.list_artifacts(run_id) if "scaler" in a.path.lower()), None)
    if scaler_artifact_name is None:
        raise Exception('No se encontró ningún archivo scaler en los artefactos del run.')

    scaler_path = client.download_artifacts(run_id, scaler_artifact_name, "./tmp_artifacts")
    scaler = joblib.load(scaler_path)

    # --- Transformar datos y predecir ---
    X_scaled = scaler.transform(X_val)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # --- Métricas ---
    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc_val = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)

    # --- Importancias globales ---
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    else:
        feature_importances = model.coef_.flatten()

    df_importances_global = pd.DataFrame({
        "Feature": features,
        "Importance": feature_importances
    }).sort_values("Importance", ascending=False)

    os.makedirs("tmp_artifacts", exist_ok=True)
    global_importances_path = f"tmp_artifacts/importancias_global_{experiment_name}.csv"
    df_importances_global.to_csv(global_importances_path, index=False)

    # --- Importancias por persona ---
    importances_df = pd.DataFrame(X_scaled, columns=features)
    for i, f in enumerate(features):
        importances_df[f + '_importance'] = X_scaled[:, i] * feature_importances[i]
    importances_df['IdPersona'] = ids_persona
    person_importances_path = f"tmp_artifacts/importancias_persona_{experiment_name}.csv"
    importances_df.to_csv(person_importances_path, index=False)

    # --- Predicciones ---
    pred_df = pd.DataFrame({
        "IdPersona": ids_persona,
        "y_true": y_val,
        "y_pred": y_pred,
        "y_prob": y_prob
    })
    pred_df['nivel_riesgo'] = pd.cut(pred_df['y_prob'], bins=[0,0.2,0.4,0.6,0.8,1],
                                     labels=["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"])
    preds_path = f"tmp_artifacts/preds_{experiment_name}.csv"
    pred_df.to_csv(preds_path, index=False)

    # --- Registrar modelo en MLflow si no está ---
    MODEL_NAME = experiment_name
    try:
        result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
        else:
            raise
    model_version = result.version

    # --- Registrar validación externa ---
    run_name = f"validacion_externa_{experiment_name}_{type(model).__name__}_v{model_version}"
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
        mlflow.set_tags({
            "type": "validacion_externa",
            "validated_model_run_id": run_id,
            "validated_model_name": type(model).__name__,
            "validated_model_version": model_version
        })

        mlflow.log_param("modelo_validado", run_id)
        mlflow.log_metrics({
            "val_accuracy": float(acc),
            "val_recall": float(rec),
            "val_f1": float(f1),
            "val_auc": float(auc_val)
        })
        mlflow.log_param("caracteristicas_importantes", str(df_importances_global.values.tolist()))
        mlflow.log_artifact(preds_path)
        mlflow.log_artifact(global_importances_path)
        mlflow.log_artifact(person_importances_path)

        # Limpiar archivos temporales
        for f in [preds_path, global_importances_path, person_importances_path]:
            if os.path.exists(f):
                os.remove(f)

    # --- Resumen ---
    print(f"\n🔎 Resultados validación externa ({experiment_name}):")
    print(f"Modelo ganador: {type(model).__name__}")
    print(f"Accuracy: {acc:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc_val:.3f}")
    print("Confusion Matrix:\n", cm)
    print("Top 5 características importantes:\n", df_importances_global.head(5))
    print("✅ Archivos generados: Predicciones, Importancias globales, Importancias por persona")
