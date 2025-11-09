import os
import time
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import mlflow.sklearn

from src.load_data.loader import cargar_datos
from src.utils.metrics import calcular_metricas
from src.utils.artifacts import *
from src.utils.plotting import *

from mlflow.models.signature import infer_signature

def entrenar_modelos(nombre_experimento: str, features: list) -> pd.DataFrame:
    """
    Entrena varios modelos, evalúa métricas y registra resultados visuales (curva ROC y matriz de confusión)
    en MLflow.
    """

    # Verifica si el directorio para los artefactos existe y si no, créalo
    if not os.path.exists('tmp_artifacts'):
        os.makedirs('tmp_artifacts')

    print("📥 Cargando y preparando datos...")
    # Cargar datos
    X, y = cargar_datos(nombre_experimento, features)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 📊 Modelos y sus hiperparámetros
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    # Hiperparámetros
    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 5, 10],
                        "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]},
        "Gradient Boosting":{"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [2, 3, 4], "subsample": [0.8, 1.0]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "poly"], "gamma": ["scale", "auto"]},
        "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
    }

    mlflow.set_experiment(nombre_experimento)

    # 🧾 Inicialización de resultados
    resultados = []
    best_model, best_auc = None, 0.0

    for nombre, modelo in modelos.items():
        print(f"\n🔍 Entrenando modelo: {nombre}")
        start = time.time()

        # Grid Search con validación cruzada
        grid = GridSearchCV(modelo, param_grids[nombre], cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model_temp = grid.best_estimator_

        # Predicciones
        y_pred = best_model_temp.predict(X_test_scaled)
        y_prob = best_model_temp.predict_proba(X_test_scaled)[:, 1]

        # Métricas
        metrics = calcular_metricas(y_test, y_pred, y_prob)  

        # Matriz de confusión
        cm_path = f"tmp_artifacts/cm_{nombre}.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)
        
        # Curva ROC
        roc_path = f"tmp_artifacts/roc_curve_{nombre}.png"
        plot_roc_curve(y_test, y_prob, nombre, roc_path)

        # Importancia de variables (solo si el modelo tiene esa capacidad)
        feat_imp_path = f"tmp_artifacts/feat_imp_{nombre}.png"
        if hasattr(best_model_temp, 'feature_importances_') or hasattr(best_model_temp, 'coef_'):
            plot_feature_importance(best_model_temp, X_train.columns, feat_imp_path)
        else:
            print(f"⚠️ El modelo {nombre} no tiene información sobre la importancia de las características.")
            feat_imp_path = None  # No generamos la imagen si no tiene importancias

        # Guardar todo en MLflow
        with mlflow.start_run(run_name=nombre) as run:
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "type": "training",
                "model_name": nombre,
                "feature_set": nombre_experimento
            })

            # Firma e input_example basados en los datos escalados
            signature = infer_signature(X_train_scaled, best_model_temp.predict(X_train_scaled))
            input_example = X_train_scaled.head(5)

            mlflow.sklearn.log_model(best_model_temp, name="model", signature=signature, input_example=input_example)

            # Guardar scaler
            scaler_path = guardar_scaler(scaler, nombre)
            mlflow.log_artifact(scaler_path)
            os.remove(scaler_path)

             # Guardar artefactos de gráficos
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(cm_path)
            if feat_imp_path:  # Solo si la imagen existe
                mlflow.log_artifact(feat_imp_path)

            # Limpiar archivos temporales
            os.remove(roc_path)
            os.remove(cm_path)
            if feat_imp_path:
                os.remove(feat_imp_path)

            # Generar la URI del modelo guardado en esta ejecución
            model_uri = f"runs:/{run.info.run_id}/model"

            # Registrar el modelo en el catálogo de modelos de MLflow
            mlflow.register_model(model_uri, f"{nombre_experimento}_{nombre.replace(' ', '_')}")
            
        
        # Comprobar si este modelo es el mejor basado en AUC
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_model = best_model_temp

        resultados.append({
            "Modelo": nombre,
            **metrics,
            "Tiempo": round(time.time() - start, 2)
        })

    # 🏁 Resultados finales
    resultados_df = pd.DataFrame(resultados).sort_values(by="auc", ascending=False)
    print("\n🏁 Resultados finales:\n", resultados_df)

    # ✅ Registro del mejor modelo para despliegue
    if best_model is not None:
        print(f"✅ El mejor modelo es: {best_model.__class__.__name__} con AUC: {best_auc:.3f}")


        # Guardamos el mejor modelo (con sus artefactos) para la validación
        with mlflow.start_run(run_name="best_model") as run:
            mlflow.log_metrics({"best_auc": best_auc})

            mlflow.sklearn.log_model(best_model, name="best_model")
            scaler_path = guardar_scaler(scaler, "best_model_scaler")
            mlflow.log_artifact(scaler_path)
            os.remove(scaler_path)
            mlflow.set_tags({"stage": "production_candidate"})

        best_model_uri = f"runs:/{run.info.run_id}/best_model"
        mlflow.register_model(best_model_uri, f"{nombre_experimento}_best_model")
        
    # 🧹 Limpieza de artefactos temporales
    shutil.rmtree("tmp_artifacts", ignore_errors=True)

    return resultados_df, best_model