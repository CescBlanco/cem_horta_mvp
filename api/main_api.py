from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field, field_validator
from typing import List, Union
import mlflow.sklearn
import os
import joblib
import json
from datetime import datetime
import random

# PATH para guardar predicciones.
PREDICCIONES_PATH = "data/predicciones_api.csv"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experimento_nombre= 'Experimento_v3'
metric_name= 'auc'  
client = MlflowClient()

def encontrar_mejor_modelo(experimento_nombre, metric_name):
    """
    Encuentra el mejor modelo registrado en MLFlow basado en una métrica específica.
    
    Parámetros:
        experimento_nombre (str): El nombre del experimento en MLFlow.
        metric_name (str): La métrica a usar para la selección del mejor modelo.

    Retorna:
        tuple: El modelo entrenado y la información del mejor modelo.
    """
    experiment = client.get_experiment_by_name(experimento_nombre)
    if experiment is None:
        print(f"El experimento {experimento_nombre} no existe en MLFlow.")
    else:
        print(f"Experimento encontrado: {experiment.name} (ID: {experiment.experiment_id})")

    runs= client.search_runs(experiment.experiment_id, filter_string="", run_view_type=mlflow.tracking.client.ViewType.ALL)
    
    mejores_runs= []
    for run in runs:
        auc= run.data.metrics.get(metric_name, None)
        
        if auc is not None:
            mejores_runs.append({
                "run_id": run.info.run_id,
                "auc": auc,
                "accuracy": run.data.metrics.get("accuracy", None),
                "f1_score": run.data.metrics.get("f1_score", None),
                "recall": run.data.metrics.get("recall", None)
            })

        df_resultados= pd.DataFrame(mejores_runs)

        if df_resultados.empty:
            print(f"No se encontraron métricas para el experimento {experimento_nombre}.")
        else:
            mejor_modelo_info= df_resultados.sort_values(by=metric_name, ascending=False).iloc[0]
            print(f"Mejor modelo encontrado: Run ID: {mejor_modelo_info['run_id']}, AUC: {mejor_modelo_info[metric_name]}")

            modelo_uri= f"runs:/{mejor_modelo_info['run_id']}/model"
            modelo_final= mlflow.sklearn.load_model(modelo_uri)

    return modelo_final, mejor_modelo_info

modelo_final, mejor_modelo_info = encontrar_mejor_modelo(experimento_nombre, metric_name)

print("Modelo cargado correctamente:", modelo_final)
print("Detalles del mejor modelo:", mejor_modelo_info)
run_id = mejor_modelo_info['run_id']

app = FastAPI(title="API Abandono flexible")

@app.get("/", summary='Mensaje de bienvenida')
def read_root():
    return {"mensaje": "API de predicción de abandono con modelo entrenado en MLflow"}


# Función utilitaria para cargar el scaler dinámicamente.
def obtener_scaler_dinamico(client, run_id):

    """
    Carga el scaler asociado a un modelo entrenado desde los artefactos en MLFlow.
    
    Parámetros:
        client (MlflowClient): Cliente de MLFlow para acceder a los artefactos.
        run_id (str): ID del run del modelo del que se desea cargar el scaler.

    Retorna:
        Scaler: Un objeto `Scaler` cargado desde los artefactos del modelo.

    Lanza:
        FileNotFoundError: Si no se encuentra el scaler en los artefactos del run.
    """

    artifacts = client.list_artifacts(run_id)
    print("Artifacts en el run:")
    scaler_path = None
    for artifact in artifacts:
        if 'scaler' in artifact.path.lower() and artifact.is_dir is False:
            scaler_path = artifact.path
            break
    if scaler_path is None:
        raise FileNotFoundError(f"No se encontró ningún scaler en los artefactos del run {run_id}")
    local_path = client.download_artifacts(run_id, scaler_path, "./tmp_artifacts")
    scaler = joblib.load(local_path)
    return scaler

scaler = obtener_scaler_dinamico(client, run_id)
print("Modelo cargado correctamente:", modelo_final)
print("Detalles del mejor modelo:", mejor_modelo_info)
print("Detalles del mejor scaler:", scaler)


def validar_columnas_esperadas(df, columnas_esperadas):

    """
    Valida que el DataFrame contenga las columnas esperadas.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos a validar.
        columnas_esperadas (list): Lista de nombres de columnas esperadas.

    Retorna:
        list: Lista de errores con las columnas faltantes o adicionales.
    """

    columnas_faltantes = set(columnas_esperadas) - set(df.columns)
    columnas_extra = set(df.columns) - set(columnas_esperadas)
    errores = []
    if columnas_faltantes:
        errores.append(f"Faltan columnas: {', '.join(columnas_faltantes)}")
    if columnas_extra:
        errores.append(f"Columnas inesperadas: {', '.join(columnas_extra)}")
    return errores

def obtener_importancia_por_persona(modelo, X_scaled, features):

    """
    Obtiene la importancia de las características para cada persona usando un modelo.
    
    Parámetros:
        modelo (sklearn model): El modelo entrenado para hacer predicciones.
        X_scaled (np.array): Datos de entrada escalados para la predicción.
        features (list): Lista de características utilizadas por el modelo.

    Retorna:
        pd.DataFrame: DataFrame con las importancias por persona.
    """

    # Obtener las importancias globales (coeficientes o feature_importances_)
    feature_importances = modelo.feature_importances_ if hasattr(modelo, "feature_importances_") else modelo.coef_.flatten()
    
    # Crear un DataFrame de importancias por persona
    importances_df = pd.DataFrame(X_scaled, columns=features)
    
    # Calcular la importancia de cada característica para cada persona
    for i, f in enumerate(features):
        importances_df[f + '_importance'] = X_scaled[:, i] * feature_importances[i]
    
    return importances_df

def guardar_predicciones_api(idpersona, variables, pred, prob, nivel, endpoint, run_id, importancia_variables=None):
    """
    Guarda las predicciones realizadas por el modelo en un archivo CSV para su posterior análisis.
    
    Parámetros:
        idpersona (int): El ID de la persona a la que corresponde la predicción.
        variables (dict): Variables de entrada utilizadas para la predicción.
        pred (int): Predicción realizada por el modelo (0 o 1).
        prob (float): Probabilidad asociada a la predicción.
        nivel (str): Nivel de riesgo asignado a la predicción.
        endpoint (str): El endpoint que hizo la predicción.
        run_id (str): El ID del run de MLFlow relacionado con el modelo.
        importancia_variables (pd.DataFrame, opcional): Importancia de las variables para esta predicción.
    
    Retorna:
        None
    """
    print(f"Guardando predicción para IdPersona: {idpersona}")
    # Crear el registro con los resultados de la predicción
    registro = {
        "IdPersona": idpersona,
        "VariablesEntrada": json.dumps(variables, ensure_ascii=False),
        "Prediccion": int(pred),
        "Probabilidad": float(prob),
        "NivelRiesgo": nivel,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Endpoint": endpoint,
        "RunIDModelo": run_id,
        "ImportanciaVariables": importancia_variables,

    }

    # Si el archivo de predicciones ya existe, se añaden nuevas filas, si no, se crea
    if os.path.exists(PREDICCIONES_PATH):
        pd.DataFrame([registro]).to_csv(PREDICCIONES_PATH, mode='a', index=False, header=False)
    else:
        pd.DataFrame([registro]).to_csv(PREDICCIONES_PATH, index=False)



# Cargar columnas esperadas.
with open("data/columnas_modelo3.txt") as f:
    columnas_modelo3 = f.read().splitlines()

# Cargar dataset de validación (para endpoint por ID).
df_validacion = pd.read_csv('data/df_validacion_Experimento_v3.csv')



# MODELOS DE DATOS
class UserData(BaseModel):

    """
    Clase que representa los datos de entrada del usuario para la predicción de abandono.
    
    Atributos:
        Edad (int): Edad de la persona. Debe ser mayor o igual a 18.
        Sexo_Mujer (bool): Indica si el usuario es mujer (True) o no (False).
        UsoServiciosExtra (bool): Indica si el usuario utiliza servicios extra (True) o no (False).
        ratio_cantidad_2025_2024 (float): Ratio de la cantidad de visitas entre los años 2025 y 2024.
        Diversidad_servicios_extra (int): Número de servicios extra que utiliza el usuario.
        TienePagos (bool): Indica si el usuario tiene pagos (True) o no (False).
        TotalVisitas (int): Total de visitas realizadas por el usuario.
        DiasActivo (int): Número de días activos del usuario.
        VisitasUlt90 (int): Número de visitas del usuario en los últimos 90 días.
        VisitasUlt180 (int): Número de visitas del usuario en los últimos 180 días.
        TieneAccesos (bool): Indica si el usuario tiene accesos (True) o no (False).
        VisitasPrimerTrimestre (int): Número de visitas realizadas en el primer trimestre del año.
        VisitasUltimoTrimestre (int): Número de visitas realizadas en el último trimestre del año.
        DiaFav_domingo (bool): Indica si el domingo es el día favorito del usuario (True) o no (False).
        DiaFav_jueves (bool): Indica si el jueves es el día favorito del usuario (True) o no (False).
        DiaFav_lunes (bool): Indica si el lunes es el día favorito del usuario (True) o no (False).
        DiaFav_martes (bool): Indica si el martes es el día favorito del usuario (True) o no (False).
        DiaFav_miercoles (bool): Indica si el miércoles es el día favorito del usuario (True) o no (False).
        DiaFav_sabado (bool): Indica si el sábado es el día favorito del usuario (True) o no (False).
        DiaFav_viernes (bool): Indica si el viernes es el día favorito del usuario (True) o no (False).
        EstFav_invierno (bool): Indica si el invierno es la estación favorita del usuario (True) o no (False).
        EstFav_otono (bool): Indica si el otoño es la estación favorita del usuario (True) o no (False).
        EstFav_primavera (bool): Indica si la primavera es la estación favorita del usuario (True) o no (False).
        EstFav_verano (bool): Indica si el verano es la estación favorita del usuario (True) o no (False).
    
    Métodos:
        field_validator: Valida los valores de ciertas características, como la edad, el ratio y las variables no negativas.
    """

    Edad: int
    Sexo_Mujer: bool
    UsoServiciosExtra: bool
    ratio_cantidad_2025_2024: float
    Diversidad_servicios_extra: int
    TienePagos: bool
    TotalVisitas: int
    DiasActivo: int
    VisitasUlt90: int
    VisitasUlt180: int
    TieneAccesos: bool
    VisitasPrimerTrimestre: int
    VisitasUltimoTrimestre: int
    DiaFav_domingo: bool
    DiaFav_jueves: bool
    DiaFav_lunes: bool
    DiaFav_martes: bool
    DiaFav_miercoles: bool
    DiaFav_sabado: bool
    DiaFav_viernes: bool
    EstFav_invierno: bool
    EstFav_otono: bool
    EstFav_primavera: bool
    EstFav_verano: bool

    # Usando field_validator para Pydantic V2
    @field_validator('Edad')
    def edad_mayor_18(cls, value):
        """
        Valida que la edad del usuario sea mayor o igual a 18.
        
        Parámetros:
            value (int): Edad de la persona.
        
        Retorna:
            int: Edad del usuario, si es válida.
        
        Lanza:
            ValueError: Si la edad es menor a 18.
        """

        if value < 18:
            raise ValueError('Edad debe ser mayor o igual a 18')
        return value
    
    @field_validator('ratio_cantidad_2025_2024')
    def ratio_no_negativo(cls, value):
        """
        Valida que el valor del ratio no sea negativo.
        
        Parámetros:
            value (float): Valor del ratio.
        
        Retorna:
            float: El valor del ratio, si es válido.
        
        Lanza:
            ValueError: Si el ratio es negativo.
        """

        if value < 0:
            raise ValueError('El ratio no puede ser negativo')
        return value

    @field_validator('Diversidad_servicios_extra', 'TotalVisitas', 'DiasActivo', 'VisitasUlt90',
                      'VisitasUlt180', 'VisitasPrimerTrimestre', 'VisitasUltimoTrimestre')
    def valores_no_negativos(cls, value, field):
        """
        Valida que las características relacionadas con visitas y actividad no sean negativas.
        
        Parámetros:
            value (int): Valor de la característica.
            field (str): Nombre de la característica a validar.
        
        Retorna:
            int: El valor de la característica, si es válido.
        
        Lanza:
            ValueError: Si el valor es negativo.
        """
        if value < 0:
            raise ValueError(f"{field.name} no puede ser negativo")
        return value

class MultiUserData(BaseModel):
    """
    Clase que permite manejar múltiples usuarios para la predicción.
    
    Atributos:
        datos (List[UserData]): Lista de datos de múltiples usuarios.
    """
    datos: List[UserData]


class IDRequest(BaseModel):
    """
    Clase para validar la solicitud con un solo ID de persona.
    
    Atributos:
        IdPersona (int): El ID de la persona.
    """

    IdPersona: int


class IDListRequest(BaseModel):
    """
    Clase para validar la solicitud con una lista de IDs de personas.
    
    Atributos:
        Ids (List[int]): Lista de IDs de personas.
    """

    Ids: List[int]


@app.get("/", summary='Mensaje de bienvenida')
def index():
    """
    Endpoint de bienvenida que devuelve un mensaje de introducción a la API.
    
    Retorna:
        dict: Mensaje de bienvenida.
    """

    return {"mensaje": "API de predicción de abandono con modelo entrenado en MLflow"}

#FUNCIONA
@app.post("/predecir_abandono_socio_simulado/", summary="Predicción por datos simulados de usuario")
def predecir_abandono_socio_simulado(data: Union[UserData, MultiUserData]):
    """
    Endpoint para realizar una predicción de abandono para uno o varios usuarios simulados,
    utilizando los datos de entrada proporcionados.
    
    Parámetros:
        data (Union[UserData, MultiUserData]): Datos de usuario (uno o varios).
    
    Retorna:
        dict: Resultado de la predicción, incluyendo probabilidad, nivel de riesgo y características importantes.
    """
    # Si es un solo usuario
    if isinstance(data, UserData):
        df = pd.DataFrame([data.dict()])
    # Si es más de un usuario
    else:
        df = pd.DataFrame([d.dict() for d in data.datos])

    # Asegurarse de que las columnas booleanas se conviertan a enteros
    bool_cols = ['Sexo_Mujer', 'UsoServiciosExtra', 'TienePagos', 'TieneAccesos',
                 'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
                 'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
                 'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']
    df[bool_cols] = df[bool_cols].astype(int)

    # Filtrar las columnas que se usan en el modelo
    df = df[columnas_modelo3]

    # Verificación de columnas necesarias
    errores = validar_columnas_esperadas(df, columnas_modelo3)
    if errores:
        raise HTTPException(status_code=400, detail="; ".join(errores))

    # Realizar la predicción
    X_scaled = scaler.transform(df)
    prediccion = modelo_final.predict(X_scaled)[0]
    probabilidad = modelo_final.predict_proba(X_scaled)[0][1]  # Probabilidad de clase 1 (abandono)

    # Categorizar el nivel de riesgo
    niveles = pd.cut(
        [probabilidad], bins=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],
        include_lowest=True)

    # Generar un ID simulado único (esto es lo que agregas para tus usuarios simulados)
    id_simulado = random.randint(100000, 999999)  # Aquí generas un ID único para el usuario simulado

        # Obtener la importancia de las características por persona
    importancia_por_persona_df = obtener_importancia_por_persona(modelo_final, X_scaled, df.columns)

    # Guardar los resultados de la predicción en CSV (solo como ejemplo, ajusta según necesites)
    guardar_predicciones_api(
        idpersona=id_simulado,  # Aquí usas el ID simulado
        variables=data.dict() if isinstance(data, UserData) else [d.dict() for d in data.datos],
        pred=prediccion,
        prob=probabilidad,
        nivel=niveles[0],
        endpoint="/predecir_abandono_socio_simulado",
        run_id=mejor_modelo_info['run_id'],  # Asegúrate de tener el run_id correspondiente
        importancia_variables=importancia_por_persona_df,  # Si no estás calculando la importancia aquí, ponlo como None
    )

    # Crear el resultado de la predicción
    resultados = [{
        "IdPersona": id_simulado,  # Aquí devolvemos el ID simulado
        "ProbabilidadAbandono": round(probabilidad, 3),
        "NivelRiesgo": niveles[0],
        "CaracterísticasImportantes":importancia_por_persona_df  # Aquí también puedes incluir las características importantes si las calculas
    }]
    
    return resultados if isinstance(data, MultiUserData) else resultados[0]


#FUNCIONA
@app.post("/predecir_abandono_por_id/", summary='Predicción por IdPersona')
def predecir_abandono_por_id(request: IDRequest):
    """
    Endpoint para realizar una predicción de abandono para una persona específica, identificada por su ID.
    
    Parámetros:
        request (IDRequest): Contiene el ID de la persona para la cual se desea hacer la predicción.
            - IdPersona (int): El ID de la persona cuyo abandono se quiere predecir.

    Retorna:
        dict: Resultado de la predicción, incluyendo:
            - IdPersona (int): El ID de la persona.
            - ProbabilidadAbandono (float): Probabilidad de abandono (entre 0 y 1).
            - NivelRiesgo (str): Nivel de riesgo categorizado ("Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto").
            - CaracterísticasImportantes (list): Lista de características importantes con su respectiva importancia.
    
    Lanza:
        HTTPException: Si no se encuentra la persona por ID en el dataset de validación o si hay un error de validación de columnas.
    """

    id_buscar = request.IdPersona
    fila = df_validacion[df_validacion['IdPersona'] == id_buscar]
    
    if fila.empty:
        raise HTTPException(status_code=404, detail=f"IdPersona {id_buscar} no encontrado")

    df = fila[columnas_modelo3].copy()
    # Asegurarse de que las columnas booleanas se conviertan a enteros
    bool_cols = ['Sexo_Mujer', 'UsoServiciosExtra', 'TienePagos', 'TieneAccesos',
                 'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
                 'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
                 'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']
    df[bool_cols] = df[bool_cols].astype(int)

    errores = validar_columnas_esperadas(df, columnas_modelo3)
    if errores:
        raise HTTPException(status_code=400, detail="; ".join(errores))
    X_scaled = scaler.transform(df)

    prediccion = modelo_final.predict(X_scaled)[0]
    probabilidad = modelo_final.predict_proba(X_scaled)[0][1]  #

    
    # Categorizar el nivel de riesgo
    nivel = pd.cut([probabilidad], bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                   labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],
                   include_lowest=True)[0]

    importancia_por_persona_df = obtener_importancia_por_persona(modelo_final, X_scaled, df.columns)

    # Guardar los resultados en CSV
    guardar_predicciones_api(
        idpersona=id_buscar,
        variables=fila.to_dict(orient="records")[0],  # Las variables del ID
        pred=prediccion,
        prob=probabilidad,
        nivel=nivel,
        endpoint="/predecir_abandono_por_id",
        run_id=mejor_modelo_info['run_id'],  # O el run_id correspondiente
        importancia_variables= importancia_por_persona_df,
      
    )
    return {
        "IdPersona": int(id_buscar),
        "ProbabilidadAbandono": round(probabilidad, 3),
        "NivelRiesgo": nivel,
        "CaracterísticasImportantes": importancia_por_persona_df.to_dict(orient="records")  # Devolvemos los resultados de las importancias
    }

#FUNCIONA
@app.post("/predecir_abandono_por_ids/", summary='Predicción por lista de IdPersona')
def predecir_abandono_por_ids(request: IDListRequest):
    """
    Endpoint para realizar predicciones de abandono para una lista de personas, identificadas por sus IDs.
    
    Parámetros:
        request (IDListRequest): Contiene una lista de IDs de las personas para las cuales se desea hacer la predicción.
            - Ids (List[int]): Lista de IDs de las personas cuyos abandonos se quieren predecir.

    Retorna:
        list: Una lista de diccionarios, donde cada diccionario contiene:
            - IdPersona (int): El ID de la persona.
            - ProbabilidadAbandono (float): Probabilidad de abandono (entre 0 y 1).
            - NivelRiesgo (str): Nivel de riesgo categorizado ("Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto").
            - CaracterísticasImportantes (list): Lista de características importantes con su respectiva importancia.
            - error (str, opcional): Si la persona no fue encontrada, se devuelve un mensaje de error.
    
    Lanza:
        HTTPException: Si hay un error de validación de columnas o en el procesamiento de los datos.
    """
    
    resultados = []

    for idpersona in request.Ids:
        fila = df_validacion[df_validacion['IdPersona'] == idpersona]
        if fila.empty:
            resultados.append({
                "IdPersona": idpersona,
                "error": "No encontrado"
            })
            continue

        df = fila[columnas_modelo3].copy()
        # Asegurarse de que las columnas booleanas se conviertan a enteros
        bool_cols = ['Sexo_Mujer', 'UsoServiciosExtra', 'TienePagos', 'TieneAccesos',
                     'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
                     'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
                     'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano']
        df[bool_cols] = df[bool_cols].astype(int)

        errores = validar_columnas_esperadas(df, columnas_modelo3)
        if errores:
            raise HTTPException(status_code=400, detail="; ".join(errores))
        X_scaled = scaler.transform(df)

        prediccion = modelo_final.predict(X_scaled)[0]
        probabilidad = modelo_final.predict_proba(X_scaled)[0][1]  #

        
        # Categorizar el nivel de riesgo
        nivel = pd.cut([probabilidad], bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                       labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],
                       include_lowest=True)[0]
        # Obtener las características más importantes
        importancia_por_persona_df = obtener_importancia_por_persona(modelo_final, X_scaled, df.columns)

                # Guardar los resultados en CSV
        guardar_predicciones_api(
            idpersona=idpersona,
            variables=fila.to_dict(orient="records")[0],  # Las variables del ID
            pred=prediccion,
            prob=probabilidad,
            nivel=nivel,
            endpoint="/predecir_abandono_por_ids",
            run_id=mejor_modelo_info['run_id'],  # O el run_id correspondiente
            importancia_variables= importancia_por_persona_df
         
        )

        resultados.append({
            "IdPersona": idpersona,
            "ProbabilidadAbandono": round(probabilidad, 3),
            "NivelRiesgo": nivel,
            "CaracterísticasImportantes": importancia_por_persona_df.to_dict(orient="records")  # Devolvemos los resultados de las importancias
        })

    return resultados
