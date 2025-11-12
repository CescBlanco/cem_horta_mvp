import streamlit as st
import pandas as pd
import requests
import json
import time
import mlflow
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow import *

mlflow.set_tracking_uri("sqlite:///mlflow.db")

LOGO_GYM= "streamlit/assets/cem_horta-removebg-preview.png"
LOGO_AYUNTAMIENTO = "streamlit/assets/LOGO-AJUNTAMENT.png"

# URL de tu API local
API_URL = "http://localhost:8001"


BOOL_COL = ["Sexo_Mujer", "UsoServiciosExtra", "TienePagos", "TieneAccesos",
                    "DiaFav_domingo", "DiaFav_jueves", "DiaFav_lunes", "DiaFav_martes",
                    "DiaFav_miercoles", "DiaFav_sabado", "DiaFav_viernes",
                    "EstFav_invierno", "EstFav_otono", "EstFav_primavera", "EstFav_verano"
                ]


NAME_EXPERIMENT_3= 'Experimento_v3'   
NAME_EXPERIMENT_2 = 'Experimento_v2'
NAME_EXPERIMENT_1 = 'Experimento_v1'
METRIC= 'auc'


#Ruta donde se guardaran cada artefacto segun el experimento 1 y 2 (no usados)
FOLDER_DESTINO_1 = 'models_mlflow/inferencia_predicciones_exp1'
FOLDER_DESTINO_2 = 'models_mlflow/inferencia_predicciones_exp2'
# Ruta donde se guardarán los artefactos descargados de la inferencia 3 (la importante)
FOLDER_DESTINO_3= 'models_mlflow/inferencia_predicciones_exp3'

def obtener_run_id_inferencias(NAME_EXPERIMENT):

    # Obtener todos los experimentos
    exp = mlflow.get_experiment_by_name(NAME_EXPERIMENT)
    # Obtener todos los runs del experimento
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    run_id_inferencia = runs[runs["tags.type"] == "validacion_externa"].iloc[0]["run_id"]
    return run_id_inferencia


def cargar_columnas_modelo(path: str) -> list:
    """
    Carga un archivo de texto que contiene las columnas de un modelo.

    Args:
        path (str): La ruta del archivo de texto.

    Returns:
        list: Una lista con las columnas leídas del archivo.
        En caso de error, devuelve una lista vacía.
    
    Raises:
        FileNotFoundError: Si no se encuentra el archivo especificado.
    """
    try:

        # Abrir el archivo especificado en modo de solo lectura
        with open(path, 'r') as f:

            # Leer todas las líneas del archivo y eliminarlas de saltos de línea (\n)
            columnas = f.read().splitlines()

         # Retornar las columnas leídas
        return columnas
    
    # Si no se encuentra el archivo, manejar la excepción
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo: {path}")
        # Retornar una lista vacía si no se encuentra el archivo
        return []

COLUMNAS_MODELO = cargar_columnas_modelo('data/columnas_modelo3.txt')


def obtener_predicciones_api(endpoint: str, data: dict) -> dict | None:
    """
    Realiza una solicitud POST a un endpoint de una API para obtener predicciones.

    Args:
        endpoint (str): El endpoint de la API al que se enviará la solicitud.
        data (dict): Los datos que se enviarán en el cuerpo de la solicitud como JSON.

    Returns:
        dict | None: El JSON de la respuesta de la API si la solicitud es exitosa, 
                      o None en caso de error.

    Raises:
        requests.exceptions.RequestException: Si ocurre un error durante la solicitud HTTP.
    """
    try:
        # Realizar una solicitud POST a la API con los datos como JSON
        response = requests.post(f"{API_URL}/{endpoint}", json=data)
        
        # Verificar si la respuesta es exitosa (código 2xx)
        response.raise_for_status()
        
        # Retornar la respuesta de la API en formato JSON
        return response.json()
    
    # Manejar cualquier excepción relacionada con la solicitud HTTP
    except requests.exceptions.RequestException as e:
        # Mostrar un mensaje de error en la interfaz de usuario si ocurre un error
        st.error(f"❌ Error: Identificación del abonado es incorrecto. No existe!" 
                 "\nPor favor ingresar un abonado existente!")
        # Retornar None si ocurre algún error durante la solicitud
        return None


def input_userdata() -> dict:
    """
    Crea un formulario interactivo en Streamlit para capturar datos del usuario.

    Returns:
        dict: Un diccionario con los valores ingresados por el usuario en el formulario.
              Las claves son los nombres de las variables y los valores son los datos proporcionados.
    """

    col1, col2 = st.columns([1, 1])

    with col1:
        # Títulos de las secciones
        st.markdown("<h3 style='color: #888;'>Información Personal</h3>", unsafe_allow_html=True)   
        

        Edad = st.number_input("Edad", min_value=18, max_value=120, value=30)

        #Casillas de selección de las variables.
        Sexo_Mujer = st.checkbox("Sexo Mujer")
        TienePagos = st.checkbox("Tiene Pagos")
        TieneAccesos = st.checkbox("Tiene Accesos")
        
    with col2:
        st.markdown("<h3 style='color: #888;'>Visitas y Actividad</h3>", unsafe_allow_html=True)
        
        # Campo para ingresar números en referencia a las visitas
        TotalVisitas = st.number_input("Total Visitas", min_value=0, value=0)
        DiasActivo = st.number_input("Días Activo", min_value=0, value=0)
        VisitasUlt90 = st.number_input("Visitas Últimos 90 días", min_value=0, value=0)
        VisitasUlt180 = st.number_input("Visitas Últimos 180 días", min_value=0, value=0)
        VisitasPrimerTrimestre = st.number_input("Visitas Primer Trimestre", min_value=0, value=0)
        VisitasUltimoTrimestre = st.number_input("Visitas Último Trimestre", min_value=0, value=0)

    st.markdown("---")  # Línea separadora para organización visual

    st.markdown("<h3 style='color: #888;'>Preferencias y Estilo de Vida</h3>", unsafe_allow_html=True)
    
    # Usamos columnas de nuevo para agrupar preferencias
    col3, col4 = st.columns([1, 1])
    
    # En la primera columna, preferencias relacionadas con estaciones
    with col3:
        EstFav_invierno = st.checkbox("Estación Favorita Invierno")
        EstFav_otono = st.checkbox("Estación Favorita Otoño")
        EstFav_primavera = st.checkbox("Estación Favorita Primavera")
        EstFav_verano = st.checkbox("Estación Favorita Verano")
    
    # En la segunda columna, preferencias sobre días de la semana
    with col4:
        DiaFav_domingo = st.checkbox("Día Favorito Domingo")
        DiaFav_jueves = st.checkbox("Día Favorito Jueves")
        DiaFav_lunes = st.checkbox("Día Favorito Lunes")
        DiaFav_martes = st.checkbox("Día Favorito Martes")
        DiaFav_miercoles = st.checkbox("Día Favorito Miércoles")
        DiaFav_sabado = st.checkbox("Día Favorito Sábado")
        DiaFav_viernes = st.checkbox("Día Favorito Viernes")

    st.markdown("---")  # Línea separadora

    # Título de la sección "Ratio y Diversidad de Servicios"
    st.markdown("<h3 style='color: #888;'>Ratio y Diversidad de Servicios</h3>", unsafe_allow_html=True)

    # Crear dos columnas para mostrar la información sobre servicios
    col5, col6 = st.columns([1, 1])

    # En la primera columna, casilla para indicar si usa servicios extra
    with col5:
        UsoServiciosExtra = st.checkbox("Uso Servicios Extra")

    # En la segunda columna, campos para ingresar ratios y diversidad de servicios    
    with col6:

        # Campo para ingresar el ratio de cantidad entre 2025 y 2024
        ratio_cantidad_2025_2024 = st.number_input("Ratio cantidad 2025/2024", value=1.0, format="%.3f")

        # Campo para ingresar la diversidad de servicios extra
        Diversidad_servicios_extra = st.number_input("Diversidad servicios extra", min_value=0, max_value=100, value=1)


    # Retornar un diccionario con todos los valores capturados del formulario
    return {
        "Edad": Edad,
        "Sexo_Mujer": Sexo_Mujer,
        "UsoServiciosExtra": UsoServiciosExtra,
        "ratio_cantidad_2025_2024": ratio_cantidad_2025_2024,
        "Diversidad_servicios_extra": Diversidad_servicios_extra,
        "TienePagos": TienePagos,
        "TotalVisitas": TotalVisitas,
        "DiasActivo": DiasActivo,
        "VisitasUlt90": VisitasUlt90,
        "VisitasUlt180": VisitasUlt180,
        "TieneAccesos": TieneAccesos,
        "VisitasPrimerTrimestre": VisitasPrimerTrimestre,
        "VisitasUltimoTrimestre": VisitasUltimoTrimestre,
        "DiaFav_domingo": DiaFav_domingo,
        "DiaFav_jueves": DiaFav_jueves,
        "DiaFav_lunes": DiaFav_lunes,
        "DiaFav_martes": DiaFav_martes,
        "DiaFav_miercoles": DiaFav_miercoles,
        "DiaFav_sabado": DiaFav_sabado,
        "DiaFav_viernes": DiaFav_viernes,
        "EstFav_invierno": EstFav_invierno,
        "EstFav_otono": EstFav_otono,
        "EstFav_primavera": EstFav_primavera,
        "EstFav_verano": EstFav_verano,
        
    }

def encontrar_metricas_experimento(NAME_EXPERIMENT: str, metric: str = 'auc') -> tuple[float, float, float, float]:
    """
    Extrae las métricas de un experimento en MLflow, ordenadas por una métrica específica.

    Args:
        NAME_EXPERIMENT (str): El nombre del experimento en MLflow.
        metric (str): La métrica por la que se desea ordenar los resultados. Por defecto es 'auc'.

    Returns:
        tuple: Un tuple con las métricas 'auc', 'accuracy', 'f1_score' y 'recall' redondeadas a 2 decimales.
    
    Raises:
        ValueError: Si no se encuentra el experimento en MLflow.
    """

    # Crear un cliente de MLflow para interactuar con el servidor de experimentos
    client = MlflowClient()

    # Obtener el experimento mediante el nombre del experimento
    experiment = client.get_experiment_by_name(NAME_EXPERIMENT)

    # Si el experimento no existe, lanzar una excepción
    if not experiment:
        raise ValueError(f"No se encontró el experimento {NAME_EXPERIMENT} en MLflow")

    # Buscar las ejecuciones (runs) del experimento y ordenarlas por la métrica especificada
    best_run = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )[0]
    run_id_exp3 = best_run.info.run_id
      
    # Obtener el run completo para obtener todas las métricas asociadas a esa ejecución
    run_exp3= mlflow.get_run(run_id_exp3)

    # Obtener las métricas del run en formato diccionario
    metrics_exp3= run_exp3.data.metrics 

    # Extraer las métricas específicas que nos interesan y redondearlas a 2 decimales
    auc_exp3 = round(metrics_exp3.get('auc', None),2)
    accuracy_exp3 = round(metrics_exp3.get('accuracy', None),2) 
    f1_exp3 = round(metrics_exp3.get('f1_score', None),2)
    recall_exp3 = round(metrics_exp3.get('recall', None),2)

    return auc_exp3, accuracy_exp3, f1_exp3, recall_exp3

def encontrar_csv_inferencias(NAME_EXPERIMENT: str, folder_destino_ex: str, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Encuentra y descarga los archivos CSV de inferencias relacionados con un experimento en MLflow.

        Args:
            NAME_EXPERIMENT (str): El nombre del experimento en MLflow.
            folder_destino_ex (str): El directorio donde se guardarán los archivos CSV descargados.
            run_id (str): El ID del run del cual descargar los artefactos.

        Returns:
            tuple: Tres dataframes correspondientes a los archivos 'importancias_global', 'importancias_persona' y 'preds'.
                Si los archivos no existen o hay un error, se retorna (None, None, None).
        """

    # Verificar si el directorio de destino no existe, si es así, crearlo
    if not os.path.exists(folder_destino_ex):
        os.makedirs(folder_destino_ex)

    try:

        # Obtener el run completo usando el run_id proporcionado
        run_inf = mlflow.get_run(run_id)

        # Obtener el URI de los artefactos (donde se encuentran los archivos generados por el run)
        artifact_uri = run_inf.info.artifact_uri

        # Descargar los artefactos desde el URI obtenido
        artifact_path = mlflow.artifacts.download_artifacts(artifact_uri)

        # Verificar si los artefactos fueron descargados correctamente
        if not os.path.exists(artifact_path):
            st.error(f"No se encontraron artefactos en el path: {artifact_path}")
            return None, None, None

    except Exception as e:
        st.error(f"Error al descargar los artefactos: {e}")
        return None, None, None

    # Si los artefactos fueron descargados correctamente, obtener los archivos en el directorio de artefactos
    archivos_descargados = os.listdir(artifact_path)

    # Si se encontraron archivos descargados, copiarlos al directorio de destino
    if archivos_descargados:
        for archivo in archivos_descargados:
            archivo_origen = os.path.join(artifact_path, archivo)
            archivo_destino = os.path.join(folder_destino_ex, archivo)

            # Copiar cada archivo de su ubicación original al directorio de destino
            shutil.copy(archivo_origen, archivo_destino)
    else:
        st.write("No se encontraron archivos en el directorio de artefactos.")

    # Listar los archivos en la carpeta de destino y filtrar aquellos que sean CSV
    archivos_guardados = os.listdir(folder_destino_ex)
    archivos_csv = [archivo for archivo in archivos_guardados if archivo.endswith('.csv')]

    # Si se encontraron archivos CSV, cargarlos en DataFrames
    if archivos_csv:
        dataframes = {}
        for archivo_csv in archivos_csv:
            nombre_variable = archivo_csv.replace('.csv', '')
            ruta_completa = os.path.join(folder_destino_ex, archivo_csv)

            try:
                # Cargar el archivo CSV en un DataFrame
                df = pd.read_csv(ruta_completa)
                dataframes[nombre_variable] = df
            except Exception as e:
                st.error(f"Error al cargar el archivo {archivo_csv}: {e}")
    else:
        st.write("No se encontraron archivos CSV en la carpeta de destino.")

    # Verificar si las claves existen en el diccionario antes de acceder a ellas
    df_archivo_global = dataframes.get(f'importancias_global_{NAME_EXPERIMENT}', None)
    df_archivo_persona = dataframes.get(f'importancias_persona_{NAME_EXPERIMENT}', None)
    df_archivo_preds = dataframes.get(f'preds_{NAME_EXPERIMENT}', None)

    # Retornar los DataFrames con los archivos CSV de importancias y predicciones
    return df_archivo_global, df_archivo_persona, df_archivo_preds      


def encontrar_metricas_inferencia(run_id: str) -> tuple[float, float, float, float]:
    """
    Obtiene las métricas de un run de MLflow especificado por su `run_id`.

    Esta función extrae las métricas asociadas al `run_id` proporcionado, limpiando las métricas que contienen el prefijo `val_` y retornando las métricas `accuracy`, `auc`, `f1` y `recall` redondeadas a dos decimales. Si alguna de las métricas no está disponible, devuelve `None` para esa métrica.

    Parámetros:
        run_id (str): El identificador único de un "run" en MLflow desde el cual se extraen las métricas.

    Retorna:
        tuple: Una tupla con las métricas `accuracy`, `auc`, `f1` y `recall` redondeadas a dos decimales. 
            Si alguna métrica no está disponible, se devuelve `None` para esa métrica.
            Ejemplo de salida: (0.92, 0.85, 0.78, 0.82)
    """

    # Obtener el 'run' de MLflow mediante el 'run_id' especificado
    run_inf = mlflow.get_run(run_id)
    
    # Acceder al diccionario de métricas almacenado en el 'run'
    metrics = run_inf.data.metrics  

    # Crear un diccionario vacío para almacenar las métricas limpias
    metricas_dict = {}

    # Iterar sobre las métricas y eliminar el prefijo 'val_' de las métricas
    # para que el nombre de la métrica sea válido como variable    
    for metric_name, metric_value in metrics.items():

        # Si el nombre de la métrica empieza con 'val_', eliminamos ese prefijo
        if metric_name.startswith('val_'):
            variable_name = metric_name[4:]  # Eliminar el prefijo 'val_' para que la variable se llame 'accuracy' en lugar de 'val_accuracy'
        else:
            variable_name = metric_name

        # Guardamos el nombre limpio y el valor de la métrica en el diccionario
        metricas_dict[variable_name] = metric_value

    # Redondeamos las métricas a dos decimales
    accuracy = round(metricas_dict.get('accuracy', None),2) 
    auc= round(metricas_dict.get('auc', None),2)  
    f1 = round(metricas_dict.get('f1', None),2)  
    recall = round(metricas_dict.get('recall', None),2)  

    # Retornamos las métricas como una tupla
    return accuracy, auc, f1, recall

def plot_importancias(df_global: pd.DataFrame) -> plt.Figure:
    """
    Genera un gráfico de barras horizontales que muestra las importancias de las características del modelo.

    Esta función crea una visualización de las importancias de las variables a partir de un DataFrame `df_global` donde se espera que haya una columna llamada `Feature` con el nombre de las características y 
    una columna llamada `Importance` con los valores de importancia de cada característica.

    Parámetros:
        df_global (pd.DataFrame): DataFrame que contiene las características y sus importancias. 
                                Debe tener las columnas 'Feature' e 'Importance'.

    Retorna:
        plt.Figure: Una figura de Matplotlib con el gráfico de barras horizontales que muestra las importancias de las variables.
    """

    # Crear una figura y un eje para el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ordenar las características por la importancia en orden descendente
    df_archivo_global= df_global.sort_values(by='Importance', ascending= False)

    # Crear un gráfico de barras horizontales
    ax.barh(df_archivo_global['Feature'], df_archivo_global['Importance'])
    ax.invert_yaxis()  # Invertir el eje Y para que la variable más importante esté arriba

    # Añadir el valor de la importancia a cada barra
    for index, value in enumerate(df_archivo_global['Importance']):
        ax.text(value, index, f'{value:.4f}', va='center', ha='left', size=6, color='white', fontweight='bold')

    # Configurar fondo transparente
    fig.patch.set_facecolor('none')  # Fondo del gráfico transparente
    ax.set_facecolor('none')  # Fondo de la parte del gráfico transparente  

    # Configurar las etiquetas y título del gráfico
    ax.set_xlabel("Importancia", color='white')
    ax.set_title("Top variables por importancia", color='white')
    ax.tick_params(axis='both', colors='white')

    # Retornar la figura del gráfico
    return fig

def plots_experimentos_sinuso(df: pd.DataFrame, variable_importante: str) -> plt.Figure:
    """
    Genera un histograma de la distribución de una variable importante por la categoría de abandono (EsChurn).

    Esta función crea un histograma para la variable especificada (`variable_importante`) en función de las dos categorías de la columna `EsChurn`, que representa si un cliente ha abandonado o no (0 para no abandono, 1 para abandono).

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos de los clientes, que debe incluir las columnas `EsChurn` y la columna correspondiente a `variable_importante`.
        variable_importante (str): El nombre de la columna en el DataFrame para la cual se generará el histograma.

    Retorna:
        plt.Figure: Una figura de Matplotlib con el histograma de la distribución de la variable por abandono.
    """


    # Filtrar los datos para obtener los valores de la variable importante para los clientes que no abandonaron (EsChurn = 0)
    abandono_0 = df[df['EsChurn'] == 0][variable_importante]

    # Filtrar los datos para obtener los valores de la variable importante para los clientes que abandonaron (EsChurn = 1)
    abandono_1 = df[df['EsChurn'] == 1][variable_importante]

    # Crear la figura y el eje del gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear el histograma para los clientes que no abandonaron (EsChurn = 0) con color azul
    ax.hist(abandono_0, bins=50, alpha=0.6, label='abandono 0', color='blue')

    # Crear el histograma para los clientes que abandonaron (EsChurn = 1) con color rojo
    ax.hist(abandono_1, bins=50, alpha=0.6, label='abandono 1', color='red')

    # Configurar las etiquetas y título del gráfico
    ax.set_xlabel(variable_importante)
    ax.set_ylabel('Cantidad de Clientes')
    ax.set_title(f'Distribución de {variable_importante} por abandono')

    # Mostrar la leyenda del gráfico
    ax.legend()

    # Activar la cuadrícula en el gráfico
    ax.grid(True)

    # Retornar la figura del histograma
    return fig

def piechart_edad(df: pd.DataFrame) -> plt.Figure:
    """
    Genera un gráfico de torta (pie chart) que muestra la distribución de los niveles de riesgo en un DataFrame.

    Esta función recibe un DataFrame que contiene una columna llamada `nivel_riesgo`, y crea un gráfico de torta mostrando la distribución de los niveles de riesgo.
      Los niveles de riesgo se reordenan para asegurar que se sigan los niveles de menor a mayor: "Muy bajo", "Bajo", "Medio", "Alto", "Muy Alto".

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene la columna `nivel_riesgo` con los diferentes niveles de riesgo.

    Retorna:
        plt.Figure: Una figura de Matplotlib con el gráfico de torta mostrando la distribución de los niveles de riesgo.
    """

    # Obtener la cuenta de los valores únicos de 'nivel_riesgo' y reordenarlos según los niveles establecidos
    nivel_riesgo_counts = df["nivel_riesgo"].value_counts().reindex(["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"])  

    # Crear el gráfico de torta (pie chart)
    fig, ax = plt.subplots(figsize=(6, 6))

    # Crear el gráfico de torta con los valores de 'nivel_riesgo_counts'
    nivel_riesgo_counts.plot(
        kind="pie",
        autopct="%1.1f%%",      # Mostrar porcentaje con un decimal
        ax=ax,
        colors=sns.color_palette("Greens", 5),  # Paleta de colores
        startangle=90,          # Rotar el gráfico para que empiece desde un ángulo de 90 grados
        legend=False ,           # No mostrar la leyenda
        labels=nivel_riesgo_counts.index  # Asegúrate de que las etiquetas estén correctas
    )

    # Personalizar el gráfico
    ax.set_ylabel("", color='white')  # Quitar la etiqueta en el eje y
    fig.patch.set_facecolor('none')  # Fondo del gráfico transparente
    ax.set_facecolor('none')  # Fondo de la parte del gráfico transparente  

    # Cambiar el color de las etiquetas de porcentaje a blanco con un fondo negro semi-transparente
    for label in ax.texts:
        label.set_color('white')
    for i, label in enumerate(ax.texts):
        if '%' in label.get_text():  # Solo aplicar el cambio a las etiquetas de porcentaje
            # Establecer el color de la etiqueta de porcentaje a blanco
            label.set_bbox(dict(facecolor='black', alpha=0.7, edgecolor='none'))  # Fondo negro semi-transparente

    # Retornar la figura con el gráfico de torta
    return fig

def tabla_recuento_resultados(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera dos tablas que muestran el recuento de clientes y el promedio de probabilidad de abandono por grupo de riesgo.

    Esta función crea dos tablas a partir de un DataFrame que contiene los datos de clientes, con columnas `y_true` (si el cliente abandonó o no) y `y_prob` (probabilidad de abandono).
      Las tablas se dividen en dos grupos: "Activo" y "Abandonado", y se calculan el número de clientes y el promedio de probabilidad de abandono para cada nivel de riesgo.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene las columnas `y_true` (valor booleano que indica si el cliente abandonó) y `y_prob` (probabilidad de abandono).

    Retorna:
        tuple: Una tupla con dos DataFrames:
            - El primer DataFrame contiene los datos para clientes "Activos".
            - El segundo DataFrame contiene los datos para clientes "Abandonados".
    """

    # Crear una nueva columna 'estado' que mapea 'y_true' a 'Activo' (False) o 'Abandonado' (True)
    df['estado'] = df['y_true'].map({False: 'Activo', True: 'Abandonado'})

    # Definir el orden correcto de los niveles de riesgo
    orden_riesgo = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]

    # Convertir 'nivel_riesgo' a un tipo categórico con un orden específico
    df['nivel_riesgo'] = pd.Categorical(df['nivel_riesgo'], categories=orden_riesgo, ordered=True)

    # Filtrar datos por estado
    df_activos = df[df['estado'] == 'Activo']
    df_abandonados = df[df['estado'] == 'Abandonado']

    # Agrupar los datos de los clientes activos por 'nivel_riesgo' y calcular el recuento de clientes y promedio de probabilidad de abandono
    grouped_activos = df_activos.groupby('nivel_riesgo').agg(
        abonados=('y_true', 'size'),  # Recuento de abonados por grupo
        promedio_probabilidad=('y_prob', 'mean')  # Promedio de probabilidad de abandono por grupo
    ).reset_index()

    # Agrupar los datos de los clientes abandonados por 'nivel_riesgo' y calcular el recuento de clientes y promedio de probabilidad de abandono
    grouped_abandonados = df_abandonados.groupby('nivel_riesgo').agg(
        abonados=('y_true', 'size'),  # Recuento de abonados por grupo
        promedio_probabilidad=('y_prob', 'mean')  # Promedio de probabilidad de abandono por grupo
    ).reset_index()

    # Renombrar las columnas para hacerlas más comprensibles
    grouped_activos.rename(columns={
        'nivel_riesgo': 'Grupo de Riesgo',
        'abonados': 'Nº Clientes Abandonados',
        'promedio_probabilidad': 'Promedio % de Abandono'
    }, inplace=True)

    grouped_abandonados.rename(columns={
        'nivel_riesgo': 'Grupo de Riesgo',
        'abonados': 'Nº Clientes Abandonados',
        'promedio_probabilidad': 'Promedio % de Abandono'
    }, inplace=True)

    # Eliminar los índices antes de pasarlos a la interfaz de usuario
    grouped_activos_reset = grouped_activos.reset_index(drop=True)
    grouped_abandonados_reset = grouped_abandonados.reset_index(drop=True)

    # Retornar las dos tablas con los resultados
    return grouped_activos_reset, grouped_abandonados_reset

def categorizacion_variables_importancia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la categorización de varias variables en un DataFrame.

    Esta función toma un DataFrame y crea nuevas columnas categóricas para representar los diferentes grupos de cada variable. 
    Las variables categorizadas incluyen `Edad_inicial`, `TotalVisitas_inicial`, `DiasActivo_inicial`, `VisitasUlt90_inicial`, `VisitasUlt180_inicial`, `VisitasPrimerTrimestre_inicial`, y `VisitasUltimoTrimestre_inicial`.

    La función utiliza `pd.cut` para categorizar la edad en grupos de rango de edad, y `pd.qcut` para dividir las otras variables en cuartiles.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene las columnas numéricas que se van a categorizar.

    Retorna:
        pd.DataFrame: El DataFrame original con nuevas columnas categóricas añadidas.
    """

     # Crear grupos de edad
    df["GrupoEdad"] = pd.cut(df["Edad_inicial"],bins=[18, 25, 35, 45, 55, 65, 80, df["Edad_inicial"].max()],
                    labels=["18–25", "26–35", "36–45", "46–55", "56–65", "66–80", "80+"],include_lowest=True)        
           
    # Dividir TotalVisitas en cuartiles
    df['TotalVisitas_categoria'] = pd.qcut(df['TotalVisitas_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])

    # Dividir DiasActivo en cuartiles
    df['DiasActivo_categoria'] = pd.qcut(df['DiasActivo_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])
    
    # Dividir VisitasUlt90 en cuartiles
    df['VisitasUlt90_categoria'] = pd.qcut(df['VisitasUlt90_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])

    # Dividir VisitasUlt180 en cuartiles
    df['VisitasUlt180_categoria'] = pd.qcut(df['VisitasUlt180_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])
                
    # Dividir VisitasPrimerTrimestre en cuartiles
    df['VisitasPrimerTrimestre_categoria'] = pd.qcut(df['VisitasPrimerTrimestre_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])

    # Dividir VisitasUltimoTrimestre en cuartiles
    df['VisitasUltimoTrimestre_categoria'] = pd.qcut(df['VisitasUltimoTrimestre_inicial'], q=4, labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])

    # Retornar el DataFrame con las nuevas categorías
    return df

def box_plot(df: pd.DataFrame, variable_x: str, variable_y: str, x_label: str) -> plt.Figure:
    """
    Genera un gráfico de caja (box plot) para visualizar la distribución de una variable continua (variable_y) 
    en función de una variable categórica (variable_x).

    Esta función crea un gráfico de caja utilizando `seaborn.boxplot`, donde el eje X representa la variable categórica 
    (y la variable_y representa la variable continua). Los valores atípicos se marcan con círculos rojos, y se utiliza 
    una paleta de colores "Greens" para el gráfico.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene las variables a graficar.
        variable_x (str): Nombre de la columna categórica en el DataFrame para el eje X.
        variable_y (str): Nombre de la columna continua en el DataFrame para el eje Y.
        x_label (str): Etiqueta que se mostrará en el eje X.

    Retorna:
        plt.Figure: Un gráfico de caja de Matplotlib mostrando la distribución de la variable continua por categoría.
    """

    # Crear una figura y un eje para el gráfico de caja
    fig, ax = plt.subplots(figsize=(10, 6))

    # Establecer los parámetros de los valores atípicos (flierprops)
    flierprops = dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none')

    # Crear el gráfico de caja utilizando seaborn
    sns.boxplot(x=variable_x, y=variable_y, data=df, palette='Greens', color='white',
                linewidth=2.8,linecolor='grey', flierprops=flierprops, ax=ax)
    
    # Etiquetar el eje X e Y con el texto proporcionado
    plt.xlabel(x_label, color= 'white')
    plt.ylabel("Probabilidad de abandono", color='white')

    # Activar la cuadrícula en el gráfico con líneas blancas
    plt.grid(True, color= 'white')

    # Configurar el fondo transparente del gráfico
    fig.patch.set_facecolor('none')  # Fondo del gráfico transparente
    ax.set_facecolor('none')  # Fondo de la parte del gráfico transparente  

    # Cambiar el color de las etiquetas de los ejes a blanco
    ax.tick_params(axis='both', colors='white')

    # Retornar la figura del gráfico
    return fig


def mostrar_grafico_y_descripcion(eleccion: str, df: pd.DataFrame) -> None:
    """
    Muestra un gráfico de caja (box plot) y proporciona un análisis técnico e interpretación para el negocio 
    en función de la elección del usuario.

    Esta función toma una elección que define qué gráfico y análisis mostrar. Dependiendo de la opción seleccionada, 
    se generará un gráfico de caja para mostrar la distribución de la probabilidad de abandono en diferentes grupos 
    (categorías) y luego se presenta una interpretación técnica y estratégica.

    Parámetros:
        eleccion (str): Cadena que especifica qué gráfico y análisis mostrar. Las opciones incluyen:
                        - "Probabilidad de Abandono por Grupos de Edad"
                        - "Probabilidad de Abandono por Grupos de Días Activos"
                        - "Probabilidad de Abandono por Grupos de Visitas Últimos 180 Días"
                        - "Probabilidad de Abandono por Visitas Primer Trimestre"
                        - "Probabilidad de Abandono por Estación Favorita Otoño"
                        - "Probabilidad de Abandono por si Tiene Pagos"
        df (pd.DataFrame): DataFrame que contiene los datos a graficar y analizar.

    Retorna:
        None: La función no retorna nada. Solo muestra el gráfico y el análisis correspondiente en Streamlit.
    """

    # Comprobamos la elección y generamos el gráfico y análisis correspondiente
    if eleccion == "Probabilidad de Abandono por Grupos de Edad":

        # Generar gráfico de caja para la probabilidad de abandono por grupos de edad
        fig_edad = box_plot(df, "GrupoEdad", "y_prob", "Grupos de Edad")
        st.pyplot(fig_edad)
       
        # Descripción del análisis técnico y la interpretación para el negocio
      
        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:

            - **`Jóvenes (18-35 años)`**: Mayor probabilidad de abandono.
            - **`Adultos mayores (66-80 y 80+)`**: Menor probabilidad de abandono.
            - **`Outliers`**: Grupo de 66-80 tiene comportamientos extremos de abandono.

            **💼📈 Interpretación para el Negocio**:

            - **`Jóvenes`**: Se deben implementar estrategias de retención específicas para este segmento (mejorar experiencia, promociones, etc.).
            - **`Mayores`**: Los usuarios de más edad parecen más comprometidos; mantener y mejorar la retención de este grupo es clave.
            """)

    elif eleccion == "Probabilidad de Abandono por Grupos de Días Activos":
        
        # Generar gráfico de caja para la probabilidad de abandono por grupos de días activos    
        fig_activos = box_plot(df, "DiasActivo_categoria", "y_prob", "Grupos de dias activos")
        st.pyplot(fig_activos)

        # Descripción del análisis técnico y la interpretación para el negocio
        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
            
            - **`Usuarios Activos`**:  Menor probabilidad de abandono.
            - **`Outliers`**:  Algunos usuarios muy activos todavía tienen alta probabilidad de abandono.

            
            **💼📈 Interpretación para el Negocio**:
                    
            - **`Más actividad`**: Fomentar la actividad continua reduce el abandono.
            - **`Estrategia`**: Enfocar recursos en mantener a los usuarios activos (notificaciones, recompensas, ofertas).
            """)
        
        
    elif eleccion == "Probabilidad de Abandono por Grupos de Visitas Últimos 180 Días":

        # Generar gráfico de caja para la probabilidad de abandono por grupos de visitas últimos 180 días
        fig_ultim180 = box_plot(df, "VisitasUlt180_categoria", "y_prob", "Grupos de visitas últimos 180 días")
        st.pyplot(fig_ultim180)

        # Descripción del análisis técnico y la interpretación para el negocio
        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
                    
            - **`Usuarios con más visitas`**: Menor probabilidad de abandono.
            - **`Usuarios con pocas visitas`**: Mayor probabilidad de abandono.
            
            
            **💼📈 Interpretación para el Negocio**:
                    
            - **`Los outliers`** también son notables, especialmente en el grupo "Alto", lo que sugiere que, aunque un número considerable de usuarios con muchas visitas tiene una baja probabilidad de abandono, algunos casos se comportan de manera diferente.
            """)
        
    elif eleccion == "Probabilidad de Abandono por Visitas Primer Trimestre":
    
        # Generar gráfico de caja para la probabilidad de abandono por visitas primer trimestre
        fig_primertrim = box_plot(df, "VisitasPrimerTrimestre_categoria", "y_prob", "Grupo de Visitas Primer Trimestre")
        st.pyplot(fig_primertrim)

        # Descripción del análisis técnico y la interpretación para el negocio
        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
                    
            - **`Usuarios con más visitas en el primer trimestre`**: Menor probabilidad de abandono.
            - **`Outliers`**: Algunos usuarios con muchas visitas siguen teniendo una alta probabilidad de abandono. Casos fuera del comportamiento (distribución de los datos)         
                
            **💼📈 Interpretación para el Negocio**:
                    
            - **`Visitas tempranas`**: Las visitas durante el primer trimestre son un buen predictor de retención a largo plazo.
            - **`Estrategia`**: Incentivar visitas frecuentes en los primeros meses (promociones para nuevos usuarios).
            """)
    
    elif eleccion =="Probabilidad de Abandono por Estación Favorita Otoño":

        # Generar gráfico de caja para la probabilidad de abandono por estación favorita Otoño
        fig_otoño = box_plot(df, "EstFav_otono_inicial", "y_prob", "Estación favorita Otoño")
        st.pyplot(fig_otoño) 

        # Descripción del análisis técnico y la interpretación para el negocio
        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
                    
            - **`Usuarios con Otoño como estación favorita`**: Levelemente más alta probabilidad de abandono.
            - **`Outliers`**: Algunas desviaciones de comportamiento entre usuarios con y sin Otoño como favorita.         
                
            **💼📈 Interpretación para el Negocio**:
                    
            - **`Estación menos significativa`**: La estación favorita tiene un impacto menor en la retención.
            - **`Oportunidad de segmentación`**: A pesar de su menor impacto, podría usarse para campañas personalizadas o promociones estacionales. 
            """)
    
    elif eleccion =="Probabilidad de Abandono por si Tiene Pagos":

        # Generar gráfico de caja para la probabilidad de abandono por si tiene pagos
        fig_pagos = box_plot(df, "TienePagos_inicial", "y_prob", "Tiene Pagos")
        st.pyplot(fig_pagos)

        # Descripción del análisis técnico y la interpretación para el negocio
        st.markdown("""
            **🧑‍💻📊 Análisis Técnico**:
            
            - **`Usuarios que pagan`**:  Mucha menor probabilidad de abandono.
            - **`Usuarios que no pagan`**:  Alta probabilidad de abandono (casi 80%).
            
            **💼📈 Interpretación para el Negocio**:
    
            - **`Usuarios pagos`**:Son mucho más valiosos y están más comprometidos.
            - **`Estrategia`**: Focalizarse en convertir usuarios gratuitos a pagos y retener a los clientes de pago a través de una mejor experiencia, soporte personalizado, y programas de fidelización.
            """)

# Diccionario de estrategias
ESTRATEGIAS_FIDELIZACION = {
    "Muy Bajo": ["""
        1. **`Programa de recompensas por uso continuo`**: Implementar un sistema de puntos para los usuarios frecuentes, que puedan canjear por descuentos, contenido exclusivo o productos premium.
        2. **`Acceso anticipado a nuevas funcionalidades`**: Los usuarios más activos y pagos pueden ser invitados a probar nuevas funciones antes que el resto de los usuarios. Esto crea un sentido de exclusividad.
        3. **`Beneficios por referencia`**: Ofrecer recompensas por recomendar la plataforma a amigos o colegas. Esto podría ser un mes gratis o un descuento para ambos (referente y referido).
        4. **`Ofertas personalizadas para el perfil de uso`**: Ofrecer descuentos o beneficios exclusivos basados en el comportamiento de uso del cliente. Ejemplo: si un usuario siempre usa una funcionalidad específica, enviarle ofertas relacionadas con esa funcionalidad.
        5. **`Eventos exclusivos en línea`**: Organizar eventos exclusivos como webinars o reuniones virtuales con expertos, donde solo los usuarios activos o pagos puedan participar.
        """],
    "Bajo": ["""
        1. **`Descuentos en renovación de suscripción`**: Ofrecer descuentos significativos si renuevan su suscripción o realizan pagos adicionales dentro de un corto periodo de tiempo
        2. **`Campañas de retargeting personalizado`**: Utilizar datos de comportamiento para ofrecerles promociones o contenidos personalizados que los inviten a retomar la actividad en la plataforma.
        3. **`Notificaciones personalizadas con ofertas de valor`**: Enviar recordatorios de productos o funciones que han utilizado previamente, junto con ofertas especiales (ejemplo: "Vuelve y consigue 10% de descuento en tu próxima compra").
        4. **`Descuentos por uso frecuente`**: Ofrecer descuentos o recompensas para aquellos usuarios que incrementen su actividad durante el mes (por ejemplo, si usan la plataforma 10 días consecutivos, obtienen un descuento del 15%).
        5. **`Recompensas por interacción con nuevas funciones`**: Incentivar a los usuarios a explorar nuevas características de la plataforma ofreciendo un beneficio como un mes adicional de suscripción o puntos de recompensa.
            """],

    "Medio": ["""
        1. **`Ofertas de reactivación personalizadas`**: Enviar un correo o notificación push ofreciendo un descuento importante o acceso a contenido exclusivo si regresan a la plataforma dentro de un plazo determinado.
        2. **`Recordatorio de funcionalidades no utilizadas`**: Utilizar los datos de comportamiento para enviar mensajes recordando las funcionalidades que no han sido exploradas por el usuario, ofreciendo tutoriales o guías rápidas.
        3. **`Campañas de contenido exclusivo para inactivos`**: Crear un catálogo de contenido exclusivo (tutoriales, seminarios web, o artículos premium) disponible solo para aquellos usuarios que regresen después de un periodo de inactividad.
        4. **`Ofrecer acceso a nuevas funcionalidades por tiempo limitado`**: Probar nuevas características de la plataforma de forma gratuita por un tiempo limitado a usuarios que han estado inactivos durante cierto periodo.
        5. **`Notificaciones de "última oportunidad"`**: Enviar un correo con un asunto como “Última oportunidad para obtener tus beneficios exclusivos”, creando un sentido de urgencia.
            """],

    "Alto": ["""
        1. **`Descuentos en el primer pago`**: Ofrecer descuentos agresivos o promociones de "primer pago gratis" si el usuario completa la conversión de gratuito a pago (por ejemplo, "Obtén un mes gratis si te suscribes ahora").
        2. **`Llamadas de atención personalizadas`**: Contactar directamente con estos usuarios a través de soporte al cliente o ventas para entender las razones de su baja actividad y ofrecer una solución personalizada (por ejemplo, “¿Te gustaría una sesión de asesoramiento para mejorar tu experiencia?”).
        3. **`Oferta de planes flexibles o a medida`**: Crear opciones de pago más flexibles o planes personalizados según el uso que hacen los usuarios. Ofrecer un “plan básico” para que comiencen a pagar a bajo costo.
        4. **`Campañas de reactivación urgente`**: Ofrecer grandes descuentos (como un 70% de descuento por tres meses) o beneficios adicionales si reactivan su cuenta dentro de las próximas 24 horas.
        5. **`Ofrecer sesiones de soporte o consultoría gratuita`**: Ofrecer sesiones gratuitas con un experto para guiar a los usuarios sobre cómo sacar el máximo provecho de la plataforma.
            """],

    "Muy Alto": ["""
        1. **`Campañas de recuperación con descuentos masivos`**: Ofrecer un descuento profundo como "90% de descuento en el primer mes si te suscribes ahora", para atraerlos a volver, aunque solo sea para probar la plataforma nuevamente.
        2. **`Encuestas de salida con incentivos`**: Enviar encuestas de salida con una recompensa por completarlas (por ejemplo, “dinos por qué te vas y recibe un 50% de descuento en tu próxima compra”).
        3. **`Planes gratuitos por tiempo limitado`**: Ofrecer un acceso completo y gratuito por 1 mes a todos los servicios premium, con la intención de engancharlos nuevamente a la plataforma.
        4. **`Comunicación directa de recuperación (SMS o Llamada)`**: Si es posible, contactar directamente con el usuario por teléfono o SMS para entender por qué no se están comprometiendo y ofrecer una oferta personalizada.
        5. **`Experiencia de onboarding personalizada`**: Crear una experiencia de reactivación guiada, con contenido paso a paso para que el usuario vuelva a usar la plataforma, mostrando cómo resolver sus puntos de dolor de manera efectiva.
        """]
}

def mostrar_estrategias(nivel_riesgo: str) -> None:
    
    """
    Muestra un conjunto de estrategias de fidelización basadas en el nivel de riesgo de abandono del usuario.

    Dependiendo del nivel de riesgo (Muy Bajo, Bajo, Medio, Alto, Muy Alto), la función presenta un conjunto
    de estrategias personalizadas para retener a los usuarios y reducir la probabilidad de abandono.

    Parámetros:
        nivel_riesgo (str): Cadena que especifica el nivel de riesgo del usuario. Las opciones válidas son:
                            - "Muy Bajo"
                            - "Bajo"
                            - "Medio"
                            - "Alto"
                            - "Muy Alto"

    Retorna:
        None: La función no retorna ningún valor. Solo muestra las estrategias correspondientes en Streamlit.
    """

    # Diccionario con estrategias para cada nivel de riesgo
    estrategias = {
        "Muy Bajo": """
            1. **`Programa de recompensas por uso continuo`**: Implementar un sistema de puntos para los usuarios frecuentes, que puedan canjear por descuentos, contenido exclusivo o productos premium.
            2. **`Acceso anticipado a nuevas funcionalidades`**: Los usuarios más activos y pagos pueden ser invitados a probar nuevas funciones antes que el resto de los usuarios. Esto crea un sentido de exclusividad.
            3. **`Beneficios por referencia`**: Ofrecer recompensas por recomendar la plataforma a amigos o colegas. Esto podría ser un mes gratis o un descuento para ambos (referente y referido).
            4. **`Ofertas personalizadas para el perfil de uso`**: Ofrecer descuentos o beneficios exclusivos basados en el comportamiento de uso del cliente. Ejemplo: si un usuario siempre usa una funcionalidad específica, enviarle ofertas relacionadas con esa funcionalidad.
            5. **`Eventos exclusivos en línea`**: Organizar eventos exclusivos como webinars o reuniones virtuales con expertos, donde solo los usuarios activos o pagos puedan participar.
        """,

        "Bajo": """
            1. **`Descuentos en renovación de suscripción`**: Ofrecer descuentos significativos si renuevan su suscripción o realizan pagos adicionales dentro de un corto periodo de tiempo
            2. **`Campañas de retargeting personalizado`**: Utilizar datos de comportamiento para ofrecerles promociones o contenidos personalizados que los inviten a retomar la actividad en la plataforma.
            3. **`Notificaciones personalizadas con ofertas de valor`**: Enviar recordatorios de productos o funciones que han utilizado previamente, junto con ofertas especiales (ejemplo: "Vuelve y consigue 10% de descuento en tu próxima compra").
            4. **`Descuentos por uso frecuente`**: Ofrecer descuentos o recompensas para aquellos usuarios que incrementen su actividad durante el mes (por ejemplo, si usan la plataforma 10 días consecutivos, obtienen un descuento del 15%).
            5. **`Recompensas por interacción con nuevas funciones`**: Incentivar a los usuarios a explorar nuevas características de la plataforma ofreciendo un beneficio como un mes adicional de suscripción o puntos de recompensa.

        """,
        "Medio": """
            1. **`Ofertas de reactivación personalizadas`**: Enviar un correo o notificación push ofreciendo un descuento importante o acceso a contenido exclusivo si regresan a la plataforma dentro de un plazo determinado.
            2. **`Recordatorio de funcionalidades no utilizadas`**: Utilizar los datos de comportamiento para enviar mensajes recordando las funcionalidades que no han sido exploradas por el usuario, ofreciendo tutoriales o guías rápidas.
            3. **`Campañas de contenido exclusivo para inactivos`**: Crear un catálogo de contenido exclusivo (tutoriales, seminarios web, o artículos premium) disponible solo para aquellos usuarios que regresen después de un periodo de inactividad.
            4. **`Ofrecer acceso a nuevas funcionalidades por tiempo limitado`**: Probar nuevas características de la plataforma de forma gratuita por un tiempo limitado a usuarios que han estado inactivos durante cierto periodo.
            5. **`Notificaciones de "última oportunidad"`**: Enviar un correo con un asunto como “Última oportunidad para obtener tus beneficios exclusivos”, creando un sentido de urgencia.
        """,
        "Alto": """
            1. **`Descuentos en el primer pago`**: Ofrecer descuentos agresivos o promociones de "primer pago gratis" si el usuario completa la conversión de gratuito a pago (por ejemplo, "Obtén un mes gratis si te suscribes ahora").
            2. **`Llamadas de atención personalizadas`**: Contactar directamente con estos usuarios a través de soporte al cliente o ventas para entender las razones de su baja actividad y ofrecer una solución personalizada (por ejemplo, “¿Te gustaría una sesión de asesoramiento para mejorar tu experiencia?”).
            3. **`Oferta de planes flexibles o a medida`**: Crear opciones de pago más flexibles o planes personalizados según el uso que hacen los usuarios. Ofrecer un “plan básico” para que comiencen a pagar a bajo costo.
            4. **`Campañas de reactivación urgente`**: Ofrecer grandes descuentos (como un 70% de descuento por tres meses) o beneficios adicionales si reactivan su cuenta dentro de las próximas 24 horas.
            5. **`Ofrecer sesiones de soporte o consultoría gratuita`**: Ofrecer sesiones gratuitas con un experto para guiar a los usuarios sobre cómo sacar el máximo provecho de la plataforma.
        """,
        "Muy Alto": """
            1. **`Campañas de recuperación con descuentos masivos`**: Ofrecer un descuento profundo como "90% de descuento en el primer mes si te suscribes ahora", para atraerlos a volver, aunque solo sea para probar la plataforma nuevamente.
            2. **`Encuestas de salida con incentivos`**: Enviar encuestas de salida con una recompensa por completarlas (por ejemplo, “dinos por qué te vas y recibe un 50% de descuento en tu próxima compra”).
            3. **`Planes gratuitos por tiempo limitado`**: Ofrecer un acceso completo y gratuito por 1 mes a todos los servicios premium, con la intención de engancharlos nuevamente a la plataforma.
            4. **`Comunicación directa de recuperación (SMS o Llamada)`**: Si es posible, contactar directamente con el usuario por teléfono o SMS para entender por qué no se están comprometiendo y ofrecer una oferta personalizada.
            5. **`Experiencia de onboarding personalizada`**: Crear una experiencia de reactivación guiada, con contenido paso a paso para que el usuario vuelva a usar la plataforma, mostrando cómo resolver sus puntos de dolor de manera efectiva.
        """
    }

    # Verificar si el nivel de riesgo está en el diccionario y mostrar las estrategias correspondientes
    if nivel_riesgo in estrategias:

        # Mostrar las estrategias en un 'expander' de Streamlit
        with st.expander(f"Estrategias de fidelización para **{nivel_riesgo}**"):
            st.markdown(estrategias[nivel_riesgo])


def color_con_riesgo(probabilidad: float) -> tuple:
        
    """
    Devuelve un color y nivel de riesgo según la probabilidad de abandono.

    La función evalúa la probabilidad de abandono y la clasifica en un nivel de riesgo. Además, devuelve un color 
    asociado a ese nivel de riesgo para su visualización, útil en gráficos u otros informes visuales.

    Parámetros:
        probabilidad (float): Valor de la probabilidad de abandono, que debe estar en el rango de [0, 1]. 
                            - `0` representa la mínima probabilidad de abandono.
                            - `1` representa la máxima probabilidad de abandono.

    Retorna:
        tuple: Una tupla que contiene dos valores:
            - `color (str)`: Código hexadecimal del color asociado al nivel de riesgo.
            - `nivel (str)`: El nivel de riesgo correspondiente a la probabilidad de abandono.
    """

    # Determinar nivel de riesgo según la probabilidad
    if probabilidad <= 0.2:
        nivel = "Muy Bajo"
    elif probabilidad <= 0.4:
        nivel = "Bajo"
    elif probabilidad <= 0.6:
        nivel = "Medio"
    elif probabilidad <= 0.8:
        nivel = "Alto"
    else:
        nivel = "Muy Alto"

    # Definir los colores asociados a cada nivel de riesgo
    colores_riesgo = {
        "Muy Bajo": "#2ca02c",
        "Bajo": "#98df8a",
        "Medio": "#ffcc00",
        "Alto": "#ff7f0e",
        "Muy Alto": "#d62728"
        }
    
    # Obtener el color asociado al nivel de riesgo
    color = colores_riesgo[nivel]

    # Retornar el color y el nivel de riesgo
    return color, nivel

def preparar_df_importancias(response: dict) -> pd.DataFrame:
    """
    Esta función recibe un diccionario `response` que contiene la clave 
    'CaracterísticasImportantes'. Extrae las variables más importantes, las ordena por valor absoluto 
    y devuelve el top 10 de las variables con mayor importancia.

    Parámetros:
        response (dict): Diccionario con las características y sus importancias.
                        Debe contener la clave 'CaracterísticasImportantes', que es un diccionario con las variables y sus importancias.

    Retorna:
        pd.DataFrame: DataFrame con el top 10 de las variables más importantes ordenadas por valor absoluto de importancia.
    """

    
    # Verificar si la clave 'CaracterísticasImportantes' existe en la respuesta
    if "CaracterísticasImportantes" not in response:
        raise ValueError("La clave 'CaracterísticasImportantes' no se encuentra en la respuesta.")

    # Convertir a DataFrame
    df_importancias = pd.DataFrame(response["CaracterísticasImportantes"])

    # Filtrar las columnas que contienen '_importance' en su nombre (importancia de cada variable)
    df_importance = df_importancias[[col for col in df_importancias.columns if "_importance" in col]]
    
    # Renombrar las columnas para eliminar '_importance' de los nombres de las variables
    df_importance.columns = [col.replace("_importance", "") for col in df_importance.columns]

    # Transponer el DataFrame para convertir las variables en filas y las importancias en valores
    df_top = df_importance.T.reset_index()
    df_top.columns = ["Variable", "Valor"]
    
    # Ordenar el DataFrame por el valor absoluto de la columna 'Valor' de mayor a menor
    df_top = df_top.sort_values(by="Valor", ascending=False)

    # Filtrar el top 10 por valor absoluto (asegurando que se tomen las 10 más importantes)
    df_top_filtered = df_top.reindex(df_top["Valor"].abs().sort_values(ascending=False).index).head(10)

    # Filtrar el top 10 por valor absoluto (asegurando que se tomen las 10 más importantes)
    return df_top_filtered

def plot_abonado_importancias(df: pd.DataFrame) -> plt.Figure:
    """
    Genera un gráfico de barras horizontales para visualizar las variables más importantes para un abonado 
    y su impacto en el riesgo de abandono. Las barras son coloreadas en rojo (positivo) o verde (negativo), 
    y se muestra el valor de cada barra en la gráfica.

    Parámetros:
        df (pd.DataFrame): DataFrame con las variables más importantes y sus valores de importancia.
                            Debe tener las columnas "Variable" y "Valor", donde "Valor" es la importancia de cada variable.

    Retorna:
        plt.Figure: Objeto de la figura del gráfico generado.
    """

    # Asignar colores dependiendo de si el valor de la importancia es positivo o negativo
    colors = ['red' if v > 0 else 'green' for v in df["Valor"]]
    
    # Crear la figura y el eje para el gráfico
    fig, ax = plt.subplots(figsize=(8, 5))

    # Crear el gráfico de barras horizontales
    ax.barh(df["Variable"], df["Valor"], color=colors)

    # Añadir una línea vertical en x=0 para separar los valores positivos de los negativos
    ax.axvline(x=0, color='white', linestyle='--', linewidth=1)

    # Etiquetas y título del gráfico
    ax.set_xlabel("Impacto en riesgo", color='white')
    ax.set_title("Variables más influyentes para este abonado", color='white')

    # Configurar fondo transparente
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # Ajustar el diseño del gráfico para que se vea bien
    plt.tight_layout()

    # Configurar la rotación de los nombres de las variables y el color de las etiquetas
    plt.yticks(rotation=0, color='white')

    # Configurar los colores de los ejes
    ax.tick_params(axis='x', colors='white')

    # Añadir los valores de cada barra en el gráfico
    for i, v in enumerate(df["Valor"]):
        ha = 'left' if v > 0 else 'right' # Posicionar la etiqueta dependiendo si el valor es positivo o negativo
        xpos = v + (0.0005 if v > 0 else 0.02) # Ajustar la posición de la etiqueta
        ax.text(
            xpos, i, f"{v:.2f}", color='black', va='center', ha=ha,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

    # Retornar el objeto de la figura
    return fig

def generar_frase_resumen(df: pd.DataFrame, nivel_riesgo: str) -> str:
    """
    Genera una frase de resumen sobre el riesgo de abandono basado en las variables más influyentes.
    La frase resume las variables que aumentan y disminuyen el riesgo, y ajusta el mensaje según el nivel de riesgo global.

    Parámetros:
        df_top_filtered (pd.DataFrame): DataFrame con las variables más importantes y sus valores de importancia.
                                        Debe tener las columnas "Variable" y "Valor".
        nivel_riesgo (str): Nivel de riesgo global del abonado, que puede ser "Muy Bajo", "Bajo", "Medio", "Alto" o "Muy Alto".

    Retorna:
        str: Frase resumen sobre el riesgo de abandono, con detalles sobre las variables que lo afectan.
    """

    # Separar las variables en dos grupos: las que aumentan el riesgo y las que lo disminuyen
    positivas = df[df["Valor"] > 0].sort_values(by="Valor", ascending=False).head(3)  # Las 3 que aumentan el riesgo
    negativas = df[df["Valor"] <= 0].sort_values(by="Valor", ascending=True).head(3)  # Las 3 que disminuyen el riesgo
    
    # Función interna para generar una lista de variables en formato adecuado
    def listar_variables(variables):
        if len(variables) == 0:
            return "ninguna" # Si no hay variables, devuelve "ninguna"
        elif len(variables) == 1:
            return variables[0] # Si hay una sola variable, la devuelve tal cual
        elif len(variables) == 2:
            return f"{variables[0]} y {variables[1]}"  # Si hay dos variables, las une con "y"
        else:
            return ", ".join(variables[:-1]) + f" y {variables[-1]}" # Si hay más de dos, las une con comas y "y" antes de la última
    
    # Obtener las listas de variables positivas y negativas
    positivas_variables = listar_variables(positivas["Variable"].tolist()) # Variables que aumentan el riesgo
    negativas_variables = listar_variables(negativas["Variable"].tolist()) # Variables que disminuyen el riesgo
    
    # Ajustar el mensaje dependiendo de las variables disponibles
    if positivas_variables == "ninguna" and negativas_variables == "ninguna":
        frase_resumen = f"No se identificaron variables que aumenten ni que disminuyan el riesgo de abandono. El riesgo global es {nivel_riesgo}."
    elif positivas_variables == "ninguna":
        frase_resumen = f"El riesgo de abandono de este abonado es reducido principalmente por {negativas_variables}, resultando en un riesgo global {nivel_riesgo}."
    elif negativas_variables == "ninguna":
        frase_resumen = f"El riesgo de abandono de este abonado aumenta principalmente por {positivas_variables}, resultando en un riesgo global {nivel_riesgo}."
    else:
        frase_resumen = f"El riesgo de abandono de este abonado aumenta principalmente por {positivas_variables}, mientras que {negativas_variables} ayudan a reducirlo, resultando en un riesgo global {nivel_riesgo}."
    
    return frase_resumen


def generar_explicacion_contexto(df: pd.DataFrame) -> None:
    """
    Genera una explicación contextual del riesgo de abandono separando las variables que aumentan y disminuyen el riesgo.
    Muestra la información en dos columnas: una para las variables que aumentan el riesgo y otra para las que lo disminuyen.

    Parámetros:
        df (pd.DataFrame): DataFrame con las variables más importantes y sus valores de importancia.
                        Debe tener las columnas "Variable" y "Valor", donde "Valor" es la importancia de cada variable.

    Retorna:
        None: La función no retorna nada, solo muestra los resultados en la interfaz.
    """

    # Separar las variables en dos grupos: las que aumentan el riesgo y las que lo disminuyen
    positivas = df[df["Valor"] > 0]
    negativas = df[df["Valor"] <= 0]

    # Crear dos columnas
    col1, col2 = st.columns(2)

    with col1:
       
        # Mostrar las variables que aumentan el riesgo
        st.markdown("### 🔺 Aumentan el riesgo")
        for _, row in positivas.iterrows():
            var = row["Variable"]
            imp = round(row["Valor"], 3)
            st.markdown(f"**{var}**: {imp}")

    with col2:
        
        # Mostrar las variables que disminuyen el riesgo
        st.markdown("### 🔻 Disminuyen el riesgo")
        for _, row in negativas.iterrows():
            var = row["Variable"]
            imp = round(row["Valor"], 3)
            st.markdown(f"**{var}**: {imp}")