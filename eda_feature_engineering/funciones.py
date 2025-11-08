"""
Este módulo contiene funciones para procesar datos de usuarios, servicios,
pagos y accesos en un centro deportivo.

Incluye:
- Features de servicios extra (entrenamientos, fisioterapia, nutrición)
- Features económicos basados en pagos y abonos
- Features de comportamiento basados en accesos
- Funciones de integración y limpieza final de dataset
"""


#IMPORTACIÓN LIBRERIAS NECESARIAS
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import re

from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


# FUNCIÓN PARA LA CARGA DE ARCHIVOS
def load_dataset(file_path, file_type=None, separator=None, encoding='utf-8', **kwargs):
    """
    Loads a dataset in different formats, with support for custom separators, encoding, and more options.
    """
    # If the file type is not specified, infer from file extension
    if not file_type:
        file_type = file_path.split('.')[-1].lower()

    # Load according to the file type
    if file_type == 'csv':
        return pd.read_csv(file_path, sep=separator or ',', encoding=encoding, **kwargs)
    elif file_type in ['xls', 'xlsx']:
        return pd.read_excel(file_path, **kwargs)
    elif file_type == 'json':
        return pd.read_json(file_path, encoding=encoding, **kwargs)
    else:
        raise ValueError(f"File format '{file_type}' not supported. Use 'csv', 'excel', or 'json'.")


#GESTIÓN DE COLUMNAS Y FILAS: DATOS INICIALES
def preparar_datos_iniciales(df: pd.DataFrame, columnas_a_eliminar: list,   columnas_a_renombrar: dict,
    columnas_numericas: list, columnas_fechas: list) -> pd.DataFrame:
    """
    Prepara un DataFrame para análisis exploratorio de datos (EDA).
    
    Parámetros:
        df (pd.DataFrame): DataFrame original
        columnas_a_eliminar (list): Columnas que se eliminarán del DataFrame
        columnas_a_renombrar (dict): Diccionario con columnas a renombrar {original: nuevo_nombre}
        columnas_numericas (list): Columnas que deben convertirse a tipo numérico
        columnas_fechas (list): Columnas que deben convertirse a tipo datetime
    
    Retorna:
        pd.DataFrame: DataFrame transformado
    """
    df_eda = df.copy()

    # Eliminar columnas
    df_eda.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')

    # Renombrar columnas
    df_eda.rename(columns=columnas_a_renombrar, inplace=True)

    # Conversión de columnas numéricas
    for col in columnas_numericas:
        if col in df_eda.columns:
            df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')

    # Conversión de columnas de fecha
    for col in columnas_fechas:
        if col in df_eda.columns:
            df_eda[col] = pd.to_datetime(df_eda[col], errors='coerce', dayfirst=True)

    return df_eda

#FUNCION PARA EL EDA BÁSICO
def eda_basica(df: pd.DataFrame, nombre_df: str = "DataFrame") -> None:
    """
    Realiza un análisis exploratorio básico sobre un DataFrame:
    - Identifica variables numéricas y categóricas
    - Detecta valores nulos y muestra una visualización si los hay
    - Revisa duplicados (filas y columnas)

    Parámetros:
        df (pd.DataFrame): El DataFrame a analizar
        nombre_df (str): Nombre para mostrar del DataFrame (opcional)
    """
    print(f"\n📋 Análisis EDA básico de: {nombre_df}")

    # 1. Tipos de variables
    print("\n📌 Tipos de Variables:")
    num_vbles = df.select_dtypes(include='number').columns.tolist()
    cat_vbles = df.select_dtypes(exclude='number').columns.tolist()
    print(f"🔢 Variables Numéricas: {num_vbles}")
    print(f"🔠 Variables Categóricas: {cat_vbles}")

    # 2. Valores nulos
    print("\n🕳️ Variables con valores nulos:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_percentage = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Total Missing': missing,
        'Percentage Missing': missing_percentage
    })

    if not missing.empty:
        display(missing_df)
        plt.figure(figsize=(10, 6))
        missing.plot(kind='barh', color='salmon')
        plt.title("Variables con Valores Nulos")
        plt.xlabel("Cantidad de valores nulos")
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("✅ No hay valores nulos en el dataset.")

    # 3. Filas duplicadas
    print("\n📎 Filas duplicadas:")
    duplicadas = df.duplicated().sum()
    if duplicadas > 0:
        print(f"🔴 Hay {duplicadas} filas duplicadas.")
        display(df[df.duplicated()])
    else:
        print("✅ No hay filas duplicadas.")

    # 4. Columnas duplicadas
    print("\n📎 Columnas duplicadas:")
    columnas_duplicadas = df.T.duplicated().sum()
    if columnas_duplicadas > 0:
        print(f"🔴 Hay {columnas_duplicadas} columnas duplicadas.")
    else:
        print("✅ No hay columnas duplicadas.")


#FILTRAR POR FECHAS DEL ESPACIO TEMPORAL ESTUDIADO
def filtrar_por_fecha(df: pd.DataFrame, columna_fecha: str, fecha_limite: str) -> pd.DataFrame:
    """
    Filtra un DataFrame manteniendo solo las filas donde la fecha en la columna especificada
    es menor o igual a una fecha límite.

    Parámetros:
        df (pd.DataFrame): DataFrame a filtrar
        columna_fecha (str): Nombre de la columna de tipo fecha
        fecha_limite (str): Fecha límite en formato 'YYYY-MM-DD'

    Retorna:
        pd.DataFrame: DataFrame filtrado
    """
    if columna_fecha not in df.columns:
        raise ValueError(f"La columna '{columna_fecha}' no existe en el DataFrame.")

    # Convertir la fecha límite a tipo datetime
    fecha_limite = pd.to_datetime(fecha_limite, errors='coerce')

    if fecha_limite is pd.NaT:
        raise ValueError("La fecha límite no es válida. Usa el formato 'YYYY-MM-DD'.")

    # Asegurar que la columna sea de tipo datetime
    df_filtrado = df.copy()
    df_filtrado[columna_fecha] = pd.to_datetime(df_filtrado[columna_fecha], errors='coerce')

    # Aplicar el filtro
    df_filtrado = df_filtrado[df_filtrado[columna_fecha] <= fecha_limite]

    return df_filtrado


#FUNCIÓN DE EXCLUSIÓN DE TIPOS DE ABONO DE LOS CLIENTES
def excluir_valores(df: pd.DataFrame, columna: str, valores_a_excluir: list) -> pd.DataFrame:
    """
    Filtra un DataFrame excluyendo las filas que contienen ciertos valores en una columna específica.

    Parámetros:
        df (pd.DataFrame): DataFrame original
        columna (str): Nombre de la columna en la que se aplicará el filtro
        valores_a_excluir (list): Lista de valores que se quieren excluir

    Retorna:
        pd.DataFrame: DataFrame filtrado, sin los valores excluidos y con índice reseteado
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    df_filtrado = df[~df[columna].isin(valores_a_excluir)].reset_index(drop=True)

    return df_filtrado

TIPOS_ABONO_EXCLUIR = ["EMP0 - EMPLEADOS CLUB SIN CUOTA",  "EMP1 - EMPLEATS D'ALTRES EMPRESES",  "CL02 - SOCIS NUMERARIS",    "CL01 - SOCIS D'HONOR", "AA0 - PROMO 9'90€ (PRIMER MES)"]

#FUNCIÓN PARA ONE-HOT ENCODING 
def codificar_one_hot(df: pd.DataFrame, columna: str, drop_first: bool = True) -> pd.DataFrame:
    """
    Aplica codificación one-hot a una columna categórica de un DataFrame y elimina la columna original.

    Parámetros:
        df (pd.DataFrame): DataFrame original
        columna (str): Nombre de la columna categórica a codificar
        drop_first (bool): Si se elimina la primera categoría para evitar multicolinealidad (default=True)

    Retorna:
        pd.DataFrame: DataFrame con columnas one-hot y sin la columna original
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    # Aplicar codificación one-hot
    dummies = pd.get_dummies(df[columna], prefix=columna, drop_first=drop_first)

    # Concatenar y eliminar columna original
    df_codificado = pd.concat([df.drop(columns=[columna]), dummies], axis=1)

    return df_codificado

#FUNCIÓN PARA CAMBIAR LOS NOMBRES D ELOS ABONOS
def cambios_nombre_abonos(df: pd.DataFrame, columna: str, mapeo_manual: dict = None) -> pd.DataFrame:
    """
    Extrae un código de una columna de texto usando una expresión regular y un mapeo manual.

    Parámetros:
        df (pd.DataFrame): DataFrame de entrada
        columna (str): Nombre de la columna a procesar
        mapeo_manual (dict): Diccionario de mapeo manual para valores sin patrón

    Retorna:
        pd.DataFrame: Una copia del DataFrame con la columna transformada
    """

    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    # Expresión regular para códigos con formato "XXXX - ..."
    patron_codigo = re.compile(r"^([A-Z0-9]{2,5})\s*-\s*")

    # Usar un dict vacío si no se pasa uno manual
    mapeo_manual = mapeo_manual or {}

    def procesar_valor(valor):
        if pd.isna(valor):
            return None
        match = patron_codigo.match(str(valor).strip())
        if match:
            return match.group(1)
        return mapeo_manual.get(valor, valor)

    df_copia = df.copy()
    df_copia[columna] = df_copia[columna].apply(procesar_valor)
    return df_copia


MAPEO_MANUAL_ABONOS = {
    'FAMILIAR (PARES MES ELS MENORS DE 18 ANYS)': 'FA00',
    'QUOTA MANTENIMENT - MENSUAL': 'QM01',
    'FAMILIAR MONOPARENTAL (AMB CARNET MONOPARENTAL)': 'FM01',
    'TEMP': 'TEMP',
    'VIP': 'VIP',
    'FAMILIAR ANUAL': 'FA12',
    'QUOTA MANTENIMENT - TRIM.': 'QM03',
    'ATURATS TOTAL': 'AT01',
    'ATURATS MATI': 'AT00',
    "AA0 - PROMO 9'90€ (PRIMER MES)": 'AA0',
    "APG03-GROUPON-SEMESTRAL": 'APG03',
    "APG04-GROUPON-ANUAL": 'APG04',
    "T07 - QUOTA I-10": 'T07',
    "T12 - QUOTA TR-25": 'T12',
    "T14 - QUOTA TR-15": 'T14',
    "T16 - QUOTA": 'T16',
    "T15 - QUOTA TR-10": 'T15',
    'T08 - QUOTA TR-30': 'T08',
}

#------------------------------------------FEATURE SOCIOS ACTIVOS--------------------------------------------

def crear_features_activos(df: pd.DataFrame,   fecha_corte_inicio: str,  fecha_corte: str,
                              columnas_fechas: dict = None,   id_col: str = 'IdPersona') -> pd.DataFrame:
    """
    Crea variables temporales de interés para análisis longitudinal de altas y abonos.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos originales.
    fecha_corte_inicio : str
        Fecha inicio del periodo de estudio (formato 'YYYY-MM-DD').
    fecha_corte : str
        Fecha de corte final (formato 'YYYY-MM-DD').
    columnas_fechas : dict, opcional
        Diccionario con nombres de columnas de fechas, con claves:
        - 'antiguedad' (por defecto 'FAntiguedad')
        - 'inscripcion' (por defecto 'FechaInscripcion')
    id_col : str, opcional
        Nombre de la columna que identifica a la persona (por defecto 'IdPersona').

    Retorna:
    --------
    pd.DataFrame
        DataFrame con las nuevas variables creadas y columnas originales importantes.
    """

    # Definir columnas fechas por defecto si no se pasan
    if columnas_fechas is None:
        columnas_fechas = {'antiguedad': 'FAntiguedad', 'inscripcion': 'FechaInscripcion'}

    # Copiar df para no modificar original
    df = df.copy()

    # Convertir columnas de fechas a datetime (por si acaso)
    df[columnas_fechas['antiguedad']] = pd.to_datetime(df[columnas_fechas['antiguedad']], errors='coerce')
    df[columnas_fechas['inscripcion']] = pd.to_datetime(df[columnas_fechas['inscripcion']], errors='coerce')

    # Convertir fechas de corte a Timestamp
    fecha_corte_inicio = pd.Timestamp(fecha_corte_inicio)
    fecha_corte = pd.Timestamp(fecha_corte)

    # 1. Antigüedad mínima por persona
    antiguedad_min = df.groupby(id_col)[columnas_fechas['antiguedad']].min().reset_index()

    # 2. Última inscripción por persona (más reciente)
    df_sorted = df.sort_values(by=columnas_fechas['inscripcion'], ascending=False)
    ultima_inscripcion = df_sorted.drop_duplicates(subset=id_col, keep='first')

    # 3. Número de altas previas al inicio del periodo
    altas_anteriores = df[df[columnas_fechas['inscripcion']] < fecha_corte_inicio] \
        .groupby(id_col).size().reset_index(name='NumAltasAntesDelPeriodo')

    # 4. Flag si tuvo alguna alta previa
    altas_flag = altas_anteriores.copy()
    altas_flag['TuvoAltasPrevias'] = True
    altas_flag = altas_flag[[id_col, 'TuvoAltasPrevias']]

    # 5. Última alta previa al inicio del periodo (fecha)
    ultima_alta_previa = df[df[columnas_fechas['inscripcion']] < fecha_corte_inicio] \
        .groupby(id_col)[columnas_fechas['inscripcion']].max().reset_index(name='UltimaAltaPrevia')

    # 6. Tiempo (meses) desde la última alta previa al inicio del periodo
    ultima_alta_previa['MesesDesdeUltimaAltaPrevia'] = (
        (fecha_corte_inicio - ultima_alta_previa['UltimaAltaPrevia']) / pd.Timedelta(days=30)
    ).astype(int)

    # --- Merge de todas las features ---
    df_final = ultima_inscripcion.copy()

    df_final = df_final.drop(columns=[columnas_fechas['antiguedad']]).merge(antiguedad_min, on=id_col, how='left')

    df_final = df_final.merge(altas_anteriores, on=id_col, how='left')
    df_final['NumAltasAntesDelPeriodo'] = df_final['NumAltasAntesDelPeriodo'].fillna(0).astype(int)

    df_final = df_final.merge(altas_flag, on=id_col, how='left')
    df_final['TuvoAltasPrevias'] = df_final['TuvoAltasPrevias'].fillna(False)

    df_final = df_final.merge(ultima_alta_previa[[id_col, 'MesesDesdeUltimaAltaPrevia']], on=id_col, how='left')
    df_final['MesesDesdeUltimaAltaPrevia'] = df_final['MesesDesdeUltimaAltaPrevia'].fillna(-1).astype(int)

    # 7. Duración del abono actual en meses hasta fecha de corte
    df_final['MesesDuracionAbonoActual'] = (
        (fecha_corte - df_final[columnas_fechas['inscripcion']]) / pd.Timedelta(days=30)
    ).astype(int)

    # Opcional: eliminar columnas que no quieras (por ejemplo 'FNacimiento')
    if 'FNacimiento' in df_final.columns:
        df_final = df_final.drop(columns='FNacimiento')

    return df_final

#-------------------------------------------FEATURES ALTAS-------------------------------------------------

def crear_features_altas_periodo(df: pd.DataFrame, fecha_inicio: str,  fecha_corte: str,
                                     fecha_col: str = 'FechaAlta',  id_col: str = 'IdPersona') -> pd.DataFrame:
    """
    Genera features a partir de altas filtradas por periodo temporal.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene las altas con columna de fecha.
    fecha_inicio : str
        Fecha inicial del periodo (formato 'YYYY-MM-DD').
    fecha_corte : str
        Fecha final del periodo (formato 'YYYY-MM-DD').
    fecha_col : str, opcional
        Nombre de la columna de fechas en df (por defecto 'FechaAlta').
    id_col : str, opcional
        Nombre columna identificadora de persona (por defecto 'IdPersona').

    Retorna:
    --------
    pd.DataFrame
        DataFrame agrupado por id_col con features:
        - NumAltasEnPeriodo: cantidad de altas en el periodo
        - FechaPrimeraAltaEnPeriodo: fecha de la primera alta en periodo
        - MesesDesdePrimeraAltaEnPeriodo: meses desde la primera alta al corte
    """
    df = df.copy()

    # Convertir columna de fecha a datetime, asumiendo día primero
    df[fecha_col] = pd.to_datetime(df[fecha_col], dayfirst=True, errors='coerce')

    # Convertir fechas de corte
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_corte = pd.to_datetime(fecha_corte)

    # Filtrar por periodo
    df_periodo = df[(df[fecha_col] >= fecha_inicio) & (df[fecha_col] <= fecha_corte)]

    # Agrupar para crear features
    altas_agg = df_periodo.groupby(id_col).agg(
        NumAltasEnPeriodo=(fecha_col, 'count'),
        FechaPrimeraAltaEnPeriodo=(fecha_col, 'min')
    ).reset_index()

    # Calcular meses desde la primera alta hasta fecha de corte
    altas_agg['MesesDesdePrimeraAltaEnPeriodo'] = (
        (fecha_corte - altas_agg['FechaPrimeraAltaEnPeriodo']) / pd.Timedelta(days=30)
    ).astype(int)

    return altas_agg

#------------------------------------UNIÓN DATAFRAME DE ACTIVOS CON ALTAS-----------------------

def preparar_union_activos_altas(df_activos: pd.DataFrame,  altas_agg: pd.DataFrame,  id_col: str = 'IdPersona',
                                     fecha_inscripcion_col: str = 'FechaInscripcion',    fecha_corte: str = '2025-09-01') -> pd.DataFrame:
    """
    Une el DataFrame de activos con las features de altas y prepara variables adicionales.

    Parámetros:
    -----------
    df_activos : pd.DataFrame
        DataFrame con los datos de los activos.
    altas_agg : pd.DataFrame
        DataFrame con las features agregadas de altas (por persona).
    id_col : str, opcional
        Nombre de la columna identificadora (por defecto 'IdPersona').
    fecha_inscripcion_col : str, opcional
        Nombre de la columna con la fecha de inscripción (por defecto 'FechaInscripcion').
    fecha_corte : str, opcional
        Fecha de corte para cálculos de duración (por defecto '2025-09-01').

    Retorna:
    --------
    pd.DataFrame
        DataFrame combinado y con nuevas columnas preparadas:
        - EsChurn: False por defecto (puedes usar para marcar bajas después)
        - NumAltasEnPeriodo: relleno 0 si no hay datos
        - MesesDesdePrimeraAltaEnPeriodo: relleno -1 si no hay datos
        - FechaFin: igual a fecha_corte
        - VidaGymMeses: meses entre fecha de inscripción y fecha fin
    """
    df = df_activos.copy()
    df_altas = altas_agg.copy()

    # Convertir fecha_corte a timestamp
    fecha_corte_ts = pd.to_datetime(fecha_corte)

    # Merge left para conservar todos los activos
    df_merged = df.merge(df_altas, on=id_col, how='left')

    # Crear columna 'EsChurn' a False (por defecto)
    df_merged['EsChurn'] = False

    # Rellenar NaNs en columnas de altas con valores por defecto
    if 'NumAltasEnPeriodo' in df_merged.columns:
        df_merged['NumAltasEnPeriodo'] = df_merged['NumAltasEnPeriodo'].fillna(0).astype(int)
    if 'MesesDesdePrimeraAltaEnPeriodo' in df_merged.columns:
        df_merged['MesesDesdePrimeraAltaEnPeriodo'] = df_merged['MesesDesdePrimeraAltaEnPeriodo'].fillna(-1).astype(int)
    if 'FechaPrimeraAltaEnPeriodo' in df_merged.columns:
        df_merged = df_merged.drop(columns=['FechaPrimeraAltaEnPeriodo'])

    # Asegurar que la fecha de inscripción es datetime
    df_merged[fecha_inscripcion_col] = pd.to_datetime(df_merged[fecha_inscripcion_col], errors='coerce')

    # Definir fecha fin = fecha_corte
    df_merged['FechaFin'] = fecha_corte_ts

    # Calcular VidaGym en días y luego en meses
    df_merged['VidaGymDias'] = (df_merged['FechaFin'] - df_merged[fecha_inscripcion_col]).dt.days
    df_merged['VidaGymMeses'] = df_merged['VidaGymDias'] / 30

    # Eliminar columna auxiliar días
    df_merged = df_merged.drop(columns=['VidaGymDias'])

    return df_merged


#-----------------------------------------FEATURE BAJAS----------------------------------------------------

def crear_features_bajas_periodo(df: pd.DataFrame, fecha_inicio: str,  fecha_corte: str,
                                    fecha_col: str = 'FechaBaja',   id_col: str = 'IdPersona') -> pd.DataFrame:
    """
    Genera features a partir de bajas filtradas por periodo temporal.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene las bajas con columna de fecha.
    fecha_inicio : str
        Fecha inicial del periodo (formato 'YYYY-MM-DD').
    fecha_corte : str
        Fecha final del periodo (formato 'YYYY-MM-DD').
    fecha_col : str, opcional
        Nombre de la columna de fechas en df (por defecto 'FechaBaja').
    id_col : str, opcional
        Nombre columna identificadora de persona (por defecto 'IdPersona').

    Retorna:
    --------
    pd.DataFrame
        DataFrame agrupado por id_col con features:
        - NumBajasEnPeriodo: cantidad de bajas en el periodo
        - FechaUltimaBajaEnPeriodo: fecha de la última baja en periodo
        - MesesDesdeUltimaBaja: meses desde la última baja al corte
    """
    df = df.copy()

    # Convertir columna de fecha a datetime, asumiendo día primero
    df[fecha_col] = pd.to_datetime(df[fecha_col], dayfirst=True, errors='coerce')

    # Convertir fechas de corte
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_corte = pd.to_datetime(fecha_corte)

    # Filtrar por periodo
    df_periodo = df[(df[fecha_col] >= fecha_inicio) & (df[fecha_col] <= fecha_corte)]

    # Agrupar para crear features
    bajas_agg = df_periodo.groupby(id_col).agg(
        NumBajasEnPeriodo=(fecha_col, 'count'),
        FechaUltimaBajaEnPeriodo=(fecha_col, 'max')
    ).reset_index()

    # Calcular meses desde la última baja hasta fecha de corte
    bajas_agg['MesesDesdeUltimaBaja'] = (
        (fecha_corte - bajas_agg['FechaUltimaBajaEnPeriodo']) / pd.Timedelta(days=30)
    ).astype(int)

    return bajas_agg



#-----------------------------------UNIÓN Y PREPARACIÓN BAJAS INFO-------------------------------------------

def merge_bajas_info(df_bajas_features, bajas_filtrado_limpios, id_col='IdPersona',
                     cols_a_unir=['IdPersona', 'Edad', 'Sexo_Mujer', 'FAntiguedadBaja', 'FechaAltaBaja', 'TipoAbonoBaja', 'FechaBaja'],
                     fecha_orden_col='FechaAltaBaja'):
    """
    Une la información histórica de bajas con las features de bajas.

    Parámetros:
    - df_bajas_features: DataFrame con las features de bajas.
    - bajas_filtrado_limpios: DataFrame histórico filtrado y limpio de bajas.
    - id_col: columna identificadora para merge (por defecto 'IdPersona').
    - cols_a_unir: columnas que quieres unir desde bajas_filtrado_limpios.
    - fecha_orden_col: columna para ordenar para obtener la última baja (por defecto 'FechaAltaBaja').

    Retorna:
    - DataFrame resultante del merge.
    """
    # Ordenamos y obtenemos la última baja por IdPersona
    df_hist_abonados = bajas_filtrado_limpios.sort_values(fecha_orden_col).drop_duplicates(id_col, keep='last')

    # Merge con df_bajas_features
    bajas_completas = df_bajas_features.merge(
        df_hist_abonados[cols_a_unir],
        on=id_col,
        how='left'
    )

    return bajas_completas


def preparar_bajas(df_bajas,  col_antiguedad='FAntiguedadBaja',  col_fecha_baja='FechaBaja',   col_fecha_alta='FechaAltaBaja',  col_tipo_abono='TipoAbonoBaja'):
    """
    Limpia y prepara el DataFrame de bajas para análisis.

    Parámetros:
    - df_bajas: DataFrame original de bajas.
    - col_antiguedad: columna de antigüedad en bajas (fecha).
    - col_fecha_baja: columna de fecha de baja.
    - col_fecha_alta: columna de fecha de alta (para renombrar).
    - col_tipo_abono: columna de tipo abono (para renombrar).

    Retorna:
    - DataFrame modificado con fechas convertidas, cálculo de vida en meses, flag EsChurn, y columnas renombradas.
    """
    df = df_bajas.copy()

    # Convertir a datetime
    df[col_antiguedad] = pd.to_datetime(df[col_antiguedad], dayfirst=True)
    df[col_fecha_baja] = pd.to_datetime(df[col_fecha_baja], dayfirst=True)

    # Calcular vida en días y meses
    df['VidaGymDias'] = (df[col_fecha_baja] - df[col_antiguedad]).dt.days
    df['VidaGymMeses'] = df['VidaGymDias'] / 30
    df = df.drop(columns='VidaGymDias')

    # Flag de churn
    df['EsChurn'] = True

    # Renombrar columnas para homogeneizar
    df = df.rename(columns={
        col_antiguedad: 'FAntiguedad',
        col_fecha_alta: 'FechaInscripcion',
        col_tipo_abono: 'TipoAbonoActual',
        col_fecha_baja: 'FechaFin'
    })

    return df


#----------------------------------GESTIÓN DE CONFLICTOS ENTRE ACTIVOS Y BAJAS-----------------------------

def analizar_inconsistencias(df_bajas_preparadas, df_completo_activos_altas):
    """
    Analiza IDs que aparecen tanto en bajas como en activos y calcula porcentaje de casos erróneos.

    Parámetros:
    - df_bajas_preparadas: DataFrame con bajas preparadas.
    - df_completo_activos_altas: DataFrame con activos y altas combinadas.

    Retorna:
    - dict con IDs en ambos DataFrames y porcentajes respecto a ambos totales.
    """
    # IDs de bajas que también están en activos
    ids_en_ambos = df_bajas_preparadas.loc[
        df_bajas_preparadas['IdPersona'].isin(df_completo_activos_altas['IdPersona']),
        'IdPersona'
    ].unique()

    bajas_en_activos = df_bajas_preparadas[
        df_bajas_preparadas['IdPersona'].isin(df_completo_activos_altas['IdPersona'])
    ]

    # Casos erróneos = cantidad de IDs en ambos
    N_erroneos = len(ids_en_ambos)

    # Cálculo de porcentajes
    total_bajas = len(df_bajas_preparadas)
    porcentaje_erroneos_bajas = (N_erroneos / total_bajas) * 100 if total_bajas > 0 else 0

    total_activos = len(df_completo_activos_altas)
    porcentaje_erroneos_activos = (N_erroneos / total_activos) * 100 if total_activos > 0 else 0

    print(f"Los casos inconsistentes representan el {porcentaje_erroneos_bajas:.2f}% del total de bajas.")
    print(f"Los casos inconsistentes representan el {porcentaje_erroneos_activos:.2f}% del total de activos.")

    return {
        'ids_en_ambos': ids_en_ambos,
        'bajas_en_activos': bajas_en_activos,
        'porcentaje_erroneos_bajas': porcentaje_erroneos_bajas,
        'porcentaje_erroneos_activos': porcentaje_erroneos_activos
    }


def eliminar_inconsistencias(df_activos, df_bajas, ids_inconsistentes):
    """
    Elimina los registros de ambos DataFrames cuyos IdPersona estén en la lista de inconsistencias.

    Parámetros:
    - df_activos: DataFrame de activos (df_completo_activos_altas)
    - df_bajas: DataFrame de bajas (df_bajas_preparadas)
    - ids_inconsistentes: lista o array de IdPersona inconsistentes (ids_en_ambos)

    Retorna:
    - Tuple con los DataFrames limpios (activos_sin_incons, bajas_sin_incons)
    """
    activos_limpio = df_activos[~df_activos['IdPersona'].isin(ids_inconsistentes)].copy()
    activos_limpio= activos_limpio.reset_index(drop=True)
    bajas_limpio = df_bajas[~df_bajas['IdPersona'].isin(ids_inconsistentes)].copy()
    bajas_limpio= bajas_limpio.reset_index(drop=True)
    
    return activos_limpio, bajas_limpio


#----------------------------UNIÓN FINAL DE ABONADOS (ACTIVOS Y BAJAS)-------------------------------------------

def preparar_df_final(df_activos, df_bajas):
    """
    Une los DataFrames de activos y bajas, y prepara el DataFrame final con las conversiones
    de fechas, tratamiento de valores faltantes y ajustes en columnas booleanas.

    Parámetros:
    - df_activos: DataFrame con usuarios activos (preparados)
    - df_bajas: DataFrame con usuarios dados de baja (preparados)

    Retorna:
    - df_usuarios_final: DataFrame final listo para análisis
    """
    import pandas as pd

    # Concatenar activos y bajas
    df_usuarios_final = pd.concat([df_activos, df_bajas], ignore_index=True)

    # Convertir columnas de fechas a datetime (manejar errores y formato día primero)
    for col in ['FechaInscripcion', 'FAntiguedad', 'FechaFin', 'FechaUltimaBajaEnPeriodo']:
        if col in df_usuarios_final.columns:
            df_usuarios_final[col] = pd.to_datetime(df_usuarios_final[col], errors='coerce', dayfirst=True)

    # Para usuarios churn (EsChurn=True), ajustar 'TuvoAltasPrevias'
    df_usuarios_final.loc[df_usuarios_final['EsChurn'] == True, 'TuvoAltasPrevias'] = False
    df_usuarios_final['TuvoAltasPrevias'] = df_usuarios_final['TuvoAltasPrevias'].astype(bool)

    # Para usuarios activos (EsChurn=False), rellenar NaNs en columnas de bajas con 0
    cols_bajas = ['NumBajasEnPeriodo', 'MesesDesdeUltimaBaja']
    for col in cols_bajas:
        if col in df_usuarios_final.columns:
            df_usuarios_final.loc[df_usuarios_final['EsChurn'] == False, col] = \
                df_usuarios_final.loc[df_usuarios_final['EsChurn'] == False, col].fillna(0)

    # Para usuarios churn (EsChurn=True), rellenar NaNs en columnas de altas con -1
    cols_altas = ['NumAltasAntesDelPeriodo', 'TuvoAltasPrevias', 'MesesDesdeUltimaAltaPrevia',
                  'MesesDuracionAbonoActual', 'NumAltasEnPeriodo', 'MesesDesdePrimeraAltaEnPeriodo']
    for col in cols_altas:
        if col in df_usuarios_final.columns:
            df_usuarios_final.loc[df_usuarios_final['EsChurn'] == True, col] = \
                df_usuarios_final.loc[df_usuarios_final['EsChurn'] == True, col].fillna(-1)
    
    return df_usuarios_final

#------------------------------------------FEATURE SERVICIOS EXTRA--------------------------------------------
def agregar_servicios(df_servicios):
    """
    Agrupa el DataFrame de servicios por IdPersona y calcula:
    - Número de conceptos únicos
    - Número de tipos de servicio únicos

    Parámetros:
    - df_servicios: DataFrame con columnas 'IdPersona', 'Concepto', 'TipoServicio'

    Retorna:
    - df_agregado: DataFrame con columnas 'IdPersona', 'Total_conceptos_unicos', 'Total_tipos_servicios_unicos'
    """
    df_agregado = df_servicios.groupby('IdPersona').agg({
        'Concepto': 'nunique',
        'TipoServicio': 'nunique'
    }).reset_index()

    df_agregado = df_agregado.rename(columns={
        'Concepto': 'Total_conceptos_unicos',
        'TipoServicio': 'Total_tipos_servicios_unicos'
    })

    return df_agregado

def aplicar_one_hot_encoding_servicios(df_servicios):
    """
    Aplica One-Hot Encoding a las columnas 'Concepto' y 'TipoServicio' y concatena las columnas resultantes
    al DataFrame original.

    Parámetros:
    - df_servicios: DataFrame con las columnas 'Concepto' y 'TipoServicio'

    Retorna:
    - df_servicios_encoded: DataFrame con las columnas originales más las columnas one-hot encoded
    """
    df_one_hot_concepto = pd.get_dummies(df_servicios['Concepto'], prefix='Concepto', drop_first=False)
    df_one_hot_tipo_servicio = pd.get_dummies(df_servicios['TipoServicio'], prefix='TipoServicio', drop_first=False)

    df_servicios_encoded = pd.concat([df_servicios, df_one_hot_concepto, df_one_hot_tipo_servicio], axis=1)

    return df_servicios_encoded

def agregar_y_unir_servicios(df_servicios_encoded, df_agregado):
    """
    Agrupa por 'IdPersona' sumando las columnas one-hot encoded,
    luego une con df_agregado y limpia columnas innecesarias,
    finalmente añade la columna 'UsoServiciosExtra' con valor True.

    Parámetros:
    - df_servicios_encoded: DataFrame con columnas one-hot encoded y 'IdPersona'
    - df_agregado: DataFrame con agregados por 'IdPersona' (total conceptos y tipos)

    Retorna:
    - df_servicios_final: DataFrame con features agregados y limpieza hecha
    """
    # Agrupar sumando one-hot encoded
    df_encoded_grouped = df_servicios_encoded.groupby('IdPersona').agg('sum').reset_index()

    # Merge con df_agregado
    df_servicios_final = pd.merge(df_encoded_grouped, df_agregado, on='IdPersona', how='left')

    # Eliminar columnas originales no necesarias (si existen)
    for col in ['Concepto', 'TipoServicio']:
        if col in df_servicios_final.columns:
            df_servicios_final = df_servicios_final.drop(columns=[col])

    # Añadir columna de uso de servicios extra
    df_servicios_final["UsoServiciosExtra"] = True

    return df_servicios_final

def preparar_servicios_final(df_servicios_final):
    """
    Ajusta tipos de datos en el dataframe final de servicios:
    - Convierte ciertas columnas numéricas a int
    - Convierte columnas de conceptos (one-hot) a booleano

    Parámetros:
    - df_servicios_final: DataFrame con columnas a convertir

    Retorna:
    - df_servicios_final modificado con tipos correctos
    """
    # Columnas que convertir a int (asegúrate que existan)
    int_cols = [
        'Cantidad_2024_servicios',
        'Cantidad_2025_servicios',
        'Cantidad_total_pagado_servicios',
        'TipoServicio_ENTRENAMENTS PERSONALS',
        'TipoServicio_FISIOTERÀPIA',
        'TipoServicio_NUTRICIÓ'
    ]
    
    for col in int_cols:
        if col in df_servicios_final.columns:
            df_servicios_final[col] = df_servicios_final[col].astype(int)
    
    # Convertir columnas de conceptos a booleano
    concepto_cols = [col for col in df_servicios_final.columns if col.startswith('Concepto_')]
    if concepto_cols:
        df_servicios_final[concepto_cols] = df_servicios_final[concepto_cols].astype(bool)

    return df_servicios_final


#---------------------------------------------------------------FEATURES ECONOMIA--------------------------------------------------------------------

def agregar_features_economia(usuario_df: pd.DataFrame) -> pd.Series:
    """
    Calcula las características económicas por usuario (IdPersona) a partir de su historial de pagos.
    """

    fechas = usuario_df['FechaRenovacion'].sort_values()
    intervalos = fechas.diff().dt.days / 30  # Diferencia en meses aproximados

    # Calcular media y desviación estándar del intervalo entre pagos
    media_intervalo = intervalos.mean() if not intervalos.empty else 0.0
    std_intervalo = intervalos.std() if not intervalos.empty else 0.0

    total_pagado = usuario_df['TotalCobrado'].sum()

    # Sumar los distintos métodos de pago
    pagos = usuario_df[['PagoMetálico', 'PagoRecibo', 'PagoTarjeta', 'PagoTransferencia']].sum()

    # Estadísticas sobre el pago total
    media_pago = usuario_df['TotalCobrado'].mean()
    std_pago = usuario_df['TotalCobrado'].std()
    
    # Coeficiente de variación del pago
    coef_var_pago = std_pago / media_pago if (media_pago and not pd.isna(media_pago)) else 0.0

    # Abono más frecuente
    modo_abono = usuario_df['TipoAbono'].mode()
    abono_mas_frecuente = modo_abono.iloc[0] if not modo_abono.empty else None

    # Cálculo de meses únicos con pago
    meses_pagados = usuario_df['FechaRenovacion'].dt.to_period("M").nunique()
    porcentaje_meses_pagados = meses_pagados / 13  # Asumiendo 13 meses de historial

    return pd.Series({
        'NumPagosEconomia': len(usuario_df),
        'MediaIntervaloMeses': media_intervalo if not pd.isna(media_intervalo) else 0.0,
        'StdIntervaloMeses': std_intervalo if not pd.isna(std_intervalo) else 0.0,
        'UltimoPagoMesEconomia': fechas.max().month if not fechas.empty else 0,
        'PrimerPagoMesEconomia': fechas.min().month if not fechas.empty else 0,
        'MesesConPagoEconomia': meses_pagados,
        'PorcentajeMesesPagados': porcentaje_meses_pagados,
        'TotalPagadoEconomia': total_pagado,
        'media_pagoEconomia': media_pago if not pd.isna(media_pago) else 0.0,
        'StdPagoEconomia': std_pago if not pd.isna(std_pago) else 0.0,
        'CoefVarPagoEconomia': coef_var_pago,
        'PctPagoMetalico': pagos['PagoMetálico'] / total_pagado if total_pagado else 0.0,
        'PctPagoRecibo': pagos['PagoRecibo'] / total_pagado if total_pagado else 0.0,
        'PctPagoTarjeta': pagos['PagoTarjeta'] / total_pagado if total_pagado else 0.0,
        'PctPagoTransferencia': pagos['PagoTransferencia'] / total_pagado if total_pagado else 0.0,
        'NumAbonosDistintos': usuario_df['TipoAbono'].nunique(),
        'AbonoMasFrecuente': abono_mas_frecuente,
        
    })


#-----------------------------------------------------PARTE ACCESOS--------------------------------------
def concatenar_dataframes(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenar una lista arbitraria de DataFrames en uno solo.
    
    Parámetros:
        *dfs: DataFrames a concatenar.
        
    Retorna:
        pd.DataFrame: DataFrame concatenado con índice reiniciado y copia.
    """
    concatenado = pd.concat(dfs, ignore_index=True)
    return concatenado.copy()


def preparar_fechas_y_horas(df_accesos: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara y limpia las columnas de fecha y hora en el DataFrame df_accesos.
    - Crea la columna 'Fecha' a partir de 'Año', 'Mes', 'Dia'.
    - Convierte y valida fechas, eliminando filas con fechas inválidas.
    - Convierte 'HoraEntrada' y 'HoraSalida' a objetos time.
    - Crea columnas 'FechaHoraEntrada' y 'FechaHoraSalida' combinando fecha y hora.
    
    Parámetros:
        df_accesos (pd.DataFrame): DataFrame con columnas Año, Mes, Dia, HoraEntrada, HoraSalida.
    
    Retorna:
        pd.DataFrame: DataFrame con las columnas preparadas y filas inválidas eliminadas.
    """
    df = df_accesos.copy()

    # Construir columna 'Fecha' en formato yyyy-mm-dd
    df['Fecha'] = (
        df['Año'].astype(int).astype(str) + '-' +
        df['Mes'].astype(int).astype(str).str.zfill(2) + '-' +
        df['Dia'].astype(int).astype(str).str.zfill(2)
    )

    # Convertir a datetime, ignorando errores
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    # Eliminar filas con fechas inválidas
    df = df.dropna(subset=['Fecha']).reset_index(drop=True)

    # Convertir HoraEntrada y HoraSalida a datetime.time
    df['HoraEntrada'] = pd.to_datetime(
        df['HoraEntrada'].astype(str), format='%H:%M:%S', errors='coerce'
    ).dt.time

    df['HoraSalida'] = pd.to_datetime(
        df['HoraSalida'].astype(str), format='%H:%M:%S', errors='coerce'
    ).dt.time

    # Combinar fecha y hora en datetime
    df['FechaHoraEntrada'] = df.apply(
        lambda row: pd.Timestamp.combine(row['Fecha'], row['HoraEntrada']) if pd.notnull(row['HoraEntrada']) else pd.NaT, axis=1
    )

    df['FechaHoraSalida'] = df.apply(
        lambda row: pd.Timestamp.combine(row['Fecha'], row['HoraSalida']) if pd.notnull(row['HoraSalida']) else pd.NaT, axis=1
    )

    return df

import holidays

def calcular_features_accesos(df_accesos: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de accesos agrupados por IdPersona.

    Parámetros:
        df_accesos (pd.DataFrame): DataFrame con las columnas necesarias:
            - 'Fecha', 'FechaHoraEntrada', 'FechaHoraSalida', 'IdPersona',
            - 'duracion_min', 'hora_decimal', 'dia_semana', 'mes', 'estacion',
            - 'EsFestivo', 'EsFinDeSemana'.

    Retorna:
        pd.DataFrame: DataFrame con features agregadas por 'IdPersona'.
    """

    # Fechas importantes y preprocesamiento
    df_accesos = df_accesos.copy()
    df_accesos['Fecha'] = pd.to_datetime(df_accesos['Fecha'])
    df_accesos['FechaHoraEntrada'] = pd.to_datetime(df_accesos['FechaHoraEntrada'])
    df_accesos['FechaHoraSalida'] = pd.to_datetime(df_accesos['FechaHoraSalida'])

    # Calcular duracion_min si no existe
    if 'duracion_min' not in df_accesos.columns:
        df_accesos['duracion_min'] = (df_accesos['FechaHoraSalida'] - df_accesos['FechaHoraEntrada']).dt.total_seconds() / 60
    
    # Calcular hora_decimal, dia_semana y mes si no existen
    if 'hora_decimal' not in df_accesos.columns:
        df_accesos['hora_decimal'] = df_accesos['FechaHoraEntrada'].dt.hour + df_accesos['FechaHoraEntrada'].dt.minute / 60
    if 'dia_semana' not in df_accesos.columns:
        df_accesos['dia_semana'] = df_accesos['Fecha'].dt.dayofweek
    if 'mes' not in df_accesos.columns:
        df_accesos['mes'] = df_accesos['Fecha'].dt.month

    # Calcular estacion si no existe
    if 'estacion' not in df_accesos.columns:
        def estacion(mes):
            if mes in [12, 1, 2]: return 'invierno'
            elif mes in [3, 4, 5]: return 'primavera'
            elif mes in [6, 7, 8]: return 'verano'
            return 'otono'
        df_accesos['estacion'] = df_accesos['mes'].apply(estacion)

    # Crear indicadores de festivos y fines de semana si no existen
    if 'EsFestivo' not in df_accesos.columns or 'EsFinDeSemana' not in df_accesos.columns:
        festivos_cat_2024 = {
            '2024-01-01', '2024-01-06', '2024-04-18', '2024-04-21', '2024-05-01',
            '2024-06-17', '2024-06-24', '2024-08-15', '2024-09-11', '2024-11-01',
            '2024-12-06', '2024-12-08', '2024-12-25', '2024-12-26'
        }
        festivos_cat_2025 = {
            '2025-01-01', '2025-01-06', '2025-04-10', '2025-04-13', '2025-05-01',
            '2025-06-16', '2025-06-24', '2025-08-15', '2025-09-11', '2025-11-01',
            '2025-12-06', '2025-12-08', '2025-12-25', '2025-12-26'
        }
        festivos_cat = pd.to_datetime(list(festivos_cat_2024.union(festivos_cat_2025)))
        festivos_es = holidays.Spain(years=[2024, 2025])
        festivos_es_dates = pd.to_datetime(list(festivos_es.keys()))
        todos_festivos = festivos_es_dates.union(festivos_cat)

        df_accesos['EsFestivo'] = df_accesos['Fecha'].isin(todos_festivos)
        df_accesos['EsFinDeSemana'] = df_accesos['Fecha'].dt.dayofweek >= 5

    hoy = pd.Timestamp.today().normalize()

    def features_usuario(grp):
        fechas = grp['Fecha'].sort_values()
        if fechas.empty:
            return pd.Series()

        primera = fechas.min()
        ultima = fechas.max()
        total_dias = (ultima - primera).days or 1
        semanas_totales = total_dias // 7 + 1

        ultimos_30 = fechas[fechas >= (ultima - pd.Timedelta(days=30))]
        ultimos_90 = fechas[fechas >= (ultima - pd.Timedelta(days=90))]
        ultimos_180 = fechas[fechas >= (ultima - pd.Timedelta(days=180))]

        primer_tri = fechas[fechas < (primera + pd.Timedelta(days=90))]
        ultimo_tri = fechas[fechas > (ultima - pd.Timedelta(days=90))]

        diffs = fechas.diff().dt.days.dropna()
        racha_sin_visita = diffs.max() if not diffs.empty else 0

        fechas_range = pd.date_range(start=primera, end=ultima)
        presencia = fechas_range.isin(fechas)
        rachas, count = [], 0
        for v in presencia:
            if v:
                count += 1
            else:
                if count:
                    rachas.append(count)
                    count = 0
        if count:
            rachas.append(count)
        max_racha_visita = max(rachas) if rachas else 0

        semanas_con_visita = pd.DatetimeIndex(fechas).isocalendar().week.nunique()

        return pd.Series({
            'TotalVisitas': len(grp),
            'DiasActivo': fechas.nunique(),
            'TiempoActivoDias': total_dias,
            'VisitasUlt30': len(ultimos_30),
            'VisitasUlt90': len(ultimos_90),
            'VisitasUlt180': len(ultimos_180),
            'PropUlt90': len(ultimos_90) / len(grp) if len(grp) else np.nan,
            'DiasDesdeUltima': (hoy - ultima).days,
            'tasa_caida_visitas': len(ultimo_tri) / len(primer_tri) if len(primer_tri) else np.nan,

            'DuracionMediaTotal': grp['duracion_min'].mean(),
            'DuracionMediaUlt90': grp.loc[grp['Fecha'] >= (ultima - pd.Timedelta(days=90)), 'duracion_min'].mean(),
            'DeltaDuracionUlt90VsTotal': (
                (grp.loc[grp['Fecha'] >= (ultima - pd.Timedelta(days=90)), 'duracion_min'].mean() - grp['duracion_min'].mean())
                / grp['duracion_min'].mean() if grp['duracion_min'].mean() else np.nan
            ),

            'StdDiasEntreVisitas': diffs.std(ddof=0) if len(diffs) > 1 else 0,
            'FrecuenciaModal': diffs.mode()[0] if not diffs.empty else np.nan,
            'MaxRachaSinVisita': racha_sin_visita,
            'MaxRachaConVisita': max_racha_visita,
            'SemanasConVisitaRatio': semanas_con_visita / semanas_totales,

            'HoraMediaAcceso': grp['hora_decimal'].mean(),
            'DiaFavorito': grp['dia_semana'].mode()[0] if not grp['dia_semana'].empty else np.nan,
            'EstacionFavorita': grp['estacion'].mode()[0] if not grp['estacion'].empty else np.nan,

            'PropVisitasFinDeSemana': grp['EsFinDeSemana'].mean(),
            'PropVisitasFestivo': grp['EsFestivo'].mean(),

            'VisitasPrimerTrimestre': len(primer_tri),
            'VisitasUltimoTrimestre': len(ultimo_tri),
        })

    df_features = df_accesos.groupby('IdPersona').apply(features_usuario).reset_index()

    # Mapeo para dias de semana y estaciones
    dias_semana_map = {0: 'lunes', 1: 'martes', 2: 'miércoles', 3: 'jueves', 4: 'viernes', 5: 'sábado', 6: 'domingo'}
    df_features['DiaFavorito'] = df_features['DiaFavorito'].map(dias_semana_map).astype('category')
    df_features['EstacionFavorita'] = df_features['EstacionFavorita'].astype('category')

    # One-hot encoding para variables categóricas
    df_features = pd.get_dummies(df_features, columns=['DiaFavorito', 'EstacionFavorita'], prefix=['DiaFav', 'EstFav'])
    mediana_frec_modal = df_features['FrecuenciaModal'].median()
    df_features['FrecuenciaModal'].fillna(mediana_frec_modal, inplace=True)
    df_features['TieneAccesos'] = True
    return df_features



#---------------------------------FUNCIONES PARA DOCUMENTO FINAL--------------------------------------
def merge_y_limpiar_usuarios_servicios(df_usuarios_final, df_servicios_final):
    """
    Realiza el merge entre usuarios y servicios, y prepara la limpieza de NaNs y tipos.
    
    Parámetros:
    - df_usuarios_final: DataFrame con info de usuarios (altas, bajas, etc)
    - df_servicios_final: DataFrame con info de servicios por usuario
    
    Retorna:
    - df_merge: DataFrame combinado y limpio listo para análisis/modelado
    """
    # Merge con left join para mantener todos usuarios
    df_merge = pd.merge(df_usuarios_final, df_servicios_final, on='IdPersona', how='left')

    # Rellenar UsoServiciosExtra con False en NaNs
    df_merge['UsoServiciosExtra'] = df_merge['UsoServiciosExtra'].fillna(False)

    # Seleccionar columnas que comienzan con 'Concepto_' y rellenar NaN con 0, luego a bool
    concepto_cols = [col for col in df_merge.columns if col.startswith('Concepto_')]
    if concepto_cols:
        df_merge[concepto_cols] = df_merge[concepto_cols].fillna(0).astype(bool)

    # Columnas numéricas que deben rellenarse con 0 para evitar NaNs
    columnas_a_rellenar = [
        'Importe_2024_servicios', 'Cantidad_2024_servicios',
        'Importe_2025_servicios', 'Cantidad_2025_servicios',
        'Importe_total_pagado_servicios', 'Cantidad_total_pagado_servicios',
        'TipoServicio_ENTRENAMENTS PERSONALS', 'TipoServicio_FISIOTERÀPIA',
        'TipoServicio_NUTRICIÓ', 'Total_conceptos_unicos',
        'Total_tipos_servicios_unicos'
    ]

    for col in columnas_a_rellenar:
        if col in df_merge.columns:
            df_merge[col] = df_merge[col].fillna(0)

    return df_merge

def crear_features_servicios(df, columnas_a_eliminar=None):
    """
    Crea features resumen sobre uso y ratios de servicios y elimina columnas indicadas.
    
    Parámetros:
    - df: DataFrame con datos mergeados de usuarios y servicios
    - columnas_a_eliminar: lista de columnas a eliminar después de crear los features (opcional)
    
    Retorna:
    - df con nuevas columnas y columnas eliminadas según parámetro
    """

    # Evitar división por cero sumando 1 en denominadores
    df["ratio_importe_2025_2024"] = df["Importe_2025_servicios"] / (df["Importe_2024_servicios"] + 1)
    df["ratio_cantidad_2025_2024"] = df["Cantidad_2025_servicios"] / (df["Cantidad_2024_servicios"] + 1)

    # Variables booleanas de uso de servicios por patrones en nombres de columnas
    df["Uso_entrenamientos_personales"] = (
        df.filter(regex="ENTRENADOR PERSONAL|ENTRENAMENTS PERSONALS").sum(axis=1) > 0
    )

    df["Uso_fisioterapia"] = (
        df.filter(regex="FISIO|FISIOTERÀPIA").sum(axis=1) > 0
    )

    df["Uso_nutricion"] = (
        df.filter(regex="NUTRI").sum(axis=1) > 0
    )

    # Diversidad: cuántos tipos de servicio usa (de esas 3 categorías)
    df["Diversidad_servicios_extra"] = (
        df[["Uso_entrenamientos_personales", "Uso_fisioterapia", "Uso_nutricion"]].sum(axis=1)
    )

    # Eliminar columnas indicadas si se pasan
    if columnas_a_eliminar:
        cols_existentes = [col for col in columnas_a_eliminar if col in df.columns]
        df = df.drop(columns=cols_existentes)

    return df

def unir_con_features_economia(df: pd.DataFrame, df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Une el DataFrame original con el DataFrame de features económicos agregados
    por la columna 'IdPersona'. Convierte 'IdPersona' a string en ambos DataFrames
    antes de hacer la unión.
    
    Retorna una copia del DataFrame resultante para evitar efectos colaterales.
    """

    df = df.copy()
    df_agg = df_agg.copy()

    # Asegurar que 'IdPersona' sea string en ambos DataFrames
    df['IdPersona'] = df['IdPersona'].astype(str)
    df_agg['IdPersona'] = df_agg['IdPersona'].astype(str)

    # Unir los DataFrames por 'IdPersona'
    union = pd.merge(df, df_agg, on='IdPersona', how='left')

    return union.copy()


def preparar_df_union_economia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el DataFrame unido con features económicos, rellenando valores nulos en columnas
    categóricas y numéricas según reglas definidas.
    
    Parámetros:
        df (pd.DataFrame): DataFrame unido que contiene las columnas a preparar.
        
    Retorna:
        pd.DataFrame: DataFrame limpio y preparado.
    """
    df = df.copy()
    
    # Rellenar columna booleana 'TienePagos' con False donde haya NaN
    if 'TienePagos' in df.columns:
        df['TienePagos'] = df['TienePagos'].fillna(False)
    
    # Columnas categóricas a rellenar con 'Indefinido'
    categorical_cols = ['AbonoMasRecuente']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Indefinido')
    
    # Columnas numéricas a rellenar con 0
    cols_a_0 = [
        'NumPagosEconomia', 'MediaIntervaloMeses', 'StdIntervaloMeses',
       'UltimoPagoMesEconomia', 'PrimerPagoMesEconomia',
       'MesesConPagoEconomia', 'PorcentajeMesesPagados', 'TotalPagadoEconomia',
       'media_pagoEconomia', 'StdPagoEconomia', 'CoefVarPagoEconomia',
       'PctPagoMetalico', 'PctPagoRecibo', 'PctPagoTarjeta',
       'PctPagoTransferencia', 'NumAbonosDistintos',
    ]
    for col in cols_a_0:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def unir_df_con_features_accesos(df: pd.DataFrame, df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Une el DataFrame original con el DataFrame de features accesos agregados
    por la columna 'IdPersona'. Convierte 'IdPersona' a string en ambos DataFrames
    antes de hacer la unión.
    
    Retorna una copia del DataFrame resultante para evitar efectos colaterales.
    """

    df = df.copy()
    df_agg = df_agg.copy()

    # Asegurar que 'IdPersona' sea string en ambos DataFrames
    df['IdPersona'] = df['IdPersona'].astype(str)
    df_agg['IdPersona'] = df_agg['IdPersona'].astype(str)

    # Unir los DataFrames por 'IdPersona'
    union = pd.merge(df, df_agg, on='IdPersona', how='left')

    return union.copy()


def preparar_final_accesos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el DataFrame con variables de visitas y comportamiento,
    rellenando valores nulos en:
    - columnas numéricas con 0
    - columnas booleanas con False

    Parámetros:
        df (pd.DataFrame): DataFrame original con columnas de comportamiento.

    Retorna:
        pd.DataFrame: DataFrame limpio y preparado.
    """
    df = df.copy()

    # Columnas numéricas a rellenar con 0
    columnas_numericas = [
        'TotalVisitas', 'DiasActivo', 'TiempoActivoDias', 'VisitasUlt30', 'VisitasUlt90',
        'VisitasUlt180', 'PropUlt90', 'DiasDesdeUltima', 'tasa_caida_visitas',
        'DuracionMediaTotal', 'DuracionMediaUlt90', 'DeltaDuracionUlt90VsTotal',
        'StdDiasEntreVisitas', 'FrecuenciaModal', 'MaxRachaSinVisita',
        'MaxRachaConVisita', 'SemanasConVisitaRatio', 'HoraMediaAcceso',
        'PropVisitasFinDeSemana', 'PropVisitasFestivo', 'VisitasPrimerTrimestre',
        'VisitasUltimoTrimestre'
    ]

    for col in columnas_numericas:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Columnas booleanas a rellenar con False
    columnas_booleanas = [
        'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
        'DiaFav_miércoles', 'DiaFav_sábado', 'DiaFav_viernes',
        'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera',
        'EstFav_verano', 'TieneAccesos'
    ]

    for col in columnas_booleanas:
        if col in df.columns:
            df[col] = df[col].fillna(False)

    return df