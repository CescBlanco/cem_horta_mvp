import pandas as pd
from src.rutas import DATA_PATH, VALIDATION_OUTPUT_PATH

def cargar_datos(nombre_experimento: str, features: list) -> tuple:
    """
        Carga y prepara el conjunto de datos, separando en entrenamiento y validación.

        Parámetros:
            nombre_experimento (str): Nombre del experimento, usado para guardar el conjunto de validación.
            features (list): Lista de columnas que se utilizarán como características para el entrenamiento del modelo.

        Retorna:
            tuple: Una tupla con:
                - pd.DataFrame: DataFrame con las características seleccionadas para el entrenamiento.
                - pd.Series: Serie con la variable objetivo 'Abandono' para el entrenamiento.

        Guardado:
            El conjunto de validación se guarda en un archivo CSV en `VALIDATION_OUTPUT_PATH` con el nombre
            `df_validacion_{nombre_experimento}.csv`.

        Excepciones:
            FileNotFoundError: Si no se encuentra el archivo de datos en `DATA_PATH`.
            KeyError: Si alguna de las columnas esperadas no está en el DataFrame cargado.
        """

    # Cargar los datos.
    df = pd.read_csv(DATA_PATH)

    # Filtrar datos de personas mayores de 18 años.    
    df = df[df['Edad'] >= 18].reset_index(drop=True)

    # Renombrar columnas.
    df = df.rename(columns={'EsChurn': 'Abandono','DiaFav_miércoles': 'DiaFav_miercoles','DiaFav_sábado': 'DiaFav_sabado'})

    def separacion_df_inferencia(df: pd.DataFrame) -> tuple:

        """
        Separa el DataFrame en dos conjuntos: entrenamiento y validación.

        Parámetros:
            df (pd.DataFrame): DataFrame con los datos a dividir.

        Retorna:
            tuple: Una tupla con:
                - pd.DataFrame: Conjunto de validación.
                - pd.DataFrame: Conjunto de entrenamiento.
        """
        # Separar por clase
        df_0 = df[df['Abandono'] == 0]
        df_1 = df[df['Abandono'] == 1]

        # Cantidad del 10% para cada clase
        n = int(0.10 * len(df))

        # Sample aleatorio (sin reemplazo)
        valid_0 = df_0.sample(n=n, random_state=42)
        valid_1 = df_1.sample(n=n, random_state=42)

        # Concatenar para tener el 20% de validación balancead
        df_valid = pd.concat([valid_0, valid_1]).reset_index(drop=True)

        #Crear conjunto de entrenamiento excluyendo los de validación
        df_train = df.drop(df_valid.index).reset_index(drop=True)
        return df_valid, df_train
   
    # Dividir los datos en entrenamiento y validación.
    df_valid, df_train = separacion_df_inferencia(df)
    
    # Guardar el conjunto de validación.
    df_valid.to_csv(f"{VALIDATION_OUTPUT_PATH}/df_validacion_{nombre_experimento}.csv", index=False)

    # Retornar las características y la variable objetivo.
    return df_train[features], df_train['Abandono']