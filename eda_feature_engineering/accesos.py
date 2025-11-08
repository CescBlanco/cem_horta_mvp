from eda_feature_engineering.funciones import *

def run():


# ------------------------------------------------------------------
            # PREPARACIÓN DE ABONADOS CON ACCESOS
# ------------------------------------------------------------------

    # Cargamos los diferentes archivos de accesos de los abonados a lo largo del intervalo de tiempo estudiado.
    accesos24 = load_dataset('data/Accessos 2024.xlsx')
    accesos = load_dataset('data/Copia de Accessos 01.01.2025 a 30.04.2025.xlsx')
    accesos2 = load_dataset('data/Copia de Accessos 01.05.2025 a 31.07.2025.xlsx')
    accesos3 = load_dataset('data/Copia de Accessos 01.08.2025 a 01.09.2025.xlsx')    
    
    # Concatenamos los diferentes archivos en uno solo
    df_accesos = concatenar_dataframes(accesos24, accesos, accesos2, accesos3)

    # Gestion de columnas y filas.
    columnas_a_eliminar = ['HEntrada', 'HSalida']
    columnas_a_renombrar = {}
    columnas_numericas = []
    columnas_fechas = []

    df_accesos_eda = preparar_datos_iniciales(df_accesos, columnas_a_eliminar, columnas_a_renombrar,
        columnas_numericas,   columnas_fechas)

    # Análisis exploratorio básico (EDA): nulos, duplicados, tipos de variable.
    eda_basica(df_accesos_eda, nombre_df="Clientes Accesos")

    # Tratamiento de fechas y horas para posteriormente crear las features necesarias.
    df_accesos_preparado = preparar_fechas_y_horas(df_accesos_eda)

    # Creamos features de comportamiento para los socios de sus accesos en fecha inicio y final de la muestra.
    df_features_accesos= calcular_features_accesos(df_accesos_preparado)

    # Volvemos a realizar un EDA para verificar que todo está correcto
    eda_basica(df_features_accesos, nombre_df="Clientes Accesos")

    # Guardar el DataFrame en un archivo CSV
    df_features_accesos.to_csv('data/resumen_accesos_pre_modelo.csv', index=False)


    return df_features_accesos