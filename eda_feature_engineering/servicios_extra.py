from eda_feature_engineering.funciones import *

def run():

# ------------------------------------------------------------------
            # PREPARACIÓN DE ABONADOS CON SERVICIOS
# ------------------------------------------------------------------

    # Cargamos el archivo de clientes con servicios extra.
    df_servicios = load_dataset('data/Fisio-Nutri 01.09.2024 a 01.09.2025 FORMATO BUENO.xlsx', sheet_name="adaptado")

    # Gestion de columnas y filas.
    columnas_a_eliminar = []
    columnas_a_renombrar = {}
    columnas_numericas = ['IdPersona','Importe_2024_servicios','Cantidad_2024_servicios', 'Importe_2025_servicios',
                    'Cantidad_2025_servicios', 'Importe_total_pagado_servicios', 'Cantidad_total_pagado_servicios']
    columnas_fechas= []

    df_servicios = preparar_datos_iniciales(df_servicios, columnas_a_eliminar, columnas_a_renombrar,
        columnas_numericas,   columnas_fechas)
    
    # Análisis exploratorio básico (EDA): nulos, duplicados, tipos de variable.
    eda_basica(df_servicios, nombre_df="Clientes con servicios extra")

    #Gestón de nulos.
    df_servicios=df_servicios.fillna(0)
    
    # Creamos features de comportamiento para los socios con servicios extra en fecha inicio y final de la muestra.
    df_servicios_agregados = agregar_servicios(df_servicios)

    # Codificamos ciertas columnas con tipado de booleano.
    df_servicios_encoded = aplicar_one_hot_encoding_servicios(df_servicios)

    # Unión del dataframe codificado con la de las features creadas de los servicios extra.
    df_servicios_final = agregar_y_unir_servicios(df_servicios_encoded, df_servicios_agregados)
    
    # Acabamos de preparar el dataframe de servicios extra para los siguientes procesos. 
    df_servicios_final = preparar_servicios_final(df_servicios_final)

    # Guardar el DataFrame en un archivo CSV
    df_servicios_final.to_csv('data/servicios_final.csv', index=False)

    return df_servicios_final