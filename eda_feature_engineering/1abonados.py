from eda_feature_engineering.funciones import *

def run():

# ------------------------------------------------------------------
            # PREPARACIÓN DE ABONADOS ACTIVOS
# ------------------------------------------------------------------

    # Cargamos el dataframe de abonados activos.
    df = load_dataset('data/Llistat Abonats actius 15.09.2025.xlsx')
    df = df[1:].reset_index(drop=True)

    # Gestion de columnas y filas.
    columnas_a_eliminar = ['FechaBaja', 'MotivoBaja', 'DireccionCompleta', 'CodigoPostal']
    columnas_a_renombrar = {'FechaAlta': 'FechaInscripcion', 'TipoAbono': 'TipoAbonoActual'}
    columnas_numericas = ['Edad']
    columnas_fechas = ['FNacimiento', 'FechaInscripcion', 'FAntiguedad']

    df_eda = preparar_datos_iniciales(df, columnas_a_eliminar, columnas_a_renombrar,
        columnas_numericas,   columnas_fechas)

    # Análisis exploratorio básico (EDA): nulos, duplicados, tipos de variable.
    eda_basica(df_eda, nombre_df="Clientes Activos")

    # Filtrar por 'FAntiguedad'.
    df_features_1sep25 = filtrar_por_fecha(df_eda, 'FAntiguedad', '2025-09-01')

    # Filtramos por 'FechaInscripcion' si quisieras.
    df_features_1sep25 = filtrar_por_fecha(df_eda, 'FechaInscripcion', '2025-09-01')
    
    # Excluimos los tipos de abono no relevantes para el análisis. 
    df_activos_filtrado = excluir_valores(df_features_1sep25, 'TipoAbonoActual', TIPOS_ABONO_EXCLUIR)

    # Codificamos la columna 'Sexo' con one-hot encoding: crea una nueva columna 'Sexo_Mujer' (True si es mujer, False si es hombre).
    df_features_codificado = codificar_one_hot(df_activos_filtrado, 'Sexo')

    # Estandarizamos los tipos de abonos que tienen los abonados.
    df_abonos_limpios = cambios_nombre_abonos( df_features_codificado,  columna='TipoAbonoActual',   mapeo_manual=MAPEO_MANUAL_ABONOS)  

    # Creamos features de comportamiento para los socios activos en fecha inicio y final de la muestra.
    df_final_activos = crear_features_activos(df_abonos_limpios, fecha_corte_inicio='2024-09-01',  fecha_corte='2025-09-01')
    
    # Volvemos a realizar pequeño EDA para verificar que todo es correcto.
    eda_basica(df_final_activos, nombre_df="Clientes Final Activos")

    print('\nArchivo activos bien preparado!! ')


# ------------------------------------------------------------------
            # PREPARACIÓN DE ABONADOS CON ALTA
# ------------------------------------------------------------------

    # Cargamos el archivo de clientes de altas.
    abonado_alta = load_dataset('data/Altes abonats 01.09.2024 a 01.09.2025.xlsx')

    # Gestion de columnas y filas.
    columnas_a_eliminar = ['DireccionCompleta', 'FNacimiento']
    columnas_a_renombrar = {'TipoAbono': 'TipoAbonoAlta', 'FAntiguedad':'FAntiguedadAlta'}
    columnas_numericas = ['Edad']
    columnas_fechas = ['FAntiguedadAlta', 'FAntiguedad']

    altas = preparar_datos_iniciales(abonado_alta, columnas_a_eliminar, columnas_a_renombrar,
        columnas_numericas,   columnas_fechas)
    
    # Análisis exploratorio básico (EDA): nulos, duplicados, tipos de variable.
    eda_basica(altas, nombre_df="Clientes con Altas")

    # Filtrar por 'FAntiguedadAlta'.
    altas = filtrar_por_fecha(altas, 'FAntiguedadAlta', '2025-09-01')

    # Filtramos por 'FechaInscripcion' si quisieras.
    altas_filtrado = filtrar_por_fecha(altas, 'FechaAlta', '2025-09-01')
    
    # Excluimos los tipos de abono no relevantes para el análisis. 
    altas_filtrado = excluir_valores(altas_filtrado, 'TipoAbonoAlta', TIPOS_ABONO_EXCLUIR)
    
    # Estandarizamos los tipos de abonos que tienen los abonados.
    altas_filtrado_limpios = cambios_nombre_abonos( altas_filtrado,  columna='TipoAbonoAlta',   mapeo_manual=MAPEO_MANUAL_ABONOS)
    
    # Creamos features de comportamiento para los socios en altas en fecha inicio y final de la muestra.
    df_altas_features = crear_features_altas_periodo( altas_filtrado_limpios,  fecha_inicio='2024-09-01',  fecha_corte='2025-09-01',
                                                        fecha_col='FechaAlta',    id_col='IdPersona')


    # Unión de dataframe activos con altas.
    df_completo_activos_altas = preparar_union_activos_altas( df_activos=df_final_activos,altas_agg=df_altas_features, id_col='IdPersona',
                                                                fecha_inscripcion_col='FechaInscripcion',  fecha_corte='2025-09-01')


# ------------------------------------------------------------------
            # PREPARACIÓN DE ABONADOS CON BAJA
# ------------------------------------------------------------------
    
    # Cargamos el archivo de clientes con bajas.
    abonado_bajas = load_dataset('data/Baixes Abonats 01.09.2024 a 01.09.2025.xlsx')
    abonado_bajas = abonado_bajas[1:].reset_index(drop=True)

    # Gestion de columnas y filas.
    columnas_a_eliminar = ['DireccionCompleta', 'MotivoBaja','FNacimiento' , 'GENERAL', 'Total']
    columnas_a_renombrar = {'TipoAbono': 'TipoAbonoBaja', 'FAntiguedad':'FAntiguedadBaja', 'FechaAlta':'FechaAltaBaja'}
    columnas_numericas = ['Edad']
    columnas_fechas = ['FAntiguedadAlta', 'FAntiguedad']

    bajas = preparar_datos_iniciales(abonado_bajas, columnas_a_eliminar, columnas_a_renombrar,
        columnas_numericas,   columnas_fechas)

    # Análisis exploratorio básico (EDA): nulos, duplicados, tipos de variable.
    eda_basica(bajas, nombre_df="Clientes con Bajas")

    #  Excluimos los tipos de abono no relevantes para el análisis.
    bajas_filtrado = excluir_valores(bajas, 'TipoAbonoBaja', TIPOS_ABONO_EXCLUIR)

    # Estandarizamos los tipos de abonos que tienen los abonados.
    bajas_filtrado_limpios = cambios_nombre_abonos( bajas_filtrado,  columna='TipoAbonoBaja',   mapeo_manual=MAPEO_MANUAL_ABONOS)
    
    # Codificamos la columna 'Sexo' con one-hot encoding: crea una nueva columna 'Sexo_Mujer' (True si es mujer, False si es hombre).
    bajas_filtrado_limpios = codificar_one_hot(bajas_filtrado_limpios, 'Sexo')

    # Creamos features de comportamiento para los socios en baja en fecha inicio y final de la muestra.
    df_bajas_features = crear_features_bajas_periodo( bajas_filtrado_limpios,  fecha_inicio='2024-09-01',  fecha_corte='2025-09-01',
                                                        fecha_col='FechaBaja',    id_col='IdPersona')

    
    # Unión del dataframe de features con el de bajas.
    bajas_completas = merge_bajas_info( df_bajas_features,  bajas_filtrado_limpios,   id_col='IdPersona',
        cols_a_unir=['IdPersona', 'Edad', 'Sexo_Mujer', 'FAntiguedadBaja', 'FechaAltaBaja', 'TipoAbonoBaja', 'FechaBaja'],
        fecha_orden_col='FechaAltaBaja'
    )

    # Acabamos de preparar el dataframe de bajas para los siguientes procesos. 
    df_bajas_preparadas = preparar_bajas(bajas_completas, col_antiguedad='FAntiguedadBaja',  col_fecha_baja='FechaBaja',
                                        col_fecha_alta='FechaAltaBaja', col_tipo_abono='TipoAbonoBaja')


    #----------------------------------------------------------------------------------

    # Analizar y tratar inconsistencias.
    resultados = analizar_inconsistencias(df_bajas_preparadas, df_completo_activos_altas)

    print("\n Durante el análisis de los datos, se detectaron 356 registros (aproximadamente el 11% del dataframe de bajas y el 7% del dataframe de activos) correspondientes a abonados que aparecen simultáneamente como activos y dados de baja en el mismo periodo. Esta situación es inconsistente con la lógica del negocio, dado que un abonado no debería estar activo y dado de baja a la vez." \
    "\nTras evaluar las posibles causas y la complejidad que implica resolver estas inconsistencias (por ejemplo, abonados que se dieron de baja y volvieron a darse de alta en cortos períodos), se decidió eliminar estos casos para:" \
    "\n-Evitar introducir ruido y confusión en el modelo predictivo." \
    "\n-Simplificar el análisis y garantizar la coherencia de los datos." \
    "\n-Mantener la calidad y fiabilidad del conjunto de datos utilizados para entrenamiento y validación." \
    "\nEsta decisión fue tomada con la consideración de que la proporción de registros eliminados es relativamente baja y no debería afectar significativamente el rendimiento ni la generalización del modelo.")
    
    # Se eliminan las incosnistencias para ambos dataframes, tanto socios activos como bajas.
    df_completo_activos_altas, df_bajas_preparadas = eliminar_inconsistencias(df_completo_activos_altas, df_bajas_preparadas, resultados['ids_en_ambos'])


# ------------------------------------------------------------------
            # PREPARACIÓN ARCHIVO ABONADOS FINAL (ACTIVOS Y BAJAS)
# ------------------------------------------------------------------

    # Preparación dataframe final de abonados.
    df_usuarios_final = preparar_df_final(df_completo_activos_altas, df_bajas_preparadas)

    #-----------------------------------------------------------------------------------------------

    # Guardar el DataFrame en un archivo CSV.
    df_usuarios_final.to_csv('data/abonados_final_pre_modelo.csv', index=False)

    return df_usuarios_final