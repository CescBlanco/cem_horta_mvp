from eda_feature_engineering.funciones import *

def run():

# ------------------------------------------------------------------
            # PREPARACIÓN DE ABONADOS CON PARTE ECONOMICA
# ------------------------------------------------------------------

    # Cargamos el archivo de clientes con parte economia.
    economia = load_dataset('data/Economia per persona 01.09.2024 a 01.09.2025.xlsx')

    # Gestion de columnas y filas.
    columnas_a_eliminar = ['IdRemesa', 'FormaPagoMetálicoCantidad','FormaPagoMetálicoImporte','FormaPagoRecibo_domiciliadoCantidad', 
        'FormaPagoRecibo_domiciliadoImporte', 'FormaPagoTarjeta_créditoCantidad','FormaPagoTarjeta_créditoImporte',
         'FormaPago_Transf_HortaEsportivaCantidad','FormaPago_Transf_HortaEsportivaImporte', 'TotalCantidad', 'TotalImporte']

    columnas_a_renombrar = {'IdUsuario': 'IdPersona','FormaPagoMetálicoImporteCobrado': 'PagoMetálico','FormaPagoRecibo_domiciliadoImporteCobrado': 'PagoRecibo',
        'FormaPagoTarjeta_créditoImporteCobrado': 'PagoTarjeta','FormaPago_Transf_HortaEsportivaImporteCobrado': 'PagoTransferencia',  'TotalImporteCobrado': 'TotalCobrado'}

    columnas_numericas = ['PagoMetálico',
        'PagoRecibo', 'PagoTarjeta', 'PagoTransferencia', 'TotalCobrado']
    columnas_fechas = ['FechaRenovacion']

    economia_eda = preparar_datos_iniciales(economia, columnas_a_eliminar, columnas_a_renombrar,
        columnas_numericas,   columnas_fechas)
    economia_eda['IdPersona'] = economia_eda['IdPersona'].astype(str)

    # Análisis exploratorio básico (EDA): nulos, duplicados, tipos de variable.
    eda_basica(economia_eda, nombre_df="Clientes Economia")

    # Gestión de nulos.
    economia_eda= economia_eda.fillna(0)

    #  Excluimos los tipos de abono no relevantes para el análisis.
    economia_eda_filtrado= excluir_valores(economia_eda, 'TipoAbono', TIPOS_ABONO_EXCLUIR)

    # Creamos copia para crear las features de comportamiento y ordenamos el dataframe por ID del aboando y la fecha de renovación.
    df_features = economia_eda_filtrado.copy()
    df_features = df_features.sort_values(['IdPersona', 'FechaRenovacion'])

    # Creamos features de comportamiento para los socios con economia en fecha inicio y final de la muestra.
    df_features_economia = df_features.groupby('IdPersona').apply(agregar_features_economia).reset_index()
    
    # Añadimos nueva columna de "TienePagos" para diferenciar posteriormente quien tiene y quien no.
    df_features_economia['TienePagos'] = True

    # Gestión final de nulos.
    df_features_economia= df_features_economia.fillna(0)

    # Guardar el DataFrame en un archivo CSV
    df_features_economia.to_csv('data/economia_final_pre_modelo.csv', index=False)
    
    return df_features_economia

