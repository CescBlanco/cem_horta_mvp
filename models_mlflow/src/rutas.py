import os

"""
Este archivo define las rutas para los datos de entrada y salida, así como las características a utilizar 
en los experimentos de predicción de abandono. Las características se organizan en tres conjuntos diferentes 
para realizar experimentos con variaciones en las características del modelo.

1. **Rutas de datos**:
    - `DATA_PATH`: Ruta al archivo CSV que contiene los datos completos de los abonados.
    - `VALIDATION_OUTPUT_PATH`: Ruta donde se guardarán los resultados de validación y otros artefactos generados durante los experimentos.

2. **Conjuntos de características**:
    - `FEATURES_1`: Lista completa de características utilizadas en el primer conjunto de experimentos. Incluye una variedad de variables que describen el comportamiento y las preferencias de los usuarios.
    - `FEATURES_2`: Un subconjunto de `FEATURES_1` que excluye la característica "TotalPagadoEconomia". Esto puede ser útil para experimentos donde se quiera observar el impacto de no usar esta variable.
    - `FEATURES_3`: Un subconjunto de `FEATURES_2` que excluye la característica "VidaGymMeses", permitiendo realizar experimentos sin esta variable.

Estos conjuntos de características se utilizan para realizar entrenamientos y validaciones del modelo con distintas combinaciones de variables.

"""

# Definición de las rutas de los archivos de datos
DATA_PATH = os.path.join( "data", "dataframe_final_abonado.csv")

VALIDATION_OUTPUT_PATH = os.path.join( "data")

# Definición de las características a usar en los diferentes experimentos
FEATURES_1 = [
    'Edad', 'VidaGymMeses', 'Sexo_Mujer', 'UsoServiciosExtra',
    'ratio_cantidad_2025_2024', 'Diversidad_servicios_extra',
    'TotalPagadoEconomia', 'TotalVisitas','TienePagos' ,'DiasActivo',
    'VisitasUlt90', 'VisitasUlt180', 'TieneAccesos', 
    'VisitasPrimerTrimestre', 'VisitasUltimoTrimestre',
    'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
    'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
    'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano'
]

FEATURES_2 = [f for f in FEATURES_1 if f != "TotalPagadoEconomia"]
FEATURES_3 = [f for f in FEATURES_2 if f != "VidaGymMeses"]
