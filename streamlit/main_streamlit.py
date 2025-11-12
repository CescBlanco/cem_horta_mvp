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

from funciones_streamlit import *

col1, colspace, col3 = st.columns([1,3,1])


with col1:
    # Mostrar el logo del gimnasio con un ancho específico de 175 píxeles
    st.image(LOGO_GYM, width=175)

with col3:
    # Mostrar el logo del ayuntamiento con un ancho específico de 175 píxeles
    st.image(LOGO_AYUNTAMIENTO, width=175)

# Configuración de la página en Streamlit: título y layout
st.set_page_config(page_title="App de Predicción de Abandono", layout="wide")

# Usar Markdown para centrar el título de la página y darle color
st.markdown("<h1 style='text-align: center; color: #66BB6A;'>Predicción de Abandono: CEM Horta Esportiva</h1>", unsafe_allow_html=True)

# Crear una barra de pestañas (tabs) para las diferentes opciones de la aplicación
tabs = st.tabs([":bar_chart: Abonados con datos inventados", ":id: Un abonado", ":memo: Múltiples abonados", ":mag: Valoración modelos"])

# ------------------- #
# TAB 1: Datos individuales
# ------------------- #

with tabs[0]:
    
    # Título de la sección de entrada de datos
    st.markdown("<h2 style='color: #888;'>📝 Datos de Entrada del Abonado</h2>", unsafe_allow_html=True)
    
    # Texto solicitando que se ingresen los datos del abonado para realizar la predicción
    st.write("Por favor, ingresa los datos del abonado simulado para realizar la predicción.")

    # Llamada a la función input_userdata() para obtener los datos del usuario (esto es un supuesto, la función debería existir)
    userdata = input_userdata()  # Suponiendo que esta función obtiene los datos del usuario
    
    st.write('----')

    # Título para la sección de predicción
    st.markdown("<h3 style='color: #888;'>🔮 Realizar la predicción</h3>", unsafe_allow_html=True)

    # Instrucción para hacer clic en el botón para realizar la predicción
    st.write("Haz clic en el botón para realizar la predicción.")

    # Botón para iniciar la predicción para un abonado inventado
    if st.button("🚀 Iniciar predicción para el abonado.", key="btn_individual"):
        
        # Crear un contenedor vacío para el mensaje de "Calculando..."
        calculating_message = st.empty()
        # Mostrar el mensaje de "Calculando..."
        calculating_message.write("Calculando las predicciones de abandono... ")
         # Esperamos un segundo antes de borrar el mensaje
        time.sleep(2)
        # Ahora, también borramos el mensaje de "Calculando..."
        calculating_message.empty()
        
        # Crear una lista vacía para almacenar los resultados de la predicción
        resultados = []

        # Convertir los valores booleanos a True/False, sin cambiar a 1/0
        # Esto se hace para asegurar que el modelo recibe los datos en el formato adecuado (True/False)
        for col in BOOL_COL:
            if col in userdata:
                userdata[col] = True if userdata[col] else False
        
        # Verificar que los datos del usuario contienen todas las columnas necesarias para la predicción
        required_columns = set(COLUMNAS_MODELO) # Establece las columnas necesarias para el modelo
        
        if not required_columns.issubset(userdata.keys()):  # Si falta alguna columna
            st.error("⚠️ Faltan algunas columnas necesarias.")
        else:
            # Realizar la predicción
            response = obtener_predicciones_api("predecir_abandono_socio_simulado/", userdata)
            
            # Usamos st.empty() para crear un contenedor vacío
            success_message = st.empty()
            # Mostrar el mensaje de éxito
            success_message.success("✅ Predicción obtenida")
            # Esperamos un segundo antes de borrar el mensaje
            time.sleep(1)
            # Borramos el mensaje de éxito
            success_message.empty()

            # Verificar que la respuesta sea un diccionario (es la forma esperada)            
            if isinstance(response, dict):  # Verifica que la respuesta es un diccionario
                res = response # Almacena la respuesta en la variable 'res'
                res['IdPersona'] = res.get('IdPersona', 'Simulado') # Si no existe 'IdPersona', asigna un valor simulado
                probabilidad = res.get("ProbabilidadAbandono", 0) # Obtiene la probabilidad de abandono (valor por defecto es 0)
                nivel_riesgo = res.get("NivelRiesgo", "Desconocido") # Obtiene el nivel de riesgo (valor por defecto es "Desconocido")
                

                # Verifica que los datos de probabilidad y nivel de riesgo sean válidos
                if probabilidad is not None and nivel_riesgo:

                    # Usar la función 'color_con_riesgo' para obtener el color y el nivel de riesgo
                    color, nivel = color_con_riesgo(probabilidad)
                    
                    # Mostrar la probabilidad de abandono y el nivel de riesgo con el color correspondiente
                    st.markdown(
                        f"""
                        <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center; color:black; font-size:24px'>
                            Probabilidad de abandono: {probabilidad:.2%} (Nivel de Riesgo: {nivel_riesgo})
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:

                    # Si no se encuentra la probabilidad o el nivel de riesgo, mostrar un mensaje de advertencia
                    st.warning("❌ No se encontró información sobre la probabilidad de abandono.")
                
                # Mostrar un espacio vacío para separar secciones
                st.markdown(" ")
                st.markdown("### Lo que impacta en la probabilidad del abonado:")
                st.markdown(" ")

                # Verificar si la respuesta contiene la clave 'CaracterísticasImportantes', que es donde están las variables del modelo
                if "CaracterísticasImportantes" in response:
                    
                    # Llamar a la función que prepara el dataframe de las variables más importantes
                    df_top_filtered = preparar_df_importancias(response)
                    
                    # Llamar a la función para generar el gráfico de barras con las importancias de las variables
                    fig_importancias_abonado = plot_abonado_importancias(df_top_filtered)
                    
                    # Mostrar el gráfico generado con Streamlit
                    st.pyplot(fig_importancias_abonado)

                    # Generar un resumen sobre las variables que afectan al riesgo de abandono
                    frase_resumen = generar_frase_resumen(df_top_filtered, nivel)
            
                    # Mostrarla en Streamlit
                    st.markdown(f"**Resumen del riesgo**: {frase_resumen}")

                    
                    # --- 3. Explicación del modelo ---
                    st.markdown("")
                
                    # Título para la sección donde se explica el comportamiento del modelo y el riesgo de abandono
                    st.markdown("### Comportamiento del riesgo: ")
                    
                    # Llamada a la función que genera la explicación detallada de las variables que afectan al riesgo
                    generar_explicacion_contexto(df_top_filtered)

                st.subheader(" ")
                st.subheader("Acción de fidelización: ")
                
                # --- 4. Estrategias de fidelización ---

                # Obtener el 'IdPersona' del abonado y su 'NivelRiesgo' desde la respuesta de la predicción
                id_persona = response.get("IdPersona")
                nivel_riesgo = response.get("NivelRiesgo")
    
                # Verificar si el nivel de riesgo está en las estrategias de fidelización definidas en ESTRATEGIAS_FIDELIZACION
                if nivel_riesgo in ESTRATEGIAS_FIDELIZACION:

                    # Usar un 'expander' en Streamlit para mostrar las estrategias de fidelización solo si el nivel de riesgo tiene estrategias
                    with st.expander(f"Estrategias de fidelización para el abonado con ID **{id_persona}** (Nivel de Riesgo: {nivel_riesgo})"):
                
                        # Iterar sobre las estrategias disponibles para el nivel de riesgo y mostrarlas con Markdown
                        for estrategia in ESTRATEGIAS_FIDELIZACION[nivel_riesgo]:
                            st.markdown(estrategia)
                    
                    # Mostrar una animación de globos (para indicar algo positivo o una acción exitosa)
                    st.balloons()
                else:
                    # Si no se encontraron estrategias para el nivel de riesgo, mostrar una advertencia
                    st.warning(f"No se encontraron estrategias para el nivel de riesgo: {nivel_riesgo}")   
  
            # Si no se obtuvo una respuesta válida para la predicción, mostrar un mensaje de error
            else:
                st.warning(f"❌ Error en la predicción para IdPersona simulado")

# ------------------- #
# TAB 1: Un ID
# ------------------- #
with tabs[1]:

    # Título para la sección de predicción por un abonado
    st.markdown("<h2 style='color: #888;'>Predicción por un abonado</h2>", unsafe_allow_html=True)

    # Campo para ingresar el ID del abonado (un número entero)
    id_persona = st.number_input("Introduce el ID del cliente", min_value=0, step=1)

    st.write('----')

    # Subtítulo para la sección de predicción
    st.markdown("<h3 style='color: #888;'>🔮 Realizar la predicción</h3>", unsafe_allow_html=True)

    # Instrucciones para el usuario sobre el botón de predicción
    st.write("Haz clic en el botón para realizar la predicción.") 

    # Botón que inicia la predicción cuando es presionado
    if st.button("🚀 Iniciar predicción por un abonado", key="btn_id"):
        
        # Crear un contenedor vacío para el mensaje de "Calculando..."
        calculating_message = st.empty()
        # Mostrar el mensaje de "Calculando..."
        calculating_message.write("Calculando las predicciones de abandono... ")
         # Esperamos un segundo antes de borrar el mensaje
        time.sleep(2)
        # Ahora, también borramos el mensaje de "Calculando..."
        calculating_message.empty()

        # Crear un diccionario con el ID de la persona que el usuario ha introducido
        data = {"IdPersona": id_persona}

        # Realizar la llamada a la API para obtener la predicción usando el ID de la persona
        response = obtener_predicciones_api("predecir_abandono_por_id/", data)

        # Si no se obtiene respuesta de la API, mostrar un mensaje de error
        if not response:
            st.error("⚠️ No se obtuvo respuesta de la API.")
        else:
             # Usamos st.empty() para crear un contenedor vacío
            success_message = st.empty()
            # Mostrar el mensaje de éxito
            success_message.success("✅ Predicción obtenida")
            # Esperamos un segundo antes de borrar el mensaje
            time.sleep(1)
            # Borramos el mensaje de éxito
            success_message.empty()

            # Intentar parsear la respuesta si es una cadena JSON (puede ser en formato string)
            try:
                if isinstance(response, str):
                    response = json.loads(response)  # Convertir la respuesta de JSON a un diccionario
            except json.JSONDecodeError as e:
                st.error(f"Error al parsear la respuesta JSON: {e}")
                response = None
            
            # Si la respuesta es válida, continuar con el procesamiento de los datos
            if response:
                # --- 1. Probabilidad de abandono ---
                probabilidad = response.get("ProbabilidadAbandono", 0)   # Obtener la probabilidad de abandono desde la respuesta             

                # Calcular el color y nivel de riesgo basados en la probabilidad
                color, nivel= color_con_riesgo(probabilidad)
            
                # Mostrar la probabilidad de abandono con el color correspondiente
                st.markdown(
                    f"""
                    <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center; color:black; font-size:24px'>
                        Probabilidad de abandono: {probabilidad:.2%} (Nivel de Riesgo: {nivel})
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Separación para mejorar la presentación
                st.markdown(" ")
                st.markdown("### Lo que impacta en la probabilidad del abonado:")
                st.markdown(" ")
                
                # --- 2. Variables más importantes ---
                # Verificar si la respuesta contiene la clave 'CaracterísticasImportantes' con las variables más relevantes
                if "CaracterísticasImportantes" in response:
                    
                    # Llamar a la función que prepara el dataframe de las variables más importantes
                    df_top_filtered = preparar_df_importancias(response)
                    
                    # Generar el gráfico de las variables más importantes
                    fig_importancias_abonado = plot_abonado_importancias(df_top_filtered)
                    st.pyplot(fig_importancias_abonado)
                 
                    # Generar una frase resumen con las variables que impactan en el riesgo
                    frase_resumen = generar_frase_resumen(df_top_filtered, nivel)
            
                    # Mostrarla en Streamlit
                    st.markdown(f"**Resumen del riesgo**: {frase_resumen}")                  
 
                    st.markdown("")

                    # Título para la sección de explicación del modelo
                    st.markdown("### Comportamiento del riesgo: ")
                    
                    # Llamar a la función que genera la explicación del comportamiento del riesgo
                    generar_explicacion_contexto(df_top_filtered)

                st.subheader(" ")
                st.subheader("Acción de fidelización: ")
                
                # --- 4. Estrategias de fidelización ---
                
                # Obtener el ID de la persona y el nivel de riesgo desde la respuesta
                id_persona = response.get("IdPersona")
                nivel_riesgo = response.get("NivelRiesgo")

                # Verificar si el nivel de riesgo tiene estrategias de fidelización definidas
                if nivel_riesgo in ESTRATEGIAS_FIDELIZACION:

                    # Mostrar las estrategias de fidelización en una sección expandible
                    with st.expander(f"Estrategias de fidelización para el abonado con ID **{id_persona}** (Nivel de Riesgo: {nivel_riesgo})"):
                        for estrategia in ESTRATEGIAS_FIDELIZACION[nivel_riesgo]:
                            st.markdown(estrategia)
                    
                    # Mostrar una animación de globos para indicar que se ha completado la acción
                    st.balloons()

                # Si no existen estrategias para el nivel de riesgo, mostrar un mensaje de advertencia
                else:
                    st.warning(f"No se encontraron estrategias para el nivel de riesgo: {nivel_riesgo}")


# ------------------- #
# TAB 2: Múltiples IDs
# ------------------- #
with tabs[2]:

    # Título para la sección de predicción por múltiples abonados
    st.markdown("<h2 style='color: #888;'>Predicción para múltiples abonados</h2>", unsafe_allow_html=True)

    # Campo de entrada para que el usuario introduzca una lista de IDs separados por comas
    ids_input = st.text_area("Introduce los diferentes IDs de los abonados, separados por comas", value="123,456,789")

    # Subtítulo para la sección de predicción
    st.markdown("<h3 style='color: #888;'>🔮 Realizar la predicción</h3>", unsafe_allow_html=True)

    # Instrucciones para el usuario sobre el botón de predicción
    st.write("Haz clic en el botón para realizar la predicción.")

    # Botón para iniciar la predicción para múltiples abonados
    if st.button("🚀 Iniciar predicción por múltiples abonados", key="btn_ids"):

        # Crear un contenedor vacío para el mensaje de "Calculando..."
        calculating_message = st.empty()
        # Mostrar el mensaje de "Calculando..."
        calculating_message.write("Calculando las predicciones de abandono... ")
         # Esperamos un segundo antes de borrar el mensaje
        time.sleep(2)
        # Ahora, también borramos el mensaje de "Calculando..."
        calculating_message.empty()

        try:
            
            # Obtener los IDs de los abonados desde el input del usuario
            # Convertir cada ID a un entero, eliminando cualquier espacio extra
            ids_list = [int(id_.strip()) for id_ in ids_input.split(",") if id_.strip()]
            data = {"Ids": ids_list}
            
            # Obtener la respuesta de la API para la predicción
            response = obtener_predicciones_api("predecir_abandono_por_ids/", data)
            
            # Si no se obtuvo respuesta de la API, mostrar un mensaje de error
            if not response:
                st.error("⚠️ No se obtuvo respuesta de la API.")
            
            else:
                all_success = True 
                # Procesar cada predicción de forma independiente
                for prediccion in response:
                    # Verificar si la respuesta contiene algún error para este ID
                    if "error" in prediccion:  # Si encontramos un error en alguno de los abonados
                        st.error(f"⚠️ El ID {prediccion.get('IdPersona')} no es válido: {prediccion.get('error')}")
                        st.error(f"⚠️ Por favor... ingrese los IDs de los abonados que existan!")
                        continue  # Continuar con la siguiente predicción (si la hay)
             
           

                    # Usamos st.empty() para crear un contenedor vacío
                    success_message = st.empty()
                    # Mostrar el mensaje de éxito
                    success_message.success("✅ Predicción obtenida")
                    # Esperamos un segundo antes de borrar el mensaje
                    time.sleep(1)
                    # Borramos el mensaje de éxito
                    success_message.empty()

                    # Asegurarse de que la predicción tenga la estructura correcta
                    id_persona = prediccion.get("IdPersona")
                    nivel_riesgo = prediccion.get("NivelRiesgo")
                    
                    st.write("---")
                    st.write(f"### Predicción para el abonado con ID {id_persona}")

                    # --- 1. Probabilidad de abandono ---
                    probabilidad = prediccion.get("ProbabilidadAbandono", 0)  # Obtener la probabilidad de abandono de la predicción
                    color, nivel = color_con_riesgo(probabilidad) # Obtener el color y nivel de riesgo según la probabilidad
                    
                    st.markdown(
                        f"""
                        <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center; color:black; font-size:24px'>
                            Probabilidad de abandono: {probabilidad:.2%} (Nivel de Riesgo: {nivel})
                        </div>
                        """,
                        unsafe_allow_html=True)
                    
                    # --- 2. Variables más importantes ---

                    # Verificar si la respuesta contiene las características importantes
                    if "CaracterísticasImportantes" in prediccion:

                        df_top_filtered = preparar_df_importancias(prediccion) # Preparar el dataframe de variables importantes
                        fig_importancias_abonado = plot_abonado_importancias(df_top_filtered) # Graficar las variables importantes
                        st.pyplot(fig_importancias_abonado)  # Mostrar el gráfico

                        # Generar un resumen de riesgo basado en las variables
                        frase_resumen = generar_frase_resumen(df_top_filtered, nivel)
                        st.markdown(f"**Resumen del riesgo**: {frase_resumen}")

                    # --- 3. Explicación del modelo ---
                    # Explicar el comportamiento del riesgo del abonado
                    st.markdown("### Comportamiento del riesgo: ")
                    generar_explicacion_contexto(df_top_filtered) # Llamar a la función que genera la explicación

                    # --- 4. Estrategias de fidelización ---

                    # Verificar si existen estrategias de fidelización para este nivel de riesgo
                    if nivel_riesgo in ESTRATEGIAS_FIDELIZACION:
                        
                        # Mostrar las estrategias de fidelización en una sección expandible
                        with st.expander(f"Estrategias de fidelización para el abonado con ID **{id_persona}** (Nivel de Riesgo: {nivel_riesgo})"):
                            for estrategia in ESTRATEGIAS_FIDELIZACION[nivel_riesgo]:
                                st.markdown(estrategia)
                        
                        st.balloons() # Mostrar globos como animación
                    else:
                        
                        # Si no existen estrategias, mostrar un mensaje de advertencia
                        st.warning(f"No se encontraron estrategias para el nivel de riesgo: {nivel_riesgo}")
                # Si todas las predicciones fueron exitosas, mostramos el mensaje de éxito
                if all_success:
                    # Usamos st.empty() para crear un contenedor vacío
                    success_message = st.empty()
                    # Mostrar el mensaje de éxito
                    success_message.success("✅ Predicción obtenida")
                    # Esperamos un segundo antes de borrar el mensaje
                    time.sleep(1)
                    # Borramos el mensaje de éxito
                    success_message.empty()

        except ValueError:
            # Si el usuario introduce un valor no válido (por ejemplo, letras en lugar de números), mostrar un error
            st.error("⚠️ Por favor, introduce solo números separados por comas")

# ------------------- #
# TAB 3: Valoración
# ------------------- #
with tabs[3]: 
        RUN_ID_INFERENCIA_1 = obtener_run_id_inferencias(NAME_EXPERIMENT_1)
        RUN_ID_INFERENCIA_2 = obtener_run_id_inferencias(NAME_EXPERIMENT_2)
        RUN_ID_INFERENCIA_3 = obtener_run_id_inferencias(NAME_EXPERIMENT_3)

        # Leer los archivos de los diferentes experimentos de inferencia
        df_archivo_global_exp3, df_archivo_persona_ex3, df_archivo_preds_ex3 = encontrar_csv_inferencias(NAME_EXPERIMENT_3, FOLDER_DESTINO_3, RUN_ID_INFERENCIA_3)
        df_archivo_global_exp2, df_archivo_persona_ex2, df_archivo_preds_ex2 = encontrar_csv_inferencias(NAME_EXPERIMENT_2, FOLDER_DESTINO_2, RUN_ID_INFERENCIA_2)
        df_archivo_global_exp1, df_archivo_persona_ex1, df_archivo_preds_ex1 = encontrar_csv_inferencias(NAME_EXPERIMENT_1, FOLDER_DESTINO_1, RUN_ID_INFERENCIA_1)

        # Obtener las métricas de rendimiento del modelo para el experimento 3 (AUC, accuracy, F1, recall)
        auc_exp3, accuracy_exp3, f1_exp3, recall_exp3= encontrar_metricas_experimento(NAME_EXPERIMENT_3, metric=METRIC)
        accuracy, auc, f1, recall= encontrar_metricas_inferencia(RUN_ID_INFERENCIA_3)
        
        # Crear una opción de radio para que el usuario elija la vista: 'Mostrar modelo entrenado' o 'Mostrar modelo post inferencia'
        view_option = st.radio("Elige la vista:", ("Mostrar modelo entrenado", "Mostrar modelo post inferencia"), horizontal=True)

        # Si el usuario elige 'Mostrar modelo entrenado'
        if view_option == 'Mostrar modelo entrenado':

            # Leer el archivo CSV con el modelo inicial entrenado (archivo con datos históricos)
            file_path_inicial =  'data\dataframe_final_abonado.csv'

            df_modelo_inicial = pd.read_csv(file_path_inicial)
            
            # Título y justificación del Experimento 1
            st.markdown("<h3 style='color: #888;'>Justificación para experimento 1 (No usado):</h3>", unsafe_allow_html=True)
                     
            col1_exp1, col2_exp1 = st.columns(2)

            # Gráfico 1: Analizar la variable 'TotalPagadoEconomia' del modelo
            with col1_exp1:
                fig_exp_1= plots_experimentos_sinuso(df_modelo_inicial, 'TotalPagadoEconomia')
                st.pyplot(fig_exp_1)

            # Gráfico 2: Mostrar la importancia de las características para el experimento 1
            with col2_exp1:
                fig_importnacias_exp1= plot_importancias(df_archivo_global_exp1)
                # Mostrar gráfico en Streamlit
                st.pyplot(fig_importnacias_exp1)
            
            st.markdown(' ')
            st.markdown("""
                🧑‍💻📊 Interpretación técnica:
                        
                - A partir de un pago de 600€, la probabilidad de abandono es casi nula.
                - Esto sugiere que el modelo está aprendiendo un patrón determinista: si un usuario paga más de 600, se clasifica automáticamente como activo.
                - **`"TotalPagadoEconomía"`**  domina el modelo, dejando de lado otras variables relevantes.
                - Esta dominancia lleva a una clasificación sesgada y menos precisa, especialmente para usuarios que pagan menos pero cuyo comportamiento de abandono depende de otros factores.
                - **`Decisión`** : Se decide eliminar esta variable para evitar el sesgo y permitir que el modelo considere mejor otras variables.        
                         """)

            # Justificación del Experimento 2
            st.markdown("<h3 style='color: #888;'>Justificación para experimento 2 (No usado):</h3>", unsafe_allow_html=True)
        
            col1_exp2, col2_exp2 = st.columns(2)

            with col1_exp2:

                # Gráfico 1: Analizar la variable 'VidaGymMeses' del modelo
                fig_exp_2= plots_experimentos_sinuso(df_modelo_inicial, 'VidaGymMeses')
                st.pyplot(fig_exp_2)
    
            with col2_exp2:

                # Gráfico 2: Mostrar la importancia de las características para el experimento 2
                fig_importnacias_exp2= plot_importancias(df_archivo_global_exp2)
                # Mostrar gráfico en Streamlit
                st.pyplot(fig_importnacias_exp2)

            st.markdown(' ')
            st.markdown("""
                🧑‍💻📊 Interpretación técnica:
                        
                - Los usuarios que abandonan tienden a tener menos meses de suscripción, mientras que los que permanecen activos tienen más tiempo en el gimnasio.
                - El modelo podría estar aprendiendo un patrón determinista: si un cliente tiene más de un valor específico de meses (aproximadamente 150 meses), se clasifica automáticamente como no abandono.
                - Este patrón podría hacer que el modelo se sobreajuste, ignorando otras variables importantes.
                - **`Decisión`**: Se decide prescindir de esta variable para evitar que el modelo dependa de este valor umbral y así mejorar la inclusión de otras características.        
                """)
    #-----------------------------------------------------------------------------------------------------------------------------------
 
            # Justificación para la elección del Experimento 3
            st.markdown("<h3 style='color: #888;'>Justificación para la elección del experimento 3:</h3>", unsafe_allow_html=True)       
            
            # Mostrar las métricas del modelo (AUC, Accuracy, F1, Recall)
            st.markdown(f"""
                Rendimiento del modelo:
                        
                - **`AUC`**: {auc_exp3} → Excelente capacidad de distinguir entre quienes se quedan y quienes abandonan.
                - **`Accuracy`**: {accuracy_exp3} → Modelo fiable en general.
                - **`F1-score`**: {f1_exp3} → Buen equilibrio entre evitar falsos positivos y capturar verdaderos abandonos.
                - **`Recall`**: {recall_exp3} → Detecta casi 8 de cada 10 abonados que realmente abandonarían.

                **`Valor de negocio`**: Permite dirigir campañas de retención de manera efectiva, priorizando a los abonados en riesgo.

                **`Comparativa`**: Este experimento supera a otros modelos porque maximiza la detección de abandonos sin generar demasiadas falsas alarmas.
                         """)
            
             #   Visualización de la importancia de las variables para el experimento 3
            st.markdown("<h3 style='color: #888;'>Visualización de la importancia de las variables para el modelo:</h3>", unsafe_allow_html=True)
            
            
            fig_importnacias_exp3= plot_importancias(df_archivo_global_exp3)
            # Mostrar gráfico en Streamlit
            st.pyplot(fig_importnacias_exp3)

           
            # Crear un expander (expandir contenido) para mostrar más información técnica
            with st.expander("🔍 Más información sobre la importancia de variables"):
                
                # **Parte técnica - Data Science:**
                st.markdown("""🧑‍💻📊 Interpretación técnica:""")

                st.markdown("""
                - **`DiasActivo`**: La cantidad de días que un abonado ha estado activo es la **variable más importante**. Los abonados con menos días activos tienen una mayor probabilidad de abandonar.
                - **`TotalVisitas`**: La cantidad total de visitas realizadas por un abonado es otro factor crucial. Un abonado que visita con regularidad es menos probable que abandone.
                - **`Edad`**: La edad del abonado tiene una relación importante con el abandono. Diferentes rangos de edad podrían tener diferentes comportamientos en cuanto a su lealtad y retención.
                - **`VisitasPrimerTrimestre`**: Las visitas en el primer trimestre de la suscripción son un indicador clave. Un alto número de visitas en este periodo podría predecir una mayor retención a largo plazo.
                - **`VisitasUlt180` y `VisitasUlt90`**: Las visitas recientes (últimos 180 y 90 días) también son importantes. Si un abonado ha estado menos activo recientemente, es más probable que abandone.
                - **`TienePagos`**: El abonado que ha realizado pagos regularmente es menos probable que abandone. Esto podría indicar un mayor compromiso o satisfacción con el servicio.
                """)
                st.markdown("""💼📈 Interpretación práctica de negocio:""")
            
                # **Parte de negocio - Interpretación práctica:**
                st.markdown("""
                - **`DiasActivo`**: Un abonado con pocos días activos debería ser un objetivo prioritario para campañas de retención, ya que es más probable que abandone. Considera ofrecer incentivos para aumentar la actividad.
                - **`TotalVisitas`**: Los abonados con pocas visitas son más propensos a abandonar. Para ellos, una estrategia podría ser ofrecer promociones de visitas o recordatorios personalizados.
                - **`Edad`**: La edad es una variable importante a tener en cuenta. Dependiendo del grupo de edad, las preferencias y expectativas sobre el servicio pueden variar. Los jóvenes podrían necesitar una experiencia más dinámica, mientras que los abonados más mayores podrían estar más interesados en un enfoque centrado en el bienestar.
                - **`VisitasPrimerTrimestre`**: Aprovecha la oportunidad en los primeros tres meses para enganchar a los nuevos abonados con una buena experiencia. Esto aumentará las probabilidades de retención a largo plazo.
                - **`VisitasUlt180` y `VisitasUlt90`**: Los abonados que no han visitado recientemente podrían estar en riesgo de abandonar. Ofrecer promociones o actividades especiales para reactivar su participación puede ser una estrategia clave.
                - **`TienePagos`**: Los abonados que pagan regularmente son menos propensos a abandonar. Tal vez se podría utilizar esta información para premiar a los abonados más fieles con beneficios exclusivos.
                """)

            st.markdown("<h3 style='color: #888;'>📝 Resumen del visual:</h3>", unsafe_allow_html=True)       


            # Sección visual con texto resumido
            st.markdown(""" 
            - **Las variables más importantes para predecir el abandono son**:
                - **`DiasActivo`**: El tiempo de actividad del abonado es crucial. Los abonados con menos días activos tienen mayor probabilidad de abandonar.
                - **`TotalVisitas`**: Un abonado que asiste más al centro está menos propenso a abandonar.
                - **`Edaad`**: La edad también influye en la probabilidad de abandono. Diferentes rangos de edad podrían tener diferentes comportamientos.
                
                        
            - **Las variables con menor impacto son**:
                - **`Diversidad_servicios_extra`** y **`UsoServiciosExtra`**: Aunque los servicios extra tienen su valor, el comportamiento central (actividad y visitas) es más importante.
            
                        
            - **Conclusiones finales**:
                - **`Mayor compromiso = menor probabilidad de abandono`**: Mantener a los usuarios activos y reactivar a los inactivos es clave.
                - **`Pagos = fuerte predictor de retención`**: Incrementar la conversión a pagos y cuidar la experiencia de los usuarios pagos reduce significativamente el abandono.
                - **`Visitas frecuentes = retención`**: Incentivar la actividad continua es clave para mantener a los abonados.
                - **`Usuarios jóvenes = mayor riesgo`**: Necesitan estrategias de retención personalizadas.
                - **`Personalización estacional`**: Promociones según preferencias de estación pueden mejorar la retención, aunque su impacto es menor.
            """)

            st.markdown("<h3 style='color: #888;'> 🔑 Posible recomendación:</h3>", unsafe_allow_html=True)       

            st.markdown("""           
            - **Actúa sobre la actividad del abonado**: Aumentar la actividad y la frecuencia de visitas será clave para retener a los abonados.
            - **Segmenta por edad y actividad**: Crea estrategias personalizadas según el nivel de actividad y la edad para mejorar la retención.
            """)

        # Si el usuario elige 'Mostrar modelo post inferencia'
        elif view_option == 'Mostrar modelo post inferencia':

            # Ruta al archivo CSV que contiene los datos de validación post-inferencia
            file_path = 'data/df_validacion_Experimento_v3.csv'

            
            # Leer el archivo CSV de validación para obtener los datos post-inferencia
            df_validacion = pd.read_csv(file_path)

            # Mostrar un título para la sección de inferencia
            st.markdown("<h2 style='color: #999;'>🧑‍💻📊 Interpretación de la inferencia</h2>", unsafe_allow_html=True)

            col1_inf, col2_inf= st.columns(2)

            with col1_inf:
                
                # Título para mostrar las métricas de rendimiento del modelo
                st.markdown("<h5 style='color: #888;'>Rendimiento de la validación del modelo:</h5>", unsafe_allow_html=True)


                # Mostrar las métricas de rendimiento obtenidas en el modelo post-inferencia
                st.markdown(f"""
                                                
                    - **`AUC`**: {auc} → Muy buena capacidad para diferenciar entre abonados que se quedarán y los que abandonarán.
                    - **`Accuracy`**: {accuracy} → Modelo fiable en general.
                    - **`F1-score`**: {f1} → Mantiene un buen equilibrio entre precisión y detección de abandonos.
                    - **`Recall`**: {recall} → Recall: 84% → Detecta más de 8 de cada 10 abonados que realmente abandonarían, mejorando la identificación de riesgo frente al entrenamiento.

                    **`Comparativa`**: El modelo mantiene un buen equilibrio entre identificar abandonos y evitar falsas alertas, demostrando robustez tras la validación del modelo elegido.
                            """)
          
            # Unir los datos de las predicciones con los datos originales de las personas
            df_persona_exp3 = df_archivo_preds_ex3.merge(df_archivo_persona_ex3, on='IdPersona', how='left')
             # Unir los datos de validación con el DataFrame combinado de predicciones y personas
            df_final_persona = pd.merge(df_persona_exp3, df_validacion[['IdPersona', "DiasActivo", "TotalVisitas", 'Edad', "VisitasPrimerTrimestre", "VisitasUlt180",  "TienePagos", "VisitasUlt90",
                                                                   "VisitasUltimoTrimestre", "EstFav_otono", "EstFav_verano"]], on='IdPersona', how='left', suffixes=('', '_inicial'))
           
            
            # Crear el gráfico de distribución por proporción de abandono según los datos procesados
            with col2_inf:

                st.markdown("<h5 style='color: #888;'>Proporción de clientes por rango de abandono:</h5>", unsafe_allow_html=True)
                
                # Crear un gráfico de pie para mostrar la distribución por edad de los clientes
                fig_piechart= piechart_edad(df_final_persona)
                # Mostrar el  en Streamlit
                st.pyplot(fig_piechart)

            col1results,col2results= st.columns(2)
    
            # Función que cuenta los resultados de clientes activos y abandonados
            grouped_activos_reset, grouped_abandonados_reset=  tabla_recuento_resultados(df_final_persona)
  
            with col1results: 
                # Mostrar las tablas separadas en Streamlit
                st.markdown("<h3 style='color: #888;'>Clientes Activos:</h3>", unsafe_allow_html=True)
                st.table(grouped_activos_reset)  # Mostrar tabla de clientes activos

            with col2results:
                st.markdown("<h3 style='color: #888;'>Clientes Abandonados:</h3>", unsafe_allow_html=True)
                st.table(grouped_abandonados_reset)  # Mostrar tabla de clientes abandonados
            
            # Mostrar los factores que afectan la probabilidad de abandono
            st.markdown("<h3 style='color: #888;'>Factores que afectan la probabilidad de abandono:</h3>", unsafe_allow_html=True)

            # Llamar a la función de categorización de variables de importancia
            df_final_persona= categorizacion_variables_importancia(df_final_persona)
            
            # Crear una lista de opciones para elegir qué gráfico mostrar
            opciones = [
                "Probabilidad de Abandono por Grupos de Edad",
                "Probabilidad de Abandono por Grupos de Días Activos",
                "Probabilidad de Abandono por Grupos de Visitas Últimos 180 Días",
                "Probabilidad de Abandono por Visitas Primer Trimestre",
                "Probabilidad de Abandono por Estación Favorita Otoño",
                "Probabilidad de Abandono por si Tiene Pagos"
            ]

            # Crear un selector para que el usuario elija qué gráfico quiere ver
            eleccion = st.selectbox("Elige un gráfico para ver:", opciones)

            # Llamar a la función que muestra el gráfico elegido y su descripción
            mostrar_grafico_y_descripcion(eleccion, df_final_persona)          
           
            
            # Título para la sección de estrategias de fidelización
            st.markdown("<h3 style='color: #888;'>Estrategias de Fidelización:</h3>", unsafe_allow_html=True)
            st.markdown("Selecciona el nivel de riesgo de abandono de los usuarios para ver las estrategias de fidelización recomendadas.")

            

            # Selector para elegir el nivel de riesgo de abandono
            nivel_riesgo = st.selectbox("Selecciona el Nivel de Riesgo:", 
                                        ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"])

            # Mostrar las estrategias según el nivel de riesgo seleccionado
            mostrar_estrategias(nivel_riesgo)

# Agregar un pie de página con los detalles de contacto
st.markdown("""<footer style='text-align:center; font-size:12px; color:#888;'>
    <p> © 2025 Cesc Blanco | Contacto: <a href='mailto:cesc.blanco98@gmail.com'>cesc.blanco98@gmail.com</a> | 
             Sígueme en LinkedIn: <a href='https://www.linkedin.com/in/cescblanco' target='_blank'>LinkedIn</a> </p></footer>""", unsafe_allow_html=True)
