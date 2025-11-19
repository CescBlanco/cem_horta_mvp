from google import genai
from dotenv import load_dotenv
import os
import streamlit as st
import time
import random


load_dotenv()  # Cargar variables de entorno desde el archivo .env

clave_api_geminai = os.getenv("clave_api_geminai")

client = genai.Client(api_key=clave_api_geminai)


def generar_resumen_gemini(probabilidad, nivel, drivers, acciones):
    drivers_text = "\n".join(
        [f"- {d['variable']}: impacto {d['impacto']} (valor={d['valor']}, contribución={d['shap']:.3f})"
         for d in drivers]
    )
    acciones_text = "\n".join([f"- {a}" for a in acciones])

    prompt = f"""
    Genera un resumen ejecutivo de 6–8 líneas para un directivo no técnico.

    1. Primera línea: una frase breve indicando probabilidad de abandono y nivel de riesgo.
    2. Salto de línea.
    3. Segundo párrafo: explica los factores que aumentan y reducen el riesgo. No menciones SHAP. Frases directas.
    4. Salto de línea.
    5. Tercer párrafo: recomendaciones accionables, claras y orientadas a negocio.

    NO añadas frases introductorias como “Aquí tienes el resumen” ni cierres tipo “en conclusión”.

    Datos del abonado:
    - Probabilidad de abandono: {probabilidad:.2%}
    - Nivel de riesgo: {nivel}

    Principales factores del riesgo:
    {drivers_text}

    Acciones recomendadas:
    {acciones_text}
    Redacta **exactamente** con la estructura solicitada.

    """
    
    # --- RETRY AUTOMÁTICO ---
    for intento in range(5):
        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            error_text = str(e).lower()
            if any(keyword in error_text for keyword in ["overload", "busy", "quota", "rate", "temporarily"]):
                espera = (2 ** intento) + random.random()
                print(f"Gemini saturado. Reintentando en {espera:.1f}s...")
                time.sleep(espera)
                continue
            else:
                raise e

    raise RuntimeError("Gemini sigue saturado tras varios intentos.")

def tarjeta_resumen(resumen_texto):
    resumen_html = resumen_texto.replace("\n", "<br>")

    st.markdown("""
        <style>
        .tarjeta-ejecutiva {
            background: rgba(255, 255, 255, 0.04);   /* Transparencia ligera */
            border: 1px solid rgba(255, 255, 255, 0.10);
            backdrop-filter: blur(6px);              /* Efecto glass */
            padding: 20px 22px;
            border-radius: 10px;
            font-size: 1.05rem;
            line-height: 1.55;
            color: #e6e6e6;                          /* Texto suave */
            margin-top: 15px;
        }

        .tarjeta-ejecutiva h3 {
            margin-top: 0;
            margin-bottom: 12px;
            color: #58a6ff;                          /* Azul GitHub Dark */
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="tarjeta-ejecutiva">
            <h3>📄 Resumen Ejecutivo</h3>
            {resumen_html}
        </div>
    """, unsafe_allow_html=True)