import streamlit as st
import joblib
import numpy as np

# Configuración principal de la aplicación web
st.set_page_config(page_title="Predicciones Académicas", page_icon="📚")

st.title("Predicción de Puntaje según Horas de Estudio 📚")
st.info("**Nota:** En este modelo la nota mas alta es 100 puntos porque así viene en la base de datos. ")
st.write("Mueve el control deslizante para calcular la nota estimada en el examen basado en el tiempo de estudio.")

# --- REQUISITO OBLIGATORIO: Datos del estudiante y Colab ---
st.sidebar.header("Información del Estudiante")
st.sidebar.write("**Nombre:Luis Aron Limache Baldera")
st.sidebar.write("**Profe pongame 20 :D")# Tu preferencia de nombre
st.sidebar.write("**Código ISIL:** [77134350]") 

# IMPORTANTE: Reemplaza este enlace por el link de compartir de tu Colab (modo Lector)
url_colab = "https://colab.research.google.com/drive/1vYNGZt2TeN-zNsgAt25phsEKcxNOFn8I?usp=sharing"
st.sidebar.markdown(f"[🔗 Ver Cuaderno de Código Colab]({url_colab})")
# -----------------------------------------------------------

# Función para cargar el modelo de forma segura
@st.cache_resource
def cargar_modelo():
    # El archivo debe estar dentro de la carpeta 'modelos' en GitHub
    return joblib.load("modelos/modelo_lr.pkl")

try:
    modelo = cargar_modelo()

    # Control interactivo para el usuario (Slider)
    horas = st.slider("Selecciona las horas de estudio diarias:", min_value=1.0, max_value=12.0, value=3.0, step=0.5)

    # Botón para realizar el cálculo
    if st.button("Calcular Puntaje Estimado"):
        # El modelo lineal espera una matriz bidimensional [[valor]]
        prediccion = modelo.predict(np.array([[horas]]))[0]
        
        # Ajustamos los límites lógicos de una nota (entre 0 y 100)
        puntaje_final = min(100.0, max(0.0, prediccion))
        
        st.success(f"🎯 Con **{horas}** horas de estudio, el puntaje estimado es: **{puntaje_final:.2f} puntos**")

except Exception as e:
    st.error(f"Falta cargar el archivo del modelo en GitHub o la ruta es incorrecta: {e}")
