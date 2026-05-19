import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

st.set_page_config(
    page_title="Predicción de aprobación estudiantil",
    layout="centered"
)

st.title("Predicción de aprobación estudiantil")
st.write(
    "Esta aplicación carga un Pipeline registrado en MLflow. "
    "Por eso se ingresan las columnas originales del dataset, no las variables encodeadas."
)

# Conexión al servidor MLflow
mlflow.set_tracking_uri("http://127.0.0.1:9090")

# Cambie la versión si MLflow registra una nueva versión del modelo
MODEL_URI = "models:/arboles_estudiantes/1"

@st.cache_resource
def cargar_modelo():
    return mlflow.sklearn.load_model(MODEL_URI)

model = cargar_modelo()

st.sidebar.header("Configuración")
st.sidebar.write(f"Modelo cargado: `{MODEL_URI}`")

st.subheader("Datos del estudiante")

col1, col2 = st.columns(2)

with col1:

    carrera = st.selectbox(
        "Carrera",
        [
            "Ingeniería",
            "Medicina",
            "Derecho",
            "Arquitectura",
            "Administración"
        ]
    )

    modalidad = st.selectbox(
        "Modalidad",
        [
            "Presencial",
            "Virtual"
        ]
    )

    beca = st.selectbox(
        "¿Tiene beca?",
        [
            "Si",
            "No"
        ]
    )

with col2:

    edad = st.number_input(
        "Edad",
        min_value=16,
        max_value=60,
        value=20
    )

    promedio = st.number_input(
        "Promedio",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    asistencias = st.number_input(
        "Porcentaje de asistencias",
        min_value=0,
        max_value=100,
        value=80
    )

datos = pd.DataFrame([{
    "carrera": carrera,
    "modalidad": modalidad,
    "beca": beca,
    "edad": edad,
    "promedio": promedio,
    "asistencias": asistencias
}])

st.subheader("Datos enviados al modelo")
st.dataframe(datos)

if st.button("Predecir"):

    prediccion = model.predict(datos)[0]

    if hasattr(model, "predict_proba"):

        proba = model.predict_proba(datos)[0]

        prob_no = proba[0]
        prob_si = proba[1]

    else:

        prob_no = None
        prob_si = None

    if prediccion == 1:

        st.success(
            "Predicción: el estudiante será APROBADO."
        )

    else:

        st.warning(
            "Predicción: el estudiante NO será aprobado."
        )

    if prob_si is not None:

        st.write(
            f"Probabilidad de NO aprobación: {prob_no:.4f}"
        )

        st.write(
            f"Probabilidad de APROBACIÓN: {prob_si:.4f}"
        )

st.caption(
    "Nota: la aplicación utiliza las columnas originales del dataset para realizar la predicción."
)