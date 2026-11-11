
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Energía Inteligente CFE",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energía Inteligente CFE")
st.write("Sistema de análisis y detección de anomalías en el consumo eléctrico.")

@st.cache_data
def cargar_datos():
    return pd.read_csv("resultados_consumo_cfe.csv")

try:
    df = cargar_datos()
except FileNotFoundError:
    st.error("No se encontró resultados_consumo_cfe.csv. Verifica que esté en el repositorio.")
    st.stop()

df["fecha"] = pd.to_datetime(df["fecha"])

# Menú lateral
st.sidebar.title("Menú principal")
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    ["Visión general", "Exploración interactiva", "Simulador / Inferencia"]
)

# SECCIÓN 1
if seccion == "Visión general":
    st.header("1. Visión general")

    total = len(df)
    medidores = df["medidor_id"].nunique()
    consumo = df["consumo_kwh"].mean()
    anomalias = (df["resultado_modelo"] == 0).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{total:,}")
    c2.metric("Medidores", medidores)
    c3.metric("Consumo promedio", f"{consumo:.2f} kWh")
    c4.metric("Anomalías detectadas", int(anomalias))

    st.subheader("Consumo eléctrico por fecha")
    grafica = px.line(
        df.groupby("fecha", as_index=False)["consumo_kwh"].mean(),
        x="fecha",
        y="consumo_kwh",
        title="Consumo promedio por fecha"
    )
    st.plotly_chart(grafica, use_container_width=True)

    st.info("Los datos utilizados son sintéticos y se emplean con fines académicos.")

# SECCIÓN 2
elif seccion == "Exploración interactiva":
    st.header("2. Exploración interactiva")

    zonas = ["Todas"] + sorted(df["zona"].dropna().unique().tolist())
    zona = st.selectbox("Selecciona una zona:", zonas)

    usuarios = ["Todos"] + sorted(df["tipo_usuario"].dropna().unique().tolist())
    usuario = st.selectbox("Tipo de usuario:", usuarios)

    filtrado = df.copy()

    if zona != "Todas":
        filtrado = filtrado[filtrado["zona"] == zona]

    if usuario != "Todos":
        filtrado = filtrado[filtrado["tipo_usuario"] == usuario]

    st.write(f"Registros encontrados: {len(filtrado):,}")

    grafica = px.scatter(
        filtrado,
        x="fecha",
        y="consumo_kwh",
        color="zona",
        title="Consumo eléctrico por zona"
    )
    st.plotly_chart(grafica, use_container_width=True)

    st.subheader("Datos filtrados")
    st.dataframe(filtrado, use_container_width=True)

# SECCIÓN 3
elif seccion == "Simulador / Inferencia":
    st.header("3. Simulador de consumo eléctrico")

    st.write("Introduce los datos de un consumo para estimar si presenta un comportamiento inusual.")

    consumo_nuevo = st.number_input(
        "Consumo eléctrico (kWh):",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    zona_nueva = st.selectbox("Zona:", ["Norte", "Sur", "Centro"])
    usuario_nuevo = st.selectbox("Tipo de usuario:", ["Residencial", "Comercial"])
    hora_nueva = st.slider("Hora del día:", 0, 23, 12)
    dia_nuevo = st.slider("Día del mes:", 1, 30, 15)

    if st.button("Analizar consumo"):
        datos = df[
            ["consumo_kwh", "zona_num", "usuario_num", "hora", "dia"]
        ].copy()

        escalador = StandardScaler()
        datos_escalados = escalador.fit_transform(datos)

        modelo = IsolationForest(
            n_estimators=100,
            contamination=0.03,
            random_state=42
        )
        modelo.fit(datos_escalados)

        zona_codigo = {"Norte": 0, "Sur": 1, "Centro": 2}
        usuario_codigo = {"Residencial": 0, "Comercial": 1}

        nuevo = pd.DataFrame([{
            "consumo_kwh": consumo_nuevo,
            "zona_num": zona_codigo[zona_nueva],
            "usuario_num": usuario_codigo[usuario_nuevo],
            "hora": hora_nueva,
            "dia": dia_nueva
        }])

        nuevo_escalado = escalador.transform(nuevo)
        resultado = modelo.predict(nuevo_escalado)[0]

        if resultado == -1:
            st.warning("El consumo presenta un comportamiento inusual. Se recomienda revisarlo.")
        else:
            st.success("El consumo presenta un comportamiento habitual según el modelo.")

        st.caption("Esta simulación es demostrativa y utiliza datos sintéticos; no determina fraude ni sustituye una revisión técnica.")

