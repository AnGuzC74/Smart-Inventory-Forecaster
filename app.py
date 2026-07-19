import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import joblib
import math
import logging

# Configurar logger
logger = logging.getLogger(__name__)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Importación de lógica local
try:
    from core.analyzer import AnalizadorInventario
except ImportError as e:
    logger.error(f"No se encontró la carpeta 'core'. {e}", exc_info=True)
    st.error("No se encontró la carpeta 'core'. Asegúrate de subirla a la rama.")

st.set_page_config(page_title="Olist Smart Forecaster - Cloud IA", layout="wide")

# Caché de recursos
@st.cache_resource
def inicializar_sistema():
    ruta_data = os.path.join(BASE_DIR, "data")
    analizador = AnalizadorInventario(ruta_data)
    analizador.dm.procesar_inventario()

    if analizador.dm.data is not None:
        # Usamos el nuevo método de segmentación K-Means de la clase InventoryAnalyzer
        analizador.dm.data = analizador.segment_kmeans(analizador.dm.data)

    return analizador

analizador = inicializar_sistema()

# Interfaz
st.title("📦 Olist Smart Inventory Forecaster")

# 5. Aviso de datos sintéticos de respaldo (persistente en la interfaz)
if getattr(analizador.dm, "synthetic_mode", False):
    st.warning("⚠️ **Modo de Respaldo Activo:** El sistema está operando actualmente con **datos sintéticos simulados** debido a que los datasets reales de Olist no se encontraron o falló su procesamiento.")

tab1, tab2 = st.tabs(["🔮 Predicción con IA Cloud", "📈 Análisis de Segmentación"])

with tab1:
    precio = st.number_input("Precio del producto ($):", min_value=1.0, value=150.0)
    if st.button("Ejecutar Predicción"):
        with st.spinner("Consultando motor ML e IA Cloud..."):
            stock = analizador.predecir_stock(precio)

            # Calcular intervalo de confianza usando varianza entre árboles
            tiene_intervalo = False
            inf, sup = 0.0, 0.0
            try:
                _, inf, sup = analizador.predecir_con_intervalo(precio)
                tiene_intervalo = True
            except Exception as e:
                logger.error(f"Error al predecir intervalo en la interfaz: {e}", exc_info=True)

            # Recuperar R2 para el prompt
            try:
                payload = joblib.load(analizador.ruta_modelo)
                r2 = payload.get('r2_score', 0.0)
            except Exception as e:
                logger.error(f"Error al cargar R2score para el agente: {e}", exc_info=True)
                r2 = 0.0

            insight = analizador.obtener_recomendacion_agente(precio, stock, r2)

            # Mostrar métricas con columnas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stock Sugerido", f"{int(stock)} uds")
            if tiene_intervalo:
                with col2:
                    st.metric("Límite Inferior (Intervalo)", f"{math.ceil(inf)} uds")
                with col3:
                    st.metric("Límite Superior (Intervalo)", f"{math.ceil(sup)} uds")

            st.success(f"**Análisis del Agente:** {insight}")

with tab2:
    if st.button("Ver Comparativa Pareto vs K-Means"):
        if analizador.dm.data is not None:
            df = pd.DataFrame(analizador.dm.data.to_dicts())
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Pareto (Negocio)", "K-Means (ML)"))
            df_p = df.groupby("clase_pareto")["unidades_vendidas"].sum().reset_index()
            fig.add_trace(go.Bar(x=df_p["clase_pareto"], y=df_p["unidades_vendidas"], name="Pareto"), row=1, col=1)
            if "cluster_km" in df.columns:
                df_k = df.groupby("cluster_km")["unidades_vendidas"].sum().reset_index()
                fig.add_trace(go.Bar(x=df_k["cluster_km"].astype(str), y=df_k["unidades_vendidas"], name="K-Means"), row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)
