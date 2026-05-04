import streamlit as st
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import joblib
from sklearn.cluster import KMeans

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Importación de lógica local
try:
    from core.analyzer import AnalizadorInventario
except ImportError:
    st.error("No se encontró la carpeta 'core'. Asegúrate de subirla a la rama.")

st.set_page_config(page_title="Olist Smart Forecaster - Cloud IA", layout="wide")

# Caché de recursos
@st.cache_resource
def inicializar_sistema():
    ruta_data = os.path.join(BASE_DIR, "data")
    analizador = AnalizadorInventario(ruta_data)
    analizador.dm.procesar_inventario()
    
    if analizador.dm.data is not None:
        analizador.dm.data = analizador.dm.data.drop_nulls()
        if len(analizador.dm.data) >= 3 and 'cluster_km' not in analizador.dm.data.columns:
            try:
                X = analizador.dm.data.select(['precio_medio', 'unidades_vendidas']).to_pandas()
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X)
                analizador.dm.data = analizador.dm.data.with_columns(cluster_km = pl.Series(clusters))
            except: pass
    return analizador

analizador = inicializar_sistema()

# Interfaz
st.title("📦 Olist Smart Inventory Forecaster")

tab1, tab2 = st.tabs(["🔮 Predicción con IA Cloud", "📈 Análisis de Segmentación"])

with tab1:
    precio = st.number_input("Precio del producto ($):", min_value=1.0, value=150.0)
    if st.button("Ejecutar Predicción"):
        with st.spinner("Consultando motor ML e IA Cloud..."):
            stock = analizador.predecir_stock(precio)
            
            # Recuperar R2 para el prompt
            try:
                payload = joblib.load(analizador.ruta_modelo)
                r2 = payload.get('r2_score', 0.0)
            except: r2 = 0.0
            
            insight = analizador.obtener_recomendacion_agente(precio, stock, r2)
            
            st.metric("Stock Sugerido", f"{int(stock)} uds")
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
