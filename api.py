import os
import sys
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.cluster import KMeans

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.analyzer import AnalizadorInventario

app = FastAPI(title="MLOps Inventory Backend - Cloud IA Ready")

analizador = None

def inyectar_clustering(df):
    try:
        X = df.select(['precio_medio', 'unidades_vendidas']).to_pandas()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        import polars as pl
        df = df.with_columns(
            cluster_km = pl.Series(kmeans.fit_predict(X))
        )
        return df
    except Exception as e:
        print(f"❌ Error al inyectar clustering: {e}")
        return df

# CARGA DE ARTEFACTOS
try:
    ruta_data = os.path.join(BASE_DIR, "data")
    analizador = AnalizadorInventario(ruta_data)
    analizador.dm.procesar_inventario()
    
    if 'cluster_km' not in analizador.dm.data.columns:
        analizador.dm.data = inyectar_clustering(analizador.dm.data)
except Exception as e:
    print(f"❌ Error crítico en arranque: {e}")

class Consulta(BaseModel):
    precio: float

@app.post("/predict")
def predecir(request: Consulta):
    stock = analizador.predecir_stock(request.precio)
    
    # Recuperar R2 para el agente
    try:
        payload = joblib.load(analizador.ruta_modelo)
        r2 = payload.get('r2_score', 0.0)
    except:
        r2 = 0.0
        
    insight = analizador.obtener_recomendacion_agente(request.precio, stock, r2)
    return {"sugerencia": int(stock), "ia_insight": insight}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
