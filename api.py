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

app = FastAPI(title="MLOps Inventory Backend - Cluster Fix")

analizador = None

def inyectar_clustering(df):
    """
    Función de emergencia para crear la columna cluster_km 
    si el DataManager no lo hizo.
    """
    try:
        # Usamos precio_medio y unidades_vendidas para el clustering
        X = df.select(['precio_medio', 'unidades_vendidas']).to_pandas()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df = df.with_columns(
            cluster_km = kmeans.fit_predict(X)
        )
        print("✅ MLOps: Columna 'cluster_km' generada mediante parche.")
        return df
    except Exception as e:
        print(f"❌ Error al inyectar clustering: {e}")
        return df

# CARGA DE ARTEFACTOS
try:
    print("⚙️ Iniciando Pipeline de MLOps...")
    ruta_data = os.path.join(BASE_DIR, "data")
    analizador = AnalizadorInventario(ruta_data)
    
    # Ejecución del proceso base
    analizador.dm.procesar_inventario()
    
    # VALIDACIÓN Y PARCHE
    if 'cluster_km' not in analizador.dm.data.columns:
        print("⚠️ 'cluster_km' no detectada. Aplicando corrección en caliente...")
        analizador.dm.data = inyectar_clustering(analizador.dm.data)

    print(f"📊 Columnas finales en RAM: {analizador.dm.data.columns}")
except Exception as e:
    print(f"❌ Error crítico: {e}")

class Consulta(BaseModel):
    precio: float

@app.get("/comparativa")
def obtener_comparativa():
    return {"data": analizador.dm.data.to_dicts()}

@app.post("/predict")
def predecir(request: Consulta):
    stock = analizador.predecir_stock(request.precio)
    url_ia = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": f"Precio: {request.precio}, Stock: {stock}. Directiva corta.",
        "stream": False
    }
    res_ia = requests.post(url_ia, json=payload, timeout=60).json()
    return {"sugerencia": int(stock), "ia_insight": res_ia.get("response")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
