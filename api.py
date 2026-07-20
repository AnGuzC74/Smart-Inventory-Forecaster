import os
import sys
import logging
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.analyzer import AnalizadorInventario  # noqa: E402 (requiere sys.path.insert previo)

app = FastAPI(title="MLOps Inventory Backend - Cloud IA Ready")

analizador = None

# CARGA DE ARTEFACTOS
try:
    ruta_data = os.path.join(BASE_DIR, "data")
    analizador = AnalizadorInventario(ruta_data)
    analizador.dm.procesar_inventario()

    if analizador.dm.data is not None:
        # Usar la lógica centralizada de segmentación K-Means
        analizador.dm.data = analizador.segment_kmeans(analizador.dm.data)
except Exception as e:
    logger.error(f"❌ Error crítico en arranque del backend: {e}", exc_info=True)

class Consulta(BaseModel):
    precio: float

@app.post("/predict")
def predecir(request: Consulta):
    if analizador is None:
        raise HTTPException(status_code=500, detail="El motor analizador no está inicializado.")

    try:
        stock = analizador.predecir_stock(request.precio)

        # Recuperar R2 para el agente
        try:
            payload = joblib.load(analizador.ruta_modelo)
            r2 = payload.get('r2_score', 0.0)
        except Exception as e:
            logger.error(f"Error al cargar R2score para el agente en predict: {e}", exc_info=True)
            r2 = 0.0

        insight = analizador.obtener_recomendacion_agente(request.precio, stock, r2)
        return {"sugerencia": int(stock), "ia_insight": insight}
    except Exception as e:
        logger.error(f"Error al realizar predicción: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
