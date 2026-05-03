import polars as pl
import joblib
import os
import math
import ollama
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from core.data_manager import DataManager

class AnalizadorInventario:
    def __init__(self, folder_path):
        self.dm = DataManager(folder_path)
        self.ruta_modelo = os.path.join("models", "modelo_inventario_rf.joblib")

    def predecir_stock(self, precio_nuevo):
        """Carga el modelo y predice con redondeo superior. Auto-sana si hay error de versión."""
        if not os.path.exists(self.ruta_modelo):
            print("⚠️ Modelo no encontrado. Entrenando...")
            self.entrenar_y_salvar()
        
        try:
            payload = joblib.load(self.ruta_modelo)
            modelo_cargado = payload["model"]
            # Intentar predicción
            prediccion = modelo_cargado.predict(np.array([[precio_nuevo]]))[0]
            return math.ceil(prediccion)
        except (AttributeError, ValueError, KeyError) as e:
            # Si el modelo guardado es incompatible con la versión actual de sklearn
            print(f"⚠️ Error de compatibilidad ({e}). Re-entrenando modelo en caliente...")
            self.entrenar_y_salvar()
            # Reintento tras re-entrenar
            payload = joblib.load(self.ruta_modelo)
            modelo_cargado = payload["model"]
            prediccion = modelo_cargado.predict(np.array([[precio_nuevo]]))[0]
            return math.ceil(prediccion)

    def obtener_recomendacion_agente(self, precio, prediccion, r2):
        """Consulta al agente IA con un enfoque de optimización logística."""
        ID_MODELO = 'llama3.2:latest'
        
        instrucciones_sistema = (
            "Actúa como un Gerente de Logística senior. "
            "Tu objetivo es optimizar el inventario y evitar quiebres de stock. "
            "Da directivas claras y breves basadas en los datos proporcionados. "
            "No saludes, ve directamente a la recomendación operativa."
        )

        prompt_usuario = (
            f"DATOS TÉCNICOS: Precio ${precio} | Stock Sugerido: {prediccion} unidades | R²: {r2:.2f}. "
            "Dicta la estrategia de reabastecimiento inmediata."
        )
        
        try:
            response = ollama.chat(model=ID_MODELO, messages=[
                {'role': 'system', 'content': instrucciones_sistema},
                {'role': 'user', 'content': prompt_usuario},
            ])
            return response['message']['content'].strip()
        except Exception as e:
            return f"Aviso del Agente: No se pudo generar recomendación (Ollama offline o error). Detalle: {e}"

    def entrenar_y_salvar(self):
        """Entrena el Random Forest y guarda el payload."""
        if self.dm.data is None:
            self.dm.procesar_inventario()
            
        X = self.dm.data.select(["precio_medio"]).to_numpy()
        y = self.dm.data["unidades_vendidas"].to_numpy()
        
        regr = RandomForestRegressor(n_estimators=100, random_state=42)
        regr.fit(X, y)
        r2 = r2_score(y, regr.predict(X))
        
        os.makedirs("models", exist_ok=True)
        joblib.dump({"model": regr, "r2_score": r2}, self.ruta_modelo)
        return r2
