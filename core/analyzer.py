import polars as pl
import joblib
import os
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from core.data_manager import DataManager

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class AnalizadorInventario:
    def __init__(self, folder_path):
        self.dm = DataManager(folder_path)
        self.ruta_modelo = os.path.join("models", "modelo_inventario_rf.joblib")
        self.client = None
        
        # Intentar inicializar cliente cloud si hay API Key disponible (Streamlit Secrets o Env)
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def predecir_stock(self, precio_nuevo):
        """Carga el modelo y predice con redondeo superior. Auto-sana si hay error de versión."""
        if not os.path.exists(self.ruta_modelo):
            print("⚠️ Modelo no encontrado. Entrenando...")
            self.entrenar_y_salvar()
        
        try:
            payload = joblib.load(self.ruta_modelo)
            modelo_cargado = payload["model"]
            prediccion = modelo_cargado.predict(np.array([[precio_nuevo]]))[0]
            return math.ceil(prediccion)
        except Exception as e:
            print(f"⚠️ Error de compatibilidad o carga ({e}). Re-entrenando...")
            self.entrenar_y_salvar()
            payload = joblib.load(self.ruta_modelo)
            return math.ceil(payload["model"].predict(np.array([[precio_nuevo]]))[0])

    def obtener_recomendacion_agente(self, precio, prediccion, r2):
        """Consulta al agente IA Cloud con enfoque de optimización logística."""
        instrucciones_sistema = (
            "Actúa como un Gerente de Logística senior. "
            "Tu objetivo es optimizar el inventario y evitar quiebres de stock. "
            "Da directivas claras y breves basadas en los datos proporcionados. "
            "No saludes, ve directamente a la recomendación operativa en máximo 20 palabras."
        )

        prompt_usuario = (
            f"DATOS TÉCNICOS: Precio ${precio} | Stock Sugerido: {prediccion} unidades | R²: {r2:.2f}. "
            "Dicta la estrategia de reabastecimiento inmediata."
        )
        
        # Caso Cloud
        if self.client:
            try:
                model_name = os.environ.get("CLOUD_MODEL_NAME", "gpt-3.5-turbo")
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": instrucciones_sistema},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    max_tokens=50
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"⚠️ Error Cloud IA: {e}"
        
        # Fallback si no hay Cloud configurado
        return "💡 Recomendación Logística: Mantener niveles de seguridad basados en la predicción de stock calculada."

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
