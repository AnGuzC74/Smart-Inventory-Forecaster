import polars as pl
import joblib
import os
import math
import numpy as np
import logging
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from core.data_manager import DataManager

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Configurar logging estándar a nivel de módulo
logger = logging.getLogger(__name__)

class AnalizadorInventario:
    def __init__(self, folder_path):
        self.dm = DataManager(folder_path)
        self.ruta_modelo = os.path.join("models", "modelo_inventario_rf.joblib")
        self.ruta_baseline = os.path.join("models", "baseline_stats.json")
        self.client = None

        # Intentar inicializar cliente cloud si hay API Key disponible (Streamlit Secrets o Env)
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def predecir_stock(self, precio_nuevo):
        """Carga el modelo y predice con redondeo superior. Auto-sana si hay error de versión."""
        if not os.path.exists(self.ruta_modelo):
            logger.warning("Modelo no encontrado. Entrenando...")
            self.entrenar_y_salvar()

        try:
            payload = joblib.load(self.ruta_modelo)
            modelo_cargado = payload["model"]
            prediccion = modelo_cargado.predict(np.array([[precio_nuevo]]))[0]
            return math.ceil(prediccion)
        except Exception as e:
            logger.error(f"⚠️ Error de compatibilidad o carga del modelo ({e}). Re-entrenando...", exc_info=True)
            self.entrenar_y_salvar()
            try:
                payload = joblib.load(self.ruta_modelo)
                return math.ceil(payload["model"].predict(np.array([[precio_nuevo]]))[0])
            except Exception as inner_e:
                logger.error(f"Error fatal en la predicción tras re-entrenamiento: {inner_e}", exc_info=True)
                raise inner_e

    def predecir_con_intervalo(self, precio_nuevo):
        """Predice el stock sugerido junto con su intervalo de confianza basado en la varianza de los estimadores (árboles)."""
        if not os.path.exists(self.ruta_modelo):
            logger.warning("Modelo no encontrado al intentar predecir con intervalo. Entrenando...")
            self.entrenar_y_salvar()

        try:
            payload = joblib.load(self.ruta_modelo)
            modelo_cargado = payload["model"]
        except Exception as e:
            logger.error(f"Error cargando modelo para predicción con intervalo: {e}. Re-entrenando...", exc_info=True)
            self.entrenar_y_salvar()
            try:
                payload = joblib.load(self.ruta_modelo)
                modelo_cargado = payload["model"]
            except Exception as inner_e:
                logger.error(f"Error fatal al cargar modelo tras re-entrenamiento en predicción con intervalo: {inner_e}", exc_info=True)
                raise inner_e

        try:
            # Obtener predicciones de cada uno de los árboles individuales (estimators)
            X_nuevo = np.array([[precio_nuevo]])
            preds = [dt.predict(X_nuevo)[0] for dt in modelo_cargado.estimators_]
            promedio = float(np.mean(preds))
            std = float(np.std(preds))

            limite_inferior = float(max(0.0, promedio - 2.0 * std))
            limite_superior = float(promedio + 2.0 * std)

            return promedio, limite_inferior, limite_superior
        except Exception as e:
            logger.error(f"Error en la predicción por árboles con intervalo: {e}", exc_info=True)
            raise e

    def segment_kmeans(self, df: pl.DataFrame, n_clusters=3) -> pl.DataFrame:
        """Aplica segmentación K-Means dinámicamente sobre el DataFrame de Polars dado."""
        if df is None or len(df) == 0:
            logger.warning("Intento de segmentación K-Means sobre un DataFrame vacío o Nulo.")
            return df

        try:
            df_clean = df.drop_nulls()
            if len(df_clean) < n_clusters:
                logger.warning(f"No hay suficientes datos ({len(df_clean)}) para realizar K-Means con {n_clusters} clusters.")
                return df

            X = df_clean.select(['precio_medio', 'unidades_vendidas']).to_pandas()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)

            # Devolver df con la nueva columna 'cluster_km'
            return df_clean.with_columns(cluster_km = pl.Series(clusters))
        except Exception as e:
            logger.error(f"Error durante la segmentación K-Means: {e}", exc_info=True)
            return df

    def obtener_recomendacion_agente(self, precio, prediccion, r2):
        """Consulta al agente IA Cloud con enfoque de optimización logística."""
        instrucciones_sistema = (
            "Actúa como un Gerente de Logística senior. "
            "Tu objetivo es optimizar el inventario y evitar quiebres de stock. "
            "Da directivas claras y breves basadas en los datos proporcionados. "
            "No saludes, ve directamente a la recomendación operativa en máximo 20 palabras."
        )

        prompt_usuario = (
            f"DATOS TÉCNICOS: Precio ${precio} | Stock Sugerido: {prediccion:.1f} unidades | R²: {r2:.2f}. "
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
                logger.error(f"Error al consultar el agente de IA en la nube: {e}", exc_info=True)
                return f"⚠️ Error Cloud IA: {e}"

        # Fallback si no hay Cloud configurado
        return "💡 Recomendación Logística: Mantener niveles de seguridad basados en la predicción de stock calculada."

    def entrenar_y_salvar(self):
        """Entrena el Random Forest y guarda el payload y los estadísticos de línea base."""
        if self.dm.data is None:
            self.dm.procesar_inventario()

        X = self.dm.data.select(["precio_medio"]).to_numpy()
        y = self.dm.data["unidades_vendidas"].to_numpy()

        regr = RandomForestRegressor(n_estimators=100, random_state=42)
        regr.fit(X, y)
        r2 = r2_score(y, regr.predict(X))

        os.makedirs("models", exist_ok=True)
        joblib.dump({"model": regr, "r2_score": r2}, self.ruta_modelo)

        # Guardar línea base con la media y desviación estándar de la métrica de costo/precio
        precios = self.dm.data["precio_medio"].to_numpy()
        baseline_stats = {
            "mean": float(np.mean(precios)),
            "std": float(np.std(precios))
        }
        try:
            with open(self.ruta_baseline, "w") as f:
                json.dump(baseline_stats, f, indent=4)
            logger.info("Línea base (baseline_stats.json) guardada correctamente.")
        except Exception as e:
            logger.error(f"Error al guardar línea base: {e}", exc_info=True)

        return r2
