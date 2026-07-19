# 📦 Olist Smart Inventory & AI Forecaster (Consolidación Final) 🚀

Este repositorio contiene la versión consolidada y recomendada del sistema híbrido de inteligencia artificial para la gestión y predicción de inventarios industriales. Integra modelos de Machine Learning tradicionales, segmentación estadística y capacidades avanzadas de MLOps con interfaces y agentes de razonamiento.

## 🌟 Características de la Versión Consolidada

1.  **Dataset Real de Olist:** Procesamiento masivo de datos reales de Olist para la estimación precisa de demanda e inventario, optimizado de forma instantánea gracias al uso de **Polars**.
2.  **Segmentación Dual (Pareto ABC + K-Means Clustering):**
    - **Pareto ABC (Negocio):** Clasificación basada en la ley del 80/20 sobre unidades vendidas.
    - **K-Means Clustering:** Agrupamiento no supervisado calculado dinámicamente sobre precio y demanda.
3.  **Intervalo de Confianza por Varianza de Árboles (Estadísticamente Correcto):** Reemplazo del margen fijo por un método estadístico riguroso utilizando la varianza de las predicciones de los estimadores individuales (`modelo.estimators_`) en el Random Forest (`promedio ± 2×std`).
4.  **Mecanismo de Detección de Drift Real (`verificador_ops.py`):**
    - Guarda una línea base (`baseline_stats.json`) con la media y desviación estándar de la característica principal (`precio_medio` / Costo_Unitario) durante el entrenamiento.
    - Realiza auditorías comparando el lote actual contra la línea base mediante un cálculo de **Z-score** simple. Si el Z-score supera un umbral de 3, se registra y reporta una alerta de Data Drift real en vez de falsas afirmaciones de éxito.
5.  **Aviso Persistente de Modo Simulado:** Si el sistema opera con datos sintéticos de respaldo por falta de archivos o fallas en el procesamiento, se activa un widget `st.warning()` persistente durante toda la sesión en la interfaz de Streamlit (`app.py`).
6.  **Errores Robustos y Logging:** Reemplazo de bloques genéricos `except:` por manejo de excepciones específico y explícito utilizando la librería estándar `logging` de Python.

## 📁 Estructura del Proyecto
```text
.
├── core/
│   ├── __init__.py
│   ├── analyzer.py         # Lógica de ML, predicción con intervalo por árboles y K-Means
│   ├── data_manager.py     # Carga de Olist con fallback sintético visible y Pareto
│   └── visualyzer.py       # Gráficos y análisis visual
├── data/                   # Dataset de Olist
├── models/
│   ├── modelo_inventario_rf.joblib
│   └── baseline_stats.json # Estadísticos de entrenamiento de la línea base
├── app.py                  # Frontend Dashboard en Streamlit con alertas persistentes
├── api.py                  # API en FastAPI para predicción de stock
├── test_backend.py         # Set de pruebas unitarias pytest (Pareto, K-Means, fallback)
├── verificador_ops.py      # Script de auditoría MLOps con verificación de Drift real
├── requirements.txt        # Dependencias del proyecto sincronizadas
└── runtime.txt             # Configuración de entorno Python 3.11
```

## 🛠️ Instalación y Uso Local

1. Instalar las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```

2. Ejecutar la suite de pruebas unitarias para validar que todo funcione correctamente:
   ```bash
   pytest -v
   ```

3. Ejecutar la verificación de MLOps y detección de Drift:
   ```bash
   python verificador_ops.py
   ```

4. Iniciar la interfaz interactiva de Streamlit:
   ```bash
   streamlit run app.py
   ```

## 📄 Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
