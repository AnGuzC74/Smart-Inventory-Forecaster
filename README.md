# 📦 Olist Smart Inventory & AI Forecaster 🚀

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-inventory-forecaster-i3uaskvwqssdukaccbwmbt.streamlit.app/)

Este proyecto representa una solución integral de **MLOps** y **Business Intelligence** diseñada para la gestión crítica de inventarios industriales (basado en datos reales de consumo de aceite). El sistema no solo predice, sino que razona y segmenta estratégicamente para optimizar la cadena de suministro.

## 🚀 Demo en Vivo
Puedes acceder a la simulación interactiva aquí: [Olist Smart Forecaster App](https://smart-inventory-forecaster-i3uaskvwqssdukaccbwmbt.streamlit.app/)

## 🌟 Pilares Tecnológicos y Valor de Negocio

### 1. Motor de Predicción Robusto (Random Forest)
Utilizamos un modelo de **Random Forest Regressor** para estimar los niveles óptimos de reserva. El sistema cuenta con un mecanismo de **Auto-Heal**, que detecta inconsistencias de versión y re-entrena el modelo en milisegundos para garantizar disponibilidad continua.

### 2. Segmentación Dual: Estrategia vs Ciencia de Datos
El núcleo del análisis compara dos metodologías potentes:
-   **Pareto ABC (Negocio):** Clasificación basada en la ley del 80/20 para identificar productos críticos.
-   **K-Means Clustering (ML No Supervisado):** Agrupamiento algorítmico que descubre patrones ocultos entre precio y demanda.
> **El Valor:** La aplicación permite visualizar si la intuición de negocio (Pareto) coincide con el comportamiento matemático (K-Means), permitiendo decisiones de inventario mucho más precisas.

### 3. Inteligencia Generativa (LLM Agent)
Integración con modelos de lenguaje de última generación (**Llama 3.2 / GPT**) para convertir métricas técnicas en **directivas operativas humanas**. El agente actúa como un Gerente de Logística Senior que dicta la estrategia inmediata.

### 4. Alto Rendimiento con Polars
Diseñado para manejar **100,000+ registros** en RAM de forma instantánea gracias al uso de **Polars**, el motor de procesamiento de datos más rápido de la actualidad.

## ☁️ Activa la IA Cloud (Versión Gratuita)
Para que el Agente de IA analice tus datos en la web, puedes usar una cuenta de **Groq** (totalmente gratuita y ultra rápida):

1.  Consigue tu API Key gratuita en [Groq Console](https://console.groq.com/).
2.  En tu panel de Streamlit Cloud, ve a **Settings** > **Secrets**.
3.  Pega lo siguiente:
    ```toml
    OPENAI_API_KEY = "tu_clave_de_groq_aqui"
    OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
    CLOUD_MODEL_NAME = "llama-3.1-70b-versatile"
    ```

## 🛠️ Instalación Local
```bash
git clone https://github.com/AnGuzC74/Smart-Inventory-Forecaster.git
cd Smart-Inventory-Forecaster
git checkout feat/olist-smart-inventory
pip install -r requirements.txt
python simulador.py
```

---
*Desarrollado para el portafolio profesional de MLOps & AI Engineering.*
