# 📦 Olist Smart Inventory & AI Forecaster 🚀

Esta aplicación es una solución avanzada de **MLOps** para la gestión de inventarios. A diferencia de versiones anteriores, utiliza una arquitectura distribuida para garantizar estabilidad y potencia de cálculo.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-inventory-forecaster-i3uaskvwqssdukaccbwmbt.streamlit.app/)

## 🚀 Demo en Vivo
Puedes acceder a la simulación interactiva aquí: [Olist Smart Forecaster App](https://smart-inventory-forecaster-i3uaskvwqssdukaccbwmbt.streamlit.app/)

## 🌟 Mejoras Clave en esta Versión
- **Segmentación Dual**: Validación cruzada entre reglas de negocio (**Pareto ABC**) y aprendizaje automático (**K-Means Clustering**).
- **Cerebro Desacoplado**: Backend en FastAPI que mantiene 100,000 registros en RAM para respuestas instantáneas.
- **Inteligencia Generativa**: Integración con **Llama 3.2** para convertir datos fríos en directivas logísticas humanas.
- **Arquitectura de Microservicios**: Separación de procesos para evitar bloqueos en la interfaz de usuario.

## 🚀 Instrucciones de Inicio Rápido
Para ver la aplicación en funcionamiento (simulación completa), ejecute:
```bash
python simulador.py
```

## ☁️ Configuración de IA Cloud (Opcional)
Para habilitar los **IA Insights** en Streamlit Cloud:
1. Ve a **Settings** > **Secrets** en tu panel de Streamlit.
2. Agrega tu clave (OpenAI o compatible):
   ```toml
   OPENAI_API_KEY = "tu_clave_aqui"
   OPENAI_BASE_URL = "https://api.openai.com/v1" # O la de Groq/Perplexity
   CLOUD_MODEL_NAME = "gpt-3.5-turbo"
   ```
Si no se configuran secretos, la aplicación usará un mensaje técnico estándar sin fallar.

---
*Rama: feat/olist-smart-inventory - Fase Avanzada (Cloud Ready)*
