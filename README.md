> ⚠️ **Esta rama fue consolidada en `feat/consolidacion-final-5943981312164777010`. Para la versión activa y recomendada, ve a esa rama.**
# Smart Inventory Management System 📦🤖 (Rama: feat/inventario-masivo-rich)

Este repositorio contiene un sistema híbrido de inteligencia artificial para la gestión y predicción de inventarios industriales. Combina modelos de Machine Learning tradicionales con las capacidades de razonamiento de Grandes Modelos de Lenguaje (LLM).

> **ESTADO DE LA RAMA:** 🛠️ **Desarrollo Activo.** Esta rama está destinada a la implementación de mejoras para el procesamiento de Big Data e inventarios masivos.

## 🚀 Componentes del Sistema

1.  **Motor de Predicción (Random Forest)**: Localizado en `analizador.py`. Utiliza un algoritmo de regresión para estimar el stock óptimo basándose en el costo y la prioridad del repuesto.
2.  **API REST (FastAPI)**: Servida a través de `app.py`. Proporciona endpoints robustos para realizar predicciones en tiempo real.
3.  **Agente de Inventario (LLM)**: Implementado en `agente_inventario.py`. Utiliza Ollama (Llama 3.2) para extraer datos de lenguaje natural y proporcionar justificaciones técnicas a las decisiones de stock.
4.  **Validación de Datos (Pydantic)**: Definida en `esquemas.py` para asegurar la integridad de las entradas y salidas del sistema.

## 🛠️ Instalación y Uso

### Requisitos Previos
- Python 3.9+
- Ollama instalado y con el modelo `llama3.2` descargado.

### Configuración
1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Iniciar el servidor API:
   ```bash
   python app.py
   ```
3. Ejecutar el Agente en otra terminal:
   ```bash
   python agente_inventario.py
   ```

## 📈 Próximos Pasos
- Migración a procesamiento de Big Data.
- Mejora de la interfaz con reportes visuales masivos.
- Optimización de los prompts del agente.

---
*Versión Estable 1.0 - Core Funcional (API + LLM + RF)*  
*Nota: Actualmente trabajando en mejoras masivas en esta rama.*


## 📄 Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
