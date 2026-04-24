# Smart-Inventory-Forecaster 📦🚀

Este proyecto implementa un pipeline industrial de Machine Learning para la predicción de inventario inteligente, integrando un backend robusto con FastAPI y una interfaz interactiva con Streamlit.

## 🛠️ Tecnologías Utilizadas
- **Python 3.9+**
- **FastAPI**: Backend para servir el modelo.
- **Streamlit**: Dashboard interactivo para el usuario final.
- **Scikit-Learn**: Modelo de Random Forest para predicción de demanda.
- **Docker**: Containerización para despliegue consistente.

## 📂 Estructura del Proyecto
```text
Smart-Inventory-Forecaster/
├── assets/                 # Capturas de pantalla y multimedia
├── analizador.py           # Lógica del modelo ML y procesamiento
├── app.py                  # API REST con FastAPI
├── interfaz.py             # Dashboard en Streamlit
├── verificador_ops.py      # Auditoría de MLOps y entrenamiento
├── requirements.txt        # Dependencias del proyecto
└── Dockerfile              # Configuración de Docker
```

## 🚀 Cómo Ejecutar

### Con Docker (Recomendado)
1. Construir la imagen:
   ```bash
   docker build -t inventory-forecaster .
   ```
2. Ejecutar el contenedor:
   ```bash
   docker run -p 8501:8501 inventory-forecaster
   ```

### Localmente
1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar el verificador de MLOps:
   ```bash
   python verificador_ops.py
   ```
3. Iniciar la interfaz:
   ```bash
   streamlit run interfaz.py
   ```

## 📸 Galería de la Interfaz

A continuación se muestran capturas del sistema en funcionamiento:

| Descripción | Imagen |
|-------------|--------|
| **Dashboard Principal** | ![Dashboard](./assets/Imagen_1.png) |
| **Análisis de Datos** | ![Análisis](./assets/Imagen_2.png) |
| **Configuración de Parámetros** | ![Configuración](./assets/Imagen_3.png) |
| **Resultados de Predicción** | ![Predicción](./assets/Imagen_4.png) |
| **Intervalos de Confianza** | ![Estadísticas](./assets/Imagen_5.png) |

---
Desarrollado para el portafolio de **Laboratorio DS**.
