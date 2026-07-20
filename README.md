# 📦 Smart Inventory Forecaster — Olist AI Edition 🚀

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-inventory-forecaster-9xchjwqfxume79ivuorpna.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![MLOps](https://img.shields.io/badge/MLOps-Drift%20Detection-orange.svg)](verificador_ops.py)
[![ML Engine](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20K--Means-success.svg)](core/analyzer.py)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](test_backend.py)
[![Lint](https://img.shields.io/badge/Lint-Ruff%20Clean-purple.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Sistema de pronóstico y segmentación de inventario con IA, construido sobre datos reales del e-commerce Olist.** Combina Machine Learning tradicional, estadística rigurosa y prácticas de MLOps de nivel productivo para responder a una pregunta simple con una respuesta seria: *¿cuánto stock necesito, y qué tan seguro puedo estar de ese número?*

---

## 🕹️ Pruébalo en vivo — instrucciones de juego

**👉 [Abrir la demo interactiva](https://smart-inventory-forecaster-9xchjwqfxume79ivuorpna.streamlit.app/)** — no necesitas instalar nada.

Una vez adentro, así se juega:

1. **💰 Ingresa un precio de producto** en el campo numérico — cualquier valor que quieras probar, desde repuestos económicos hasta productos premium.
2. **🎯 Presiona "Ejecutar Predicción"** — el sistema te devuelve la cantidad óptima de stock sugerida, junto con un **intervalo de confianza real** (no un margen inventado — calculado a partir de la varianza entre los árboles del bosque aleatorio).
3. **📊 Presiona "Ver Comparativa Pareto vs K-Means"** — aquí está lo interesante: verás el inventario segmentado de dos formas distintas y comparadas lado a lado. Pareto ABC te dice qué productos importan más *para el negocio* (regla 80/20 sobre ventas); K-Means te dice cómo se agrupan los productos *según los datos*, sin que nadie le diga qué buscar. A veces coinciden. A veces no — y ahí está la parte interesante para descubrir.
4. **⚠️ Si en algún momento ves un aviso amarillo de "datos simulados"**, no es un error — es el sistema siendo honesto contigo: significa que no pudo cargar el dataset real en ese momento y está operando en modo de respaldo. Ningún resultado se presenta como real si no lo es.

---

## 🌟 Qué hay debajo del capó

| Capacidad | Cómo está construida |
|---|---|
| **Dataset real de e-commerce** | Datos reales de Olist (Brasil), procesados con **Polars** para velocidad sobre volúmenes grandes |
| **Segmentación dual** | Pareto ABC (regla de negocio 80/20) + K-Means (clustering no supervisado, recalculado dinámicamente) |
| **Intervalo de confianza estadísticamente correcto** | Varianza entre los estimadores del Random Forest (`promedio ± 2×std`), no un margen fijo arbitrario |
| **Detección de drift real** | `verificador_ops.py` compara cada lote contra una línea base guardada mediante Z-score; si supera el umbral, se reporta una alerta real — nunca un falso "✅ completado" |
| **Transparencia de datos simulados** | Aviso visible y persistente en la interfaz si el sistema opera con datos de respaldo |
| **Manejo de errores robusto** | Excepciones explícitas con `logging` estándar de Python, sin bloques `except` silenciosos |

## 📁 Estructura del proyecto

```text
.
├── core/
│   ├── __init__.py
│   ├── analyzer.py         # ML: predicción con intervalo por árboles + K-Means
│   ├── data_manager.py     # Carga de Olist con fallback sintético visible + Pareto
│   └── visualyzer.py       # Gráficos y análisis visual
├── data/                   # Dataset de Olist
├── models/
│   ├── modelo_inventario_rf.joblib
│   └── baseline_stats.json # Línea base para detección de drift
├── app.py                  # Dashboard Streamlit (interfaz en vivo)
├── api.py                  # API FastAPI para predicción de stock
├── test_backend.py         # Suite de pruebas (Pareto, K-Means, fallback, intervalo)
├── verificador_ops.py      # Auditoría MLOps con detección de drift real
├── requirements.txt
└── runtime.txt              # Python 3.11
```

## 🛠️ Instalación y uso local

```bash
# 1. Crea un entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Instala dependencias
pip install -r requirements.txt

# 3. Corre la suite de pruebas
pytest test_backend.py -v

# 4. Corre la verificación de MLOps y drift
python verificador_ops.py

# 5. Lanza el dashboard
streamlit run app.py
```

## ✅ Calidad de código

```bash
ruff check .        # 0 errores
pytest test_backend.py -v   # 3/3 passing
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT — libre para usar, modificar y distribuir. Consulta [LICENSE](LICENSE) para el texto completo.

---

<div align="center">
<sub>Construido explorando el límite entre Machine Learning aplicado y buenas prácticas de MLOps.</sub>
</div>
