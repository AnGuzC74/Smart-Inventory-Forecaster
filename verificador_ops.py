import os
import sys
import json
import logging
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.analyzer import AnalizadorInventario  # noqa: E402 (requiere sys.path.insert previo)

def auditoria_sistema():
    logger.info("🔍 Iniciando protocolo de inspección MLOps...")

    # 1. Instanciar analizador
    ruta_data = os.path.join(BASE_DIR, "data")
    analizador = AnalizadorInventario(ruta_data)

    # 2. Verificar existencia del modelo
    if not os.path.exists(analizador.ruta_modelo) or not os.path.exists(analizador.ruta_baseline):
        logger.warning("⚠️ Crítico: Modelo o línea base no encontrados. Disparando entrenamiento...")
        analizador.entrenar_y_salvar()
        logger.info("✅ Modelo y línea base generados correctamente.")
    else:
        logger.info("✅ Archivo de modelo (RF) y línea base (baseline_stats.json) detectados.")

    # 3. Procesar datos del lote/batch actual
    logger.info("🔄 Procesando lote de datos actual...")
    analizador.dm.procesar_inventario()

    if analizador.dm.data is None or len(analizador.dm.data) == 0:
        logger.error("❌ Error de integridad: No se pudieron procesar datos para la auditoría.")
        sys.exit(1)

    # 4. Verificación de Data Drift real
    logger.info("🔬 Realizando verificación de Data Drift...")
    try:
        with open(analizador.ruta_baseline, "r") as f:
            baseline = json.load(f)

        baseline_mean = baseline["mean"]
        baseline_std = baseline["std"]

        # Calcular estadísticas del lote actual para 'precio_medio' (Costo/Precio del producto)
        precios_actuales = analizador.dm.data["precio_medio"].to_numpy()
        current_mean = float(np.mean(precios_actuales))
        current_std = float(np.std(precios_actuales))

        logger.info(f"📊 Estadísticas Baseline (Entrenamiento): Media={baseline_mean:.4f}, Std={baseline_std:.4f}")
        logger.info(f"📊 Estadísticas Batch Actual: Media={current_mean:.4f}, Std={current_std:.4f}")

        if baseline_std > 1e-9:
            z_score = abs(current_mean - baseline_mean) / baseline_std
        else:
            z_score = 0.0

        logger.info(f"📈 Z-Score calculado para Drift: {z_score:.4f}")

        threshold = 3.0
        if z_score > threshold:
            logger.warning(f"🚨 ¡ALERTA DE DATA DRIFT DETECTADO! Z-Score ({z_score:.4f}) supera el umbral de {threshold}.")
            print(f"❌ Inspección de integridad de datos completada: ¡Drift Detectado! (Z-Score: {z_score:.4f} > {threshold})")
            drift_detected = True
        else:
            logger.info("✅ Sin drift significativo detectado.")
            print(f"✅ Inspección de integridad de datos completada: No se detectó Data Drift (Z-Score: {z_score:.4f} <= {threshold}).")
            drift_detected = False

    except Exception as e:
        logger.error(f"❌ Error durante el cálculo de Data Drift: {e}", exc_info=True)
        print("❌ Error en protocolo de inspección de integridad de datos.")
        sys.exit(1)

    # 5. Diagnóstico final del sistema
    if not drift_detected:
        print("✅ Protocolo de autocuración finalizado. Sistema estable y sin drift.")
    else:
        print("⚠️ Protocolo finalizado con advertencias: Se detectó Data Drift en el lote actual.")

if __name__ == "__main__":
    auditoria_sistema()
