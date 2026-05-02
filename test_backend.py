import os
import sys
import time
import traceback

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from core.analyzer import AnalizadorInventario
    print("✅ Módulos cargados correctamente.")
except ImportError as e:
    print(f"❌ Error al importar core: {e}")
    sys.exit()

def diagnostico_proceso():
    print("-" * 40)
    print("🚀 INICIANDO DIAGNÓSTICO DE CARGA DE DATOS")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        # 1. Inicialización
        ruta_data = os.path.join(BASE_DIR, "data")
        analizador = AnalizadorInventario(ruta_data)
        print(f"📂 Carpeta de datos: {ruta_data}")

        # 2. Procesamiento de 100k datos
        print("\n⚙️ Procesando 100,000 datos (Pareto + K-Means)...")
        p_start = time.time()
        analizador.dm.procesar_inventario()
        p_end = time.time()
        print(f"⏱️ Tiempo de procesamiento en terminal: {p_end - p_start:.4f} segundos")

        # 3. Validación de resultados
        if analizador.dm.data is not None:
            df = analizador.dm.data
            print(f"📊 Registros procesados: {len(df)}")
            print(f"🔍 Columnas generadas: {df.columns}")
            
            # Verificación de Pareto
            if "clase_pareto" in df.columns:
                print("✅ Columna 'clase_pareto' detectada.")
                print(df.group_by("clase_pareto").count())
            
            # Verificación de Clusters
            if "cluster_km" in df.columns:
                print("✅ Columna 'cluster_km' detectada.")
                print(df.group_by("cluster_km").count())
        else:
            print("❌ El DataFrame está vacío.")

    except Exception as e:
        print("🔥 ERROR DETECTADO:")
        traceback.print_exc()

    print("-" * 40)
    print(f"🏁 Diagnóstico finalizado en {time.time() - start_time:.2f} segundos.")

if __name__ == "__main__":
    diagnostico_proceso()
