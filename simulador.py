import subprocess
import time
import sys
import os

def ejecutar_aplicacion():
    # Detectar la ruta del proyecto
    ruta_base = os.path.dirname(os.path.abspath(__file__))

    print("🛰️ Iniciando sistema Olist Pro...")

    # 1. Iniciar el Backend (El Cerebro)
    print("🧠 Paso 1: Cargando datos y modelos en FastAPI...")
    api_proc = subprocess.Popen([sys.executable, "api.py"], cwd=ruta_base)

    # Tiempo de gracia para que la RAM se llene con los 100k registros
    time.sleep(7)

    # 2. Iniciar el Frontend (La Interfaz)
    print("📊 Paso 2: Lanzando Dashboard en Streamlit...")
    ui_proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=ruta_base)

    print("\n✅ Todo listo. La simulación está corriendo.")
    print("Presiona CTRL+C en esta terminal para apagar todos los servicios.")

    try:
        # Mantener el script activo mientras ambos procesos funcionen
        api_proc.wait()
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Apagando servicios de forma segura...")
        api_proc.terminate()
        ui_proc.terminate()

if __name__ == "__main__":
    ejecutar_aplicacion()
