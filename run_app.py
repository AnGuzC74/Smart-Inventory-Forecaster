import subprocess
import time
import sys
import os

def lanzar_proyecto():
    # Obtener la ruta base
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("🚀 Iniciando Orquestador de MLOps...")

    # 1. Lanzar el Backend (API)
    print("📡 Encendiendo el Cerebro (API)...")
    api_proc = subprocess.Popen([sys.executable, "api.py"], cwd=base_dir)

    # Esperar unos segundos para que la RAM cargue los datos
    time.sleep(5)

    # 2. Lanzar el Frontend (Streamlit)
    print("🎨 Encendiendo la Interfaz (Streamlit)...")
    # Usamos 'python -m streamlit' para asegurar que use el entorno virtual actual
    ui_proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=base_dir)

    try:
        # Mantener el script vivo mientras los procesos corren
        api_proc.wait()
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo servicios...")
        api_proc.terminate()
        ui_proc.terminate()

if __name__ == "__main__":
    lanzar_proyecto()
