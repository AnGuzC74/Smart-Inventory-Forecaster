import subprocess
import time
import socket
import sys
import os

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def puerto_abierto(puerto):
    """Revisa si el puerto ya está escuchando."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', puerto)) == 0

def iniciar_sistema():
    print("🚀 Iniciando el ecosistema de la aplicación...")

    # 1. Iniciar la API (FastAPI)
    if not puerto_abierto(8001):
        print("🌐 Iniciando servidor API en el puerto 8001...")
        # Usamos el intérprete de Python actual para ejecutar api.py
        api_path = os.path.join(BASE_DIR, "api.py")
        subprocess.Popen([sys.executable, api_path], cwd=BASE_DIR)

        # Esperar hasta que la API responda
        while not puerto_abierto(8001):
            print("⏳ Esperando a que la API responda...")
            time.sleep(1)
        print("✅ API lista y conectada.")
    else:
        print("✅ La API ya estaba en ejecución.")

    # 2. Iniciar Streamlit
    print("📊 Iniciando interfaz de usuario...")
    app_path = os.path.join(BASE_DIR, "app.py")
    # Ejecutamos el comando 'streamlit run app.py'
    subprocess.Popen(["streamlit", "run", "app.py"], cwd=BASE_DIR)

    print("\n🔥 Todo listo. Revisa tu navegador.")

if __name__ == "__main__":
    iniciar_sistema()
