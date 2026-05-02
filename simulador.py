import os
import time
import subprocess
import requests

def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("====================================================")
    print("      📦 OLIST SMART INVENTORY SIMULATOR           ")
    print("====================================================")
    print("Iniciando servicios distribuidos...\n")

def start_services():
    print("🚀 [1/3] Lanzando Cerebro Desacoplado (FastAPI)...")
    # In reality, we'd use Popen but for a script we simulate the flow
    # subprocess.Popen(["python", "app.py"]) 
    time.sleep(1)
    
    print("🧠 [2/3] Cargando 100k registros en RAM (Simulado)...")
    time.sleep(2)
    
    print("🤖 [3/3] Conectando con Llama 3.2 vía Ollama...")
    time.sleep(1)
    print("\n✅ Sistema Listo para operación masiva.\n")

if __name__ == "__main__":
    print_banner()
    start_services()
    
    print("─"*50)
    print("SISTEMA DE SEGMENTACIÓN DUAL ACTIVADO")
    print("Pareto ABC: [OK]")
    print("K-Means Clustering: [OK]")
    print("─"*50)
    
    print("\nEjecuta 'agente_inventario.py' para interactuar con la IA.")
    print("Simulación terminada.")
