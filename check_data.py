import sys
import os

print(f"🏠 Directorio de trabajo: {os.getcwd()}")
print(f"🐍 Python Path: {sys.path[0]}")

# Verificamos si los archivos existen realmente donde Python espera
core_path = os.path.join(os.getcwd(), 'core', 'data_manager.py')
print(f"🔍 ¿Existe data_manager.py?: {os.path.exists(core_path)}")
