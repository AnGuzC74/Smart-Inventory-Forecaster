import polars as pl
import os
import time

def verificar_datos_reales():
    ruta = "data/olist_inventory.csv"

    if not os.path.exists(ruta):
        print(f"❌ No se encuentra el archivo en {ruta}. Revisa el paso 2.")
        return

    print(f"📂 Leyendo: {ruta}...")

    start = time.time()
    # Leemos solo las primeras 5 filas para una vista rápida
    df = pl.read_csv(ruta)
    end = time.time()

    print("\n--- INFO DEL DATASET ---")
    print(f"✅ Filas detectadas: {len(df)}")
    print(f"✅ Columnas: {df.columns}")
    print(f"⏱️ Tiempo de carga: {end - start:.4f} segundos")
    print("\n--- MUESTRA DE DATOS ---")
    print(df.head(5))

if __name__ == "__main__":
    verificar_datos_reales()
