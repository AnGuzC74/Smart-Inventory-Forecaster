import re
import polars as pl

# SNIPPET 1: Limpieza de moneda con REGEX
def clean_currency(text: str) -> float:
    """Extrae números de un string sucio (ej: '$1.200,50 USD')"""
    if text is None:
        return 0.0
    # Eliminamos todo lo que no sea número o punto decimal
    clean_val = re.sub(r'[^\d.]', '', str(text))
    return float(clean_val) if clean_val else 0.0

# SNIPPET 2: Reporte de Calidad de Datos
def data_quality_report(df: pl.DataFrame):
    """Muestra un resumen de salud de los datos en la terminal"""
    print("\n" + "═"*40)
    print("📋 REPORTE DE CALIDAD DE DATOS")
    print("═"*40)
    for col in df.columns:
        nulls = df[col].null_count()
        pct = (nulls / len(df)) * 100
        print(f"🔹 {col:.<20} | Nulos: {nulls} ({pct:.1f}%) | Tipo: {df[col].dtype}")
    print("═"*40 + "\n")
