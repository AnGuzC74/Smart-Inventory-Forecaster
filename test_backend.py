import os
import sys
import pytest

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.analyzer import AnalizadorInventario  # noqa: E402 (requiere sys.path.insert previo)
from core.data_manager import DataManager  # noqa: E402 (requiere sys.path.insert previo)

def test_kmeans_and_pareto_flow():
    """Prueba que el procesamiento de datos y la segmentación K-Means funcionan de verdad y generan las columnas correspondientes."""
    ruta_data = os.path.join(BASE_DIR, "data")
    analizador = AnalizadorInventario(ruta_data)

    # Procesar inventario
    df_procesado = analizador.dm.procesar_inventario()
    assert df_procesado is not None, "El procesamiento de inventario devolvió None"

    # Aplicar K-Means segmentación
    df_segmentado = analizador.segment_kmeans(df_procesado)
    assert df_segmentado is not None, "La segmentación K-Means devolvió None"

    # Validaciones y aserciones explícitas
    assert "clase_pareto" in df_segmentado.columns, "La columna 'clase_pareto' no se encuentra en el DataFrame"
    assert "cluster_km" in df_segmentado.columns, "La columna 'cluster_km' no se encuentra en el DataFrame"

    # Verificar que hay datos
    assert len(df_segmentado) > 0, "El DataFrame resultante está vacío"

    # Verificar Pareto (A, B)
    valores_pareto = df_segmentado["clase_pareto"].unique().to_list()
    assert "A" in valores_pareto or "B" in valores_pareto, "La columna clase_pareto no contiene clases A o B válidas"

def test_synthetic_fallback_on_forced_failure():
    """Prueba que el aviso de 'modo simulado' (synthetic_mode) se activa correctamente ante una carga de datos fallida forzada."""
    # Carpeta que no existe o no tiene datos correctos
    ruta_falsa = os.path.join(BASE_DIR, "non_existent_folder_xyz")

    dm = DataManager(ruta_falsa)
    df_resultado = dm.procesar_inventario()

    # Asegurar que se activó el modo sintético
    assert dm.synthetic_mode is True, "El DataManager debería haber activado synthetic_mode ante una ruta inexistente"
    assert df_resultado is not None, "A pesar de fallar la carga real, debería retornar el DataFrame simulado"
    assert len(df_resultado) == 1000, "El DataFrame simulado de respaldo debería contener 1000 filas por defecto"

def test_prediction_with_interval():
    """Prueba que la predicción con intervalo de confianza basada en varianza de árboles funciona correctamente."""
    ruta_data = os.path.join(BASE_DIR, "data")
    analizador = AnalizadorInventario(ruta_data)

    # Forzar entrenamiento para asegurar que el modelo existe
    analizador.entrenar_y_salvar()

    # Obtener predicciones
    precio_test = 150.0
    promedio, limite_inf, limite_sup = analizador.predecir_con_intervalo(precio_test)

    assert isinstance(promedio, float), "La predicción promedio debería ser float"
    assert isinstance(limite_inf, float), "El límite inferior debería ser float"
    assert isinstance(limite_sup, float), "El límite superior debería ser float"

    # El límite inferior debe ser no negativo
    assert limite_inf >= 0.0, "El límite inferior de la demanda no puede ser negativo"
    # El promedio debe estar dentro del intervalo
    assert limite_inf <= promedio <= limite_sup, "El promedio de predicción debe estar dentro del intervalo de confianza [inf, sup]"

if __name__ == "__main__":
    print("🚀 Ejecutando pruebas unitarias directamente...")
    pytest.main([__file__, "-v"])
