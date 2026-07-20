import os
import sys
import joblib
from rich.console import Console
from rich.panel import Panel
from core.analyzer import AnalizadorInventario

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

console = Console()

def ejecutar_gestion_logistica():
    """
    Ejecuta el pipeline de optimización de inventario estándar.
    """
    console.print(Panel("[bold green]SISTEMA DE GESTIÓN DE SUMINISTROS - OPTIMIZACIÓN[/bold green]", expand=False))

    ruta_data = os.path.join(BASE_DIR, "data")

    try:
        # 1. Instanciamos el analizador estándar
        ia = AnalizadorInventario(ruta_data)

        # 2. Procesamos datos y predecimos
        ia.dm.procesar_inventario()
        precio_test = 250.0
        stock_requerido = ia.predecir_stock(precio_test)

        # 3. Recuperamos métricas
        payload = joblib.load(ia.ruta_modelo)
        r2_info = payload['r2_score']

        # 4. Generamos recomendación operativa con la IA
        with console.status("[bold cyan]Generando recomendación operativa..."):
            # Usamos el método de recomendación que ya tenías en AnalizadorInventario
            analisis_ia = ia.obtener_recomendacion_agente(precio_test, stock_requerido, r2_info)

        # 5. Visualización profesional
        console.print(Panel(
            f"PRODUCTO: [bold]${precio_test}[/bold]\n"
            f"RESERVA DE STOCK: [bold blue]{stock_requerido} UNIDADES[/bold blue]\n"
            f"FIABILIDAD DEL MODELO: {r2_info:.4f}",
            title="MÉTRICAS DE PLANIFICACIÓN",
            border_style="blue"
        ))

        console.print(Panel(
            analisis_ia,
            title="📦 RECOMENDACIÓN DE ABASTECIMIENTO",
            border_style="green",
            padding=(1, 2)
        ))

    except Exception as e:
        console.print(f"[bold red]ERROR DE EJECUCIÓN:[/bold red] {e}")

if __name__ == "__main__":
    ejecutar_gestion_logistica()
