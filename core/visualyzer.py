import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import os

def generar_evidencia_visual(df: pl.DataFrame):
    # Convertimos a Pandas para visualización
    pdf = df.to_pandas()
    os.makedirs("exports", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Pareto (La concentración de ventas)
    sns.lineplot(ax=axes[0], data=pdf, x=range(len(pdf)), y="pct_acum", color="royalblue")
    axes[0].axhline(y=0.8, color='red', linestyle='--', label="Umbral 80%")
    axes[0].set_title("Análisis de Pareto: Concentración de Ventas")
    axes[0].legend()

    # Gráfico 2: K-Means (Estructura de comportamiento)
    if "cluster_km" in pdf.columns:
        sns.scatterplot(ax=axes[1], data=pdf, x="precio_medio", y="unidades_vendidas",
                        hue="cluster_km", palette="viridis", alpha=0.6)
        axes[1].set_yscale('log')
        axes[1].set_title("Clustering K-Means: Precio vs Ventas")
    else:
        axes[1].text(0.5, 0.5, "Clustering no disponible", ha='center')

    plt.tight_layout()
    plt.savefig("exports/analisis_masivo.png")
    return "✅ Gráficos guardados en 'exports/analisis_masivo.png'"
