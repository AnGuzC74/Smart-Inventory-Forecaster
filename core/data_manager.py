import polars as pl
import os

class DataManager:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data = None

    def procesar_inventario(self):
        # Escaneo dinámico de archivos
        if not os.path.exists(self.data_folder):
             os.makedirs(self.data_folder)
             
        archivos = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        path_v, path_p = None, None

        for f in archivos:
            ruta = os.path.join(self.data_folder, f)
            try:
                headers = pl.read_csv(ruta, n_rows=1).columns
                if "price" in headers: path_v = ruta
                elif "product_category_name" in headers: path_p = ruta
            except:
                continue

        if not path_v or not path_p:
            # Si no hay datos, creamos una simulación básica para que no rompa
            print("⚠️ No se detectaron tablas reales. Generando simulación de 100k registros...")
            return self._generar_datos_simulados()

        # Procesamiento Lazy y Pareto
        self.data = (
            pl.scan_csv(path_v)
            .join(pl.scan_csv(path_p), on="product_id")
            .group_by("product_category_name")
            .agg([
                pl.col("price").mean().alias("precio_medio"),
                pl.len().alias("unidades_vendidas")
            ])
            .sort("unidades_vendidas", descending=True)
            .collect()
        )
        
        return self._aplicar_pareto()

    def _aplicar_pareto(self):
        total = self.data["unidades_vendidas"].sum()
        self.data = self.data.with_columns([
            (pl.col("unidades_vendidas").cum_sum() / total).alias("pct_acum")
        ]).with_columns([
            pl.when(pl.col("pct_acum") <= 0.8).then(pl.lit("A"))
            .otherwise(pl.lit("B")).alias("clase_pareto")
        ])
        return self.data

    def _generar_datos_simulados(self):
        import numpy as np
        n = 100000
        self.data = pl.DataFrame({
            "product_category_name": [f"Cat_{i}" for i in range(n)],
            "precio_medio": np.random.uniform(10, 2000, n),
            "unidades_vendidas": np.random.randint(1, 500, n)
        }).sort("unidades_vendidas", descending=True)
        return self._aplicar_pareto()
