import polars as pl
import os
import logging

# Configurar logger para este módulo
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data = None
        self.synthetic_mode = False # Haz visible si estamos usando datos sintéticos de respaldo

    def procesar_inventario(self):
        # Escaneo dinámico de archivos
        if not os.path.exists(self.data_folder):
             os.makedirs(self.data_folder)

        archivos = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        path_v, path_p = None, None

        for f in archivos:
            ruta = os.path.join(self.data_folder, f)
            try:
                # Lectura rápida de headers
                headers = pl.read_csv(ruta, n_rows=1).columns
                if "price" in headers: path_v = ruta
                elif "product_category_name" in headers: path_p = ruta
            except Exception as e:
                logger.error(f"Error al leer cabecera del archivo {f}: {e}", exc_info=True)
                continue

        # Si no hay archivos o falta uno, generamos simulación profesional
        if not path_v or not path_p:
            logger.warning("⚠️ Datos reales no encontrados. Generando simulación masiva (100k registros)...")
            self.synthetic_mode = True
            return self._generar_datos_simulados()

        try:
            # Procesamiento Lazy con limpieza de nulos
            self.data = (
                pl.scan_csv(path_v)
                .join(pl.scan_csv(path_p), on="product_id")
                .drop_nulls()
                .group_by("product_category_name")
                .agg([
                    pl.col("price").mean().alias("precio_medio"),
                    pl.len().alias("unidades_vendidas")
                ])
                .sort("unidades_vendidas", descending=True)
                .collect()
            )

            # Si después del join el resultado es muy pequeño, simulamos para evitar errores de ML
            if len(self.data) < 5:
                logger.warning("Pocos datos resultantes del procesamiento real (< 5). Activando modo simulación.")
                self.synthetic_mode = True
                return self._generar_datos_simulados()

            self.synthetic_mode = False
            return self._aplicar_pareto()
        except Exception as e:
            logger.error(f"❌ Error en procesamiento de datos reales: {e}. Activando modo simulación.", exc_info=True)
            self.synthetic_mode = True
            return self._generar_datos_simulados()

    def _aplicar_pareto(self):
        if self.data is None or len(self.data) == 0:
            return self.data

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
        # Generamos 1000 categorías simuladas (suficiente para demostración masiva)
        n = 1000
        self.data = pl.DataFrame({
            "product_category_name": [f"Categoria_{i}" for i in range(n)],
            "precio_medio": np.random.uniform(20, 1500, n),
            "unidades_vendidas": np.random.randint(5, 1000, n)
        }).sort("unidades_vendidas", descending=True)
        return self._aplicar_pareto()
