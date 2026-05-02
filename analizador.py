import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class AnalizadorInventario:
    def __init__(self):
        self.model = None
        self.archivo_modelo = "modelo_inventario.joblib"

    def entrenar(self):
        # Simulación de carga masiva (Olist dataset style)
        print("Cargando y procesando 100,000 registros para entrenamiento masivo...")
        n = 100000
        data = {
            'costo': np.random.uniform(10, 2000, n),
            'prioridad_encoded': np.random.randint(1, 4, n),
            'stock_optimo': np.random.randint(1, 50, n)
        }
        df = pd.DataFrame(data)
        X = df[['costo', 'prioridad_encoded']]
        y = df['stock_optimo']
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        joblib.dump(self.model, self.archivo_modelo)

    def entrenar_si_no_existe(self):
        if os.path.exists(self.archivo_modelo):
            self.model = joblib.load(self.archivo_modelo)
        else:
            self.entrenar()

    def predecir_con_intervalo(self, costo, prioridad_texto):
        prioridades = {"Bajo": 1, "Medio": 2, "Alto": 3}
        p_val = prioridades.get(prioridad_texto, 1)
        entrada = np.array([[costo, p_val]])
        prediccion = self.model.predict(entrada)[0]
        return round(prediccion, 2), round(prediccion * 0.9, 2), round(prediccion * 1.1, 2)
