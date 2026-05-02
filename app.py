from fastapi import FastAPI
import uvicorn
from analizador import AnalizadorInventario
from esquemas import PrediccionRequest, PrediccionResponse

app = FastAPI()
motor = AnalizadorInventario()
motor.entrenar_si_no_existe()

@app.post("/predict", response_model=PrediccionResponse)
async def predict(data: PrediccionRequest):
    pred, inf, sup = motor.predecir_con_intervalo(data.costo, data.prioridad)
    return {
        "prediccion": pred,
        "limite_inferior": inf,
        "limite_superior": sup,
        "mensaje": "Predicción procesada correctamente"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
