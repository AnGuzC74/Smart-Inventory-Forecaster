from pydantic import BaseModel, Field, validator

class PrediccionRequest(BaseModel):
    costo: float = Field(..., gt=0)
    prioridad: str = Field(...)

    @validator('prioridad')
    def validar_prioridad(cls, v):
        valor = v.strip().lower()
        if valor in ['bajo', 'baja']:
            return 'Bajo'
        if valor in ['medio', 'media']:
            return 'Medio'
        if valor in ['alto', 'alta']:
            return 'Alto'
        raise ValueError("Prioridad debe ser: Bajo, Medio o Alto")

class PrediccionResponse(BaseModel):
    prediccion: float
    limite_inferior: float
    limite_superior: float
    mensaje: str
