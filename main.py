import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat
from predictor import predict

# Configuration des logs
logger = logging.getLogger(__name__)

# Création de l'application
app = FastAPI(
    title="ML Regression API",
    description="API de prédiction utilisant un modèle de régression linéaire",
    version="1.0.0"
)


# Schéma des données d'entrée
class InputData(BaseModel):
    features: list[FiniteFloat] = Field(
        ...,
        min_length=1,
        max_length=1,
        examples=[[1.5]]
    )


@app.get("/")
def root():
    return {"message": "Welcome to ML Regression API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(data: InputData):
    try:
        prediction = predict(data.features)

        return {
            "status": "success",
            "input": data.features,
            "prediction": prediction
        }

    except Exception:
        logger.exception("Erreur lors de la prédiction")

        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction"
        )
