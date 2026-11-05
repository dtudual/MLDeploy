
from pathlib import Path
import joblib
import numpy as np

# Chemin vers le modèle
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml_api_demo_main" / "model.pkl"

# Vérification de l'existence du modèle
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Le modèle est introuvable : {MODEL_PATH}"
    )

# Chargement du modèle une seule fois
model = joblib.load(MODEL_PATH)


def predict(features: list[float]) -> float:
    """
    Effectue une prédiction avec le modèle préentraîné.

    Args:
        features: Liste contenant les valeurs des features.

    Returns:
        Prédiction du modèle sous forme de float.
    """

    # Conversion au format attendu par scikit-learn
    X_input = np.array([features], dtype=float)

    # Prédiction
    prediction = model.predict(X_input)

    return float(prediction[0])
