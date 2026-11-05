import joblib
import numpy as np

model = joblib.load('ml_api_demo_main/model.pkl')

X_test = np.array([[1.5], [3]])

predictions = model.predict(X_test)
print(f"Entrée : {X_test} | prédiction : {predictions}")

model_type = type(model).__name__
n_features = getattr(model, 'n_features_in_', None)
print(f"Type du modèle : {model_type}")
print(f"Nombre de features attendues : {n_features}")