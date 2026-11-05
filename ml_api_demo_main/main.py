from fastapi import FastAPI
import platform
import socket
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Simple training data: y = 2 * x
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction endpoint
@app.get("/predict/{val}")
def predict(val: int):
    pred = model.predict([[val]])
    return {
        "input": val,
        "prediction": float(pred[0])
    }

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Hello from my Cloud Lab 🚀",
        "python_version": platform.python_version(),
        "server_hostname": socket.gethostname()
    }

# Square endpoint
@app.get("/square/{number}")
def square(number: int):
    return {
        "number": number,
        "square": number * number
    }
    

# Save the model
joblib.dump(model, "model.pkl") 
print("Model saved as model.pkl")
