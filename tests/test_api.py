from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid():
    response = client.post("/predict", json={"features": [1.5]})
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["input"] == [1.5]
    assert data["prediction"] == pytest.approx(3.0)

def test_predict_invalid_type():
    response = client.post("/predict", json={"features": ["abc"]})

    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_invalid_length():
    response = client.post("/predict", json={"features": [1.5, 2.0]})

    assert response.status_code == 422
    assert "detail" in response.json()