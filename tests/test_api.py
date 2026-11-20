import pytest
from starlette.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


VALID_WINE = {
    "wine_id": "WINE_TEST001",
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}

SECOND_WINE = {
    "wine_id": "WINE_TEST002",
    "fixed acidity": 11.2,
    "volatile acidity": 0.28,
    "citric acid": 0.56,
    "residual sugar": 1.9,
    "chlorides": 0.075,
    "free sulfur dioxide": 17.0,
    "total sulfur dioxide": 60.0,
    "density": 0.998,
    "pH": 3.16,
    "sulphates": 0.58,
    "alcohol": 9.8,
}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Wine Quality Prediction API"
    assert data["version"] == "1.0.0"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert data["model_name"] is not None
    assert data["model_uri"] is not None


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_valid_prediction(client):
    response = client.post("/predict", json=VALID_WINE)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "quality_probability" in data
    assert 0.0 <= data["quality_probability"] <= 1.0


def test_valid_prediction_snake_case(client):
    payload = {
        "wine_id": "WINE_SNAKE",
        "fixed_acidity": 8.0,
        "volatile_acidity": 0.5,
        "citric_acid": 0.25,
        "residual_sugar": 2.0,
        "chlorides": 0.08,
        "free_sulfur_dioxide": 15.0,
        "total_sulfur_dioxide": 45.0,
        "density": 0.996,
        "pH": 3.4,
        "sulphates": 0.65,
        "alcohol": 10.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["quality_probability"] <= 1.0


def test_invalid_prediction_negative_alcohol(client):
    payload = VALID_WINE.copy()
    payload["alcohol"] = -5.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert any("alcohol" in d["field"] for d in data["details"])


def test_invalid_prediction_negative_ph(client):
    payload = VALID_WINE.copy()
    payload["pH"] = -1.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert any("pH" in d["field"] for d in data["details"])


def test_missing_required_field(client):
    payload = VALID_WINE.copy()
    del payload["alcohol"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert any("alcohol" in d["field"] for d in data["details"])


def test_invalid_datatype(client):
    payload = VALID_WINE.copy()
    payload["alcohol"] = "high"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"


def test_batch_prediction_wines(client):
    payload = {"wines": [VALID_WINE, SECOND_WINE]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["prediction"] in [0, 1]
        assert 0.0 <= pred["quality_probability"] <= 1.0


def test_batch_prediction_customers_alias(client):
    payload = {"customers": [VALID_WINE, SECOND_WINE]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2


def test_batch_prediction_empty(client):
    payload = {"wines": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_extreme_inputs(client):
    payload = VALID_WINE.copy()
    payload["fixed acidity"] = 50.0
    payload["alcohol"] = 25.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
