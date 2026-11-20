import time
import statistics
from starlette.testclient import TestClient

from api.main import app


def create_client():
    return TestClient(app)


VALID_WINE = {
    "wine_id": "PERF001",
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


def test_single_request_latency():
    with create_client() as client:
        start_time = time.perf_counter()
        response = client.post("/predict", json=VALID_WINE)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        assert response.status_code == 200
        print(f"\nSingle request latency: {latency_ms:.2f} ms")


def test_repeated_request_latency():
    repetitions = 20
    latencies = []

    with create_client() as client:
        for _ in range(repetitions):
            start_time = time.perf_counter()
            response = client.post("/predict", json=VALID_WINE)
            end_time = time.perf_counter()

            assert response.status_code == 200
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

    average_latency = statistics.mean(latencies)
    minimum_latency = min(latencies)
    maximum_latency = max(latencies)
    median_latency = statistics.median(latencies)

    print("\nRepeated request performance:")
    print(f"Requests: {repetitions}")
    print(f"Average latency: {average_latency:.2f} ms")
    print(f"Median latency: {median_latency:.2f} ms")
    print(f"Minimum latency: {minimum_latency:.2f} ms")
    print(f"Maximum latency: {maximum_latency:.2f} ms")

    assert len(latencies) == repetitions


def test_prediction_consistency():
    repetitions = 10
    predictions = []
    probabilities = []

    with create_client() as client:
        for _ in range(repetitions):
            response = client.post("/predict", json=VALID_WINE)
            assert response.status_code == 200
            data = response.json()
            predictions.append(data["prediction"])
            probabilities.append(data["quality_probability"])

    assert len(set(predictions)) == 1
    assert len(set(probabilities)) == 1

    print("\nPrediction consistency:")
    print(f"Prediction: {predictions[0]}")
    print(f"Probability: {probabilities[0]:.6f}")


def test_api_reliability():
    repetitions = 30
    successful_requests = 0

    with create_client() as client:
        for _ in range(repetitions):
            response = client.post("/predict", json=VALID_WINE)
            if response.status_code == 200:
                successful_requests += 1

    success_rate = (successful_requests / repetitions) * 100
    print("\nAPI reliability:")
    print(f"Total requests: {repetitions}")
    print(f"Successful requests: {successful_requests}")
    print(f"Success rate: {success_rate:.2f}%")

    assert successful_requests == repetitions


def test_batch_prediction_performance():
    wines = []
    for i in range(20):
        wine = VALID_WINE.copy()
        wine["wine_id"] = f"PERF{i:03d}"
        wines.append(wine)

    payload = {"wines": wines}

    with create_client() as client:
        start_time = time.perf_counter()
        response = client.post("/predict/batch", json=payload)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 20
        print(f"\nBatch (20 items) latency: {latency_ms:.2f} ms")
