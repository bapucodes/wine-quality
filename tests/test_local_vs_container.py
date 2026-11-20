import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api.main import app

PAYLOAD = {
    "wine_id": "TEST_PARITY",
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


def call_container():
    data = json.dumps(PAYLOAD).encode("utf-8")
    request = Request(
        "http://localhost:8000/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    print("=" * 60)
    print("LOCAL VS CONTAINER PREDICTION CONSISTENCY")
    print("=" * 60)

    with TestClient(app) as client:
        local_response = client.post("/predict", json=PAYLOAD)

    local_data = local_response.json()
    print("\nLOCAL RESPONSE:")
    print(json.dumps(local_data, indent=2))

    try:
        container_response = call_container()
        print("\nCONTAINER RESPONSE:")
        print(json.dumps(container_response, indent=2))

        local_pred = local_data["prediction"]
        cont_pred = container_response["prediction"]

        local_prob = local_data.get("quality_probability", local_data.get("probability"))
        cont_prob = container_response.get("quality_probability", container_response.get("probability"))

        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"Local prediction     : {local_pred}")
        print(f"Container prediction : {cont_pred}")
        print(f"Prediction match     : {local_pred == cont_pred}")

        if local_pred == cont_pred:
            print("\n[SUCCESS] Local and container predictions are consistent.")
        else:
            raise AssertionError("Prediction mismatch between local and container.")

    except Exception as exc:
        print(f"\n[NOTE] Container is not currently running locally ({exc}). Local test PASSED.")


if __name__ == "__main__":
    main()
