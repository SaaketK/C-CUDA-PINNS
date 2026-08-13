"""End-to-end contracts for the FastAPI/Gradio application."""

from fastapi.testclient import TestClient

from app import app


def test_health_and_model_info():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "backend": "native-c-modal",
            "model_loaded": True,
        }

        info = client.get("/api/v1/model/info")
        assert info.status_code == 200
        assert info.json()["metadata"]["equation"] == "heat1d-modal"


def test_compare_contract_and_validation():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict/compare",
            json={"a1": 0.7, "a2": -0.4, "alpha": 0.1},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["pinn"]) == 100
        assert len(body["pinn"][0]) == 100
        assert body["relative_l2_error"] < 0.01

        invalid = client.post(
            "/api/v1/predict/pinn",
            json={"a1": 2.0, "a2": 0.0, "alpha": 0.1},
        )
        assert invalid.status_code == 422


def test_gradio_is_mounted_at_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "C/CUDA PINN Heat 1D" in response.text
