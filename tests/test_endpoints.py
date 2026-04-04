"""Basic endpoint tests for the Python Intelligence API."""


def test_root():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Monreale OS Python Intelligence API"


def test_health():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
