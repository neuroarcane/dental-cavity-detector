"""
Smoke tests for the serving API. These stub the model so the API layer
(routing, validation, JSON shape) is verified without needing GPU or the
model weights. Run: pytest -q
"""
import io
import pytest
from PIL import Image

import app as api


@pytest.fixture
def client(monkeypatch):
    # Replace real inference with a deterministic stub.
    monkeypatch.setattr(api, "run_inference", lambda image, conf=0.25: [
        {"label": "Cavity", "confidence": 0.91,
         "box": {"x1": 10.0, "y1": 20.0, "x2": 60.0, "y2": 80.0}},
    ])
    api.app.config.update(TESTING=True)
    return api.app.test_client()


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_returns_detections(client):
    r = client.post("/predict", data={"image": (io.BytesIO(_png_bytes()), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] == 1
    d = body["predictions"][0]
    assert d["label"] == "Cavity" and "box" in d and "confidence" in d


def test_predict_missing_file(client):
    r = client.post("/predict")
    assert r.status_code == 400
    assert "image" in r.get_json()["error"]


def test_predict_not_an_image(client):
    r = client.post("/predict", data={"image": (io.BytesIO(b"not an image"), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
