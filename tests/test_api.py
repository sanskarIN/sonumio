from fastapi.testclient import TestClient
import pytest
from sonumio import api

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    return TestClient(api.app)

def test_health_and_models(client):
    assert client.get("/health").json()["original_sonumio_models_available"] == 0
    models = client.get("/v1/models").json()
    assert {m["id"] for m in models} == {"procedural-sfx-v0", "meta-musicgen-small"}
    assert all(not model["original_sonumio_model"] for model in models)

def test_generate_and_download(client):
    response = client.post("/v1/generations", json={
        "prompt": "Wind and ocean waves", "duration_seconds": 0.5, "seed": 21,
    })
    assert response.status_code == 201
    item = response.json()
    assert item["sample_rate"] == 24000
    assert "not a trained AI" in item["disclaimer"]
    file_response = client.get(item["audio_url"])
    assert file_response.status_code == 200
    assert file_response.content[:4] == b"RIFF"
    assert file_response.headers["content-type"] == "audio/wav"

def test_reject_bad_prompts_and_missing_files(client):
    bad = client.post("/v1/generations", json={"prompt": "unknown sound class"})
    assert bad.status_code == 422
    bad = client.post("/v1/generations", json={"prompt": "rain", "duration_seconds": 100})
    assert bad.status_code == 422
    assert client.get("/v1/generations/../../etc/passwd/audio").status_code != 200
    assert client.get("/v1/generations/not-a-uuid/audio").status_code == 404
