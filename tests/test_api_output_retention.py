"""HTTP-level tests for Sonumio's bounded local audio output store."""
import threading
import time

from fastapi.testclient import TestClient
import pytest

from sonumio import api
from sonumio.output_store import LocalAudioStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    with TestClient(api.app) as instance:
        yield instance


def test_sync_audio_is_in_managed_subdirectory_not_cli_folder(client, tmp_path):
    manually_saved = tmp_path / "my-song.wav"
    manually_saved.write_bytes(b"CLI output should not be touched")
    response = client.post("/v1/generations", json={
        "prompt": "rain", "duration_seconds": 0.5, "seed": 12,
    })
    assert response.status_code == 201
    item = response.json()
    store = api._store_for(tmp_path)
    assert store.root == tmp_path / ".sonumio-api-audio"
    assert (store.root / (item["id"] + ".wav")).is_file()
    assert manually_saved.read_bytes() == b"CLI output should not be touched"
    download = client.get(item["audio_url"])
    assert download.status_code == 200
    assert download.content.startswith(b"RIFF")
    assert download.headers["cache-control"] == "no-store"
    assert store.delete(item["id"])
    assert client.get(item["audio_url"]).status_code == 404
    assert manually_saved.exists()


def test_sync_audio_expires_on_download_without_background_timer(client, tmp_path, monkeypatch):
    response = client.post("/v1/generations", json={
        "prompt": "ocean", "duration_seconds": 0.5,
    })
    assert response.status_code == 201
    item = response.json()
    store = api._store_for(tmp_path)
    path = store.root / (item["id"] + ".wav")
    assert path.is_file()
    expiration = path.stat().st_mtime + store.ttl_seconds + 1
    monkeypatch.setattr(store, "_clock", lambda: expiration)
    assert client.get(item["audio_url"]).status_code == 404
    assert not path.exists()


def test_api_returns_bounded_storage_error_instead_of_unbounded_write(client, tmp_path, monkeypatch):
    restricted = LocalAudioStore(
        tmp_path / ".sonumio-api-audio", max_total_bytes=120, max_file_bytes=60,
    )
    monkeypatch.setattr(api, "_store_for", lambda root: restricted)
    response = client.post("/v1/generations", json={
        "prompt": "rain", "duration_seconds": 0.5,
    })
    assert response.status_code == 507
    assert "storage" in response.json()["detail"].lower()
    assert restricted.usage().files == 0


def test_queued_job_keeps_storage_from_submission_even_if_config_changes(client, tmp_path, monkeypatch):
    original = api.generate_audio
    entered = threading.Event()
    release = threading.Event()

    def slow_generate(**kwargs):
        entered.set()
        assert release.wait(4)
        return original(**kwargs)

    monkeypatch.setattr(api, "generate_audio", slow_generate)
    base = tmp_path
    destination_a = api._store_for(base)
    destination_b_path = tmp_path / "changed-output"
    try:
        created = client.post("/v1/jobs/generations", json={
            "prompt": "wind", "duration_seconds": 0.5,
        })
        assert created.status_code == 202
        assert entered.wait(3)
        monkeypatch.setattr(api, "OUTPUT_DIR", destination_b_path)
    finally:
        release.set()

    deadline = time.monotonic() + 5
    while True:
        result = client.get(created.json()["status_url"])
        assert result.status_code == 200
        if result.json()["status"] in ("succeeded", "failed", "cancelled"):
            break
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert result.json()["status"] == "succeeded"
    generated = result.json()["result"]
    assert (destination_a.root / (generated["id"] + ".wav")).is_file()
    assert client.get(generated["audio_url"]).status_code == 404
    monkeypatch.setattr(api, "OUTPUT_DIR", base)
    assert client.get(generated["audio_url"]).status_code == 200
