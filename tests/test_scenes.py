import io
import json
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sonumio import api
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.scenes import SceneLayer, compose_scene


def test_two_layer_scene_valid_and_deterministic():
    layers = [SceneLayer("rain", 0, 1.3, gain_db=-6, seed=10),
              SceneLayer("birds in a forest", 0.8, 1.2, seed=4)]
    a = compose_scene(layers, 2.5)
    b = compose_scene(layers, 2.5)
    assert a.wav_bytes == b.wav_bytes
    assert a.metrics["layer_count"] == 2
    assert a.metrics["clipping_fraction"] == 0
    with wave.open(io.BytesIO(a.wav_bytes), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getnframes() == 60_000
        pcm = np.frombuffer(wav.readframes(60_000), dtype="<i2")
        assert pcm.std() > 0
        assert pcm[:int(0.7 * 24_000)].std() > 0
        assert pcm[-1000:].std() == 0


def test_scene_has_silent_leadin_and_fades():
    result = compose_scene([SceneLayer("wind", 0.5, 0.5, fade_in_seconds=0.1, fade_out_seconds=0.1)], 2)
    samples = result.result.samples
    assert np.all(samples[:12000] == 0)
    assert samples[12000] == 0
    assert abs(samples[24000]) < 1e-6
    assert result.metrics["pre_limit_peak"] > 0


@pytest.mark.parametrize("layer,total", [
    (SceneLayer("rain", 1.2, 2), 3),
    (SceneLayer("rain", 0, 1, gain_db=float("nan")), 2),
    (SceneLayer("rain", 0, 1, fade_in_seconds=0.8, fade_out_seconds=0.5), 2),
    (SceneLayer("rain", -1, 1), 2),
    (SceneLayer("rain", 0, 1, seed=True), 2),
    (SceneLayer("unrecognized", 0, 1), 2),
])
def test_scene_rejects_invalid_layers(layer, total):
    with pytest.raises(GenerationError):
        compose_scene([layer], total)


def test_scene_limit():
    with pytest.raises(GenerationError):
        compose_scene([], 2)
    with pytest.raises(GenerationError):
        compose_scene([SceneLayer("rain")] * 9, 3)


def test_scene_api_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    client = TestClient(api.app)
    response = client.post("/v1/scenes", json={
        "duration_seconds": 2.0,
        "layers": [
            {"prompt": "rain", "duration_seconds": 1.0},
            {"prompt": "thunder", "duration_seconds": 0.5, "start_seconds": 1.0},
        ],
    })
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["layer_count"] == 2
    audio = client.get(payload["audio_url"])
    assert audio.status_code == 200
    assert audio.content[:4] == b"RIFF"
    assert client.post("/v1/scenes", json={"layers": [], "duration_seconds": 2}).status_code == 422
    assert client.post("/v1/scenes", json={"layers": [{"prompt": "rain", "start_seconds": 28}]}).status_code == 422


def test_scene_cli(tmp_path):
    payload = {"duration_seconds": 1.5,
               "layers": [{"prompt": "ocean", "duration_seconds": 0.5, "start_seconds": 0.4}]}
    source, dest = tmp_path / "scene.json", tmp_path / "scene.wav"
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["scene", str(source), "--output", str(dest)]) == 0
    assert dest.read_bytes().startswith(b"RIFF")
