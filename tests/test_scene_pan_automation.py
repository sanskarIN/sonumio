"""Optional sample-exact stereo pan motion, preserving static/mono scene output."""
from __future__ import annotations

import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sonumio import api, jobs
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.scenes import (
    SceneLayer, _pan_sweep_gains, compose_scene, validate_scene_plan,
)


def _decode_wav(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getsampwidth() == 2
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        return pcm.reshape(-1, wav.getnchannels())


def test_exact_endpoints_and_equal_power_midpoint():
    left, right = _pan_sweep_gains(-1.0, 1.0, 1001)
    assert left.shape == right.shape == (1001,)
    assert left[0] == 1.0 and right[0] == 0.0
    assert left[-1] == 0.0 and right[-1] == 1.0
    np.testing.assert_allclose(left**2 + right**2, 1.0, atol=5e-15)
    assert left[500] == pytest.approx(right[500], abs=1e-14)
    assert np.all(np.diff(left) <= 1e-14)
    assert np.all(np.diff(right) >= -1e-14)


def test_reverse_and_single_frame_sweeps():
    left, right = _pan_sweep_gains(1.0, -1.0, 19)
    assert (left[0], right[0]) == (0.0, 1.0)
    assert (left[-1], right[-1]) == (1.0, 0.0)
    one_left, one_right = _pan_sweep_gains(-1.0, 1.0, 1)
    assert (one_left[0], one_right[0]) == (1.0, 0.0)
    with pytest.raises(ValueError):
        _pan_sweep_gains(-1.0, 1.0, 0)


@pytest.mark.parametrize("pan", [-1.0, -0.4, 0.0, 0.35, 1.0])
def test_static_pan_end_keeps_identical_wav_and_samples(pan):
    base = SceneLayer("wind", duration_seconds=0.5, seed=18, pan=pan)
    same = SceneLayer("wind", duration_seconds=0.5, seed=18,
                      pan=pan, pan_end=pan)
    left = compose_scene([base], 1.0)
    right = compose_scene([same], 1.0)
    assert left.wav_bytes == right.wav_bytes
    np.testing.assert_array_equal(left.result.samples, right.result.samples)
    assert _decode_wav(left.wav_bytes).shape[1] == (1 if pan == 0 else 2)


def test_center_to_right_sweep_requires_stereo_and_is_repeatable():
    layer = SceneLayer("rain", duration_seconds=0.5, seed=7,
                       pan=0.0, pan_end=1.0,
                       fade_in_seconds=0, fade_out_seconds=0)
    a = compose_scene([layer], 1.0)
    b = compose_scene([layer], 1.0)
    assert a.wav_bytes == b.wav_bytes
    pcm = _decode_wav(a.wav_bytes)
    assert pcm.shape == (24000, 2)
    assert np.any(pcm[:200, 0] != 0)
    assert np.any(pcm[-12000:-11800, 1] != 0)
    assert np.all(pcm[12000:, :] == 0)
    assert np.max(np.abs(a.result.samples)) <= 0.951
    assert a.metrics["clipping_fraction"] == 0.0


def test_full_left_right_sweep_changes_channels_across_layer():
    layer = SceneLayer("wind", duration_seconds=1.0, seed=3,
                       pan=-1.0, pan_end=1.0,
                       fade_in_seconds=0, fade_out_seconds=0)
    pcm = _decode_wav(compose_scene([layer], 1.0).wav_bytes)
    assert pcm.shape == (24000, 2)
    assert np.any(pcm[:1000, 0] != 0)
    assert np.any(pcm[-1000:, 1] != 0)
    assert np.max(np.abs(pcm[:1000, 1])) < np.max(np.abs(pcm[:1000, 0]))
    assert np.max(np.abs(pcm[-1000:, 0])) < np.max(np.abs(pcm[-1000:, 1]))


def test_overlap_and_moving_scene_master_gain_ceiling():
    layers = [SceneLayer(
        "rain", duration_seconds=1.0, seed=seed, gain_db=12,
        pan=-1.0 if seed % 2 else 1.0,
        pan_end=1.0 if seed % 2 else -1.0,
    ) for seed in range(8)]
    rendered = compose_scene(layers, 1.0)
    assert rendered.result.samples.shape == (2, 24000)
    assert np.max(np.abs(rendered.result.samples)) <= 0.951
    assert rendered.metrics["clipping_fraction"] == 0
    assert rendered.metrics["master_gain"] < 1


@pytest.mark.parametrize("bad", [
    -1.0001, 1.0001, float("nan"), float("inf"), -float("inf"),
    True, False, "right", {}, [],
])
def test_invalid_pan_end_rejected_before_expensive_render(bad, monkeypatch):
    def fail_on_render(*_a, **_kw):
        pytest.fail("Invalid plan reached procedural engine")
    monkeypatch.setattr("sonumio.scenes.generate_audio", fail_on_render)
    with pytest.raises(GenerationError):
        compose_scene([SceneLayer("rain", duration_seconds=0.5,
                                  pan_end=bad)], 1.0)


def test_cli_scene_pan_sweep_exports_stereo(tmp_path):
    source = tmp_path / "sweep.json"
    destination = tmp_path / "sweep.wav"
    source.write_text(json.dumps({
        "duration_seconds": 1.0,
        "layers": [{
            "prompt": "ocean", "duration_seconds": 0.5,
            "pan": -1, "pan_end": 1, "seed": 9,
        }],
    }), encoding="utf-8")
    assert main(["scene", str(source), "--output", str(destination)]) == 0
    assert _decode_wav(destination.read_bytes()).shape == (12000, 2)


def test_api_scene_pan_sweep_and_invalid_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    with TestClient(api.app) as client:
        payload = {
            "duration_seconds": 1.0,
            "layers": [{
                "prompt": "rain", "duration_seconds": 0.5,
                "pan": -1.0, "pan_end": 1.0,
            }],
        }
        response = client.post("/v1/scenes", json=payload)
        assert response.status_code == 201, response.text
        assert response.json()["channels"] == 2
        wav = client.get(response.json()["audio_url"])
        assert wav.status_code == 200
        assert _decode_wav(wav.content).shape == (12000, 2)
        payload["layers"][0]["pan_end"] = 1.1
        assert client.post("/v1/scenes", json=payload).status_code == 422


def test_queue_rejects_invalid_sweep_before_enqueue(tmp_path, monkeypatch):
    queue = jobs.LocalJobQueue(max_active=2, max_history=4)
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "JOBS", queue)
    try:
        with TestClient(api.app) as client:
            response = client.post("/v1/jobs/scenes", json={
                "duration_seconds": 1.0,
                "layers": [{"prompt": "wind", "duration_seconds": 0.5,
                            "pan": 0.0, "pan_end": -1.5}],
            })
            assert response.status_code == 422
            assert not queue._jobs
    finally:
        queue.shutdown()
