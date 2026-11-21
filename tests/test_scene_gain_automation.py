"""Scene layer gain sweeps: deterministic, frame-exact, backwards compatible."""
from __future__ import annotations

import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sonumio import api, jobs
from sonumio.audio import AudioResult, SAMPLE_RATE
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.scenes import SceneLayer, _gain_sweep_db, compose_scene
from sonumio.service import Generated


def _pcm(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getsampwidth() == 2
        return np.frombuffer(wav.readframes(wav.getnframes()),
                             dtype="<i2").reshape(-1, wav.getnchannels())


def test_frame_exact_logarithmic_endpoints_and_constant_rate():
    gains = _gain_sweep_db(0.0, -60.0, 1001)
    assert gains.shape == (1001,)
    assert gains[0] == 1.0
    assert gains[-1] == pytest.approx(0.001, abs=1e-16)
    assert gains[500] == pytest.approx(10 ** (-30 / 20), rel=1e-14)
    assert np.all(np.diff(gains) < 0)
    assert _gain_sweep_db(-6.0, 12.0, 1)[0] == pytest.approx(10 ** (-6 / 20))
    with pytest.raises(ValueError):
        _gain_sweep_db(0, -12, 0)


@pytest.mark.parametrize("pan,pan_end", [(0, None), (-1, None), (0, 1), (1, -1)])
def test_unchanged_static_gain_wav_even_with_explicit_equal_endpoint(pan, pan_end):
    layer = SceneLayer("rain", duration_seconds=0.5, gain_db=-7.5,
                       seed=5, pan=pan, pan_end=pan_end)
    same = SceneLayer("rain", duration_seconds=0.5, gain_db=-7.5,
                      gain_end_db=-7.5, seed=5, pan=pan, pan_end=pan_end)
    a = compose_scene([layer], 1.0)
    b = compose_scene([same], 1.0)
    assert a.wav_bytes == b.wav_bytes
    np.testing.assert_array_equal(a.result.samples, b.result.samples)


def test_mono_gain_sweep_uses_frame_position_not_scene_position(monkeypatch):
    def constant(prompt, duration_seconds, model_id, seed):
        return Generated(
            result=AudioResult(np.full(round(SAMPLE_RATE * duration_seconds), 0.1,
                                       dtype=np.float32), SAMPLE_RATE, model_id, seed),
            wav_bytes=b"", metrics={},
        )
    monkeypatch.setattr("sonumio.scenes.generate_audio", constant)
    layer = SceneLayer("rain", start_seconds=0.25, duration_seconds=0.5,
                       fade_in_seconds=0, fade_out_seconds=0,
                       gain_db=0, gain_end_db=-12)
    actual = compose_scene([layer], 1.0)
    samples = actual.result.samples
    assert samples.shape == (SAMPLE_RATE,)
    assert np.all(samples[:6000] == 0)
    assert samples[6000] == pytest.approx(0.1, abs=1e-7)
    assert samples[6000 + 6000] == pytest.approx(0.1 * 10 ** (-6 / 20), abs=1e-5)
    assert samples[17999] == pytest.approx(0.1 * 10 ** (-12 / 20), abs=1e-7)
    assert np.all(samples[18000:] == 0)
    assert actual.metrics["master_gain"] == 1
    assert _pcm(actual.wav_bytes).shape == (24000, 1)


def test_stereo_gain_sweep_with_pan_sweep_and_master_limit():
    layers = [SceneLayer("wind", duration_seconds=0.5, gain_db=12, gain_end_db=-20,
                         pan=-1, pan_end=1, seed=i) for i in range(8)]
    rendered = compose_scene(layers, 1)
    assert rendered.result.samples.shape == (2, SAMPLE_RATE)
    assert rendered.wav_bytes == compose_scene(layers, 1).wav_bytes
    assert np.max(np.abs(rendered.result.samples)) <= 0.951
    assert rendered.metrics["clipping_fraction"] == 0
    assert rendered.metrics["master_gain"] < 1
    assert np.any(_pcm(rendered.wav_bytes)[:, 0] != 0)
    assert np.any(_pcm(rendered.wav_bytes)[:, 1] != 0)


@pytest.mark.parametrize("invalid", [-60.001, 12.001, float("nan"), float("inf"),
                                      -float("inf"), True, False, "loud"])
def test_invalid_end_gain_is_rejected_before_render(invalid, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("invalid layer reached the procedural renderer")
    monkeypatch.setattr("sonumio.scenes.generate_audio", fail)
    with pytest.raises(GenerationError):
        compose_scene([SceneLayer("rain", duration_seconds=0.5,
                                  gain_end_db=invalid)], 1.0)


def test_cli_scene_gain_automation(tmp_path):
    source = tmp_path / "plan.json"
    target = tmp_path / "mix.wav"
    source.write_text(json.dumps({
        "duration_seconds": 1,
        "layers": [{"prompt": "rain", "duration_seconds": 0.5,
                    "gain_db": -5, "gain_end_db": -30,
                    "pan": -1, "pan_end": 1}],
    }), encoding="utf-8")
    assert main(["scene", str(source), "--output", str(target)]) == 0
    assert _pcm(target.read_bytes()).shape == (24000, 2)


def test_sync_and_queued_api_gain_end_support(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    queue = jobs.LocalJobQueue(max_active=2, max_history=4)
    monkeypatch.setattr(api, "JOBS", queue)
    payload = {"duration_seconds": 1.0, "layers": [{
        "prompt": "rain", "duration_seconds": 0.5,
        "gain_db": -3, "gain_end_db": -30, "seed": 4,
    }]}
    try:
        with TestClient(api.app) as client:
            direct = client.post("/v1/scenes", json=payload)
            assert direct.status_code == 201, direct.text
            wav = client.get(direct.json()["audio_url"])
            assert wav.status_code == 200
            assert _pcm(wav.content).shape == (24000, 1)
            submitted = client.post("/v1/jobs/scenes", json=payload)
            assert submitted.status_code == 202, submitted.text
            job_id = submitted.json()["id"]
            import time
            for _ in range(100):
                status = client.get(f"/v1/jobs/{job_id}").json()
                if status["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.02)
            assert status["status"] == "succeeded", status
            queued_wav = client.get(status["result"]["audio_url"])
            assert queued_wav.status_code == 200
            assert queued_wav.content == wav.content
            payload["layers"][0]["gain_end_db"] = 12.1
            assert client.post("/v1/scenes", json=payload).status_code == 422
            assert client.post("/v1/jobs/scenes", json=payload).status_code == 422
    finally:
        queue.shutdown()
