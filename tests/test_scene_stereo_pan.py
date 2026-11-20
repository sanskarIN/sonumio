"""Stereo scene spatialization: deterministic PCM and safe opt-in semantics."""
from __future__ import annotations

import io
import json
import math
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sonumio import api
from sonumio.audio import encode_wav_channels, metrics_channels
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.scenes import SceneLayer, compose_scene


def _wav(wav_bytes: bytes) -> tuple[int, int, np.ndarray]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        channels = handle.getnchannels()
        frames = handle.getnframes()
        assert handle.getframerate() == 24_000
        assert handle.getsampwidth() == 2
        raw = np.frombuffer(handle.readframes(frames), dtype="<i2")
    return channels, frames, raw.reshape(frames, channels)


@pytest.mark.parametrize("pan,live_channel,quiet_channel", [
    (-1.0, 0, 1), (1.0, 1, 0),
])
def test_hard_pan_uses_two_channels_and_silences_opposite_side(
    pan, live_channel, quiet_channel
):
    layer = SceneLayer("rain", duration_seconds=0.5,
                       fade_in_seconds=0, fade_out_seconds=0,
                       seed=17, pan=pan)
    rendered = compose_scene([layer], 1.0)
    channels, frames, samples = _wav(rendered.wav_bytes)
    assert channels == 2
    assert frames == 24_000
    assert np.max(np.abs(samples[:12_000, live_channel])) > 100
    assert np.all(samples[:, quiet_channel] == 0)
    assert np.all(samples[12_000:] == 0)
    assert rendered.result.samples.shape == (2, 24_000)
    assert rendered.metrics["clipping_fraction"] == 0.0


def test_mono_default_path_is_unchanged_and_explicit_center_remains_mono():
    original = compose_scene([SceneLayer("rain", duration_seconds=0.5, seed=9)], 1.0)
    centered = compose_scene([SceneLayer("rain", duration_seconds=0.5,
                                         seed=9, pan=0.0)], 1.0)
    assert original.wav_bytes == centered.wav_bytes
    channels, frames, _ = _wav(original.wav_bytes)
    assert channels == 1 and frames == 24_000
    assert original.result.samples.ndim == 1


def test_stereo_scene_determinism_alignment_and_channel_separation():
    layers = [
        SceneLayer("wind", start_seconds=0, duration_seconds=1.5,
                   gain_db=-3, seed=3, pan=-0.75),
        SceneLayer("birds", start_seconds=0.5, duration_seconds=1,
                   gain_db=-6, seed=7, pan=0.75),
    ]
    a = compose_scene(layers, 2.0)
    b = compose_scene(layers, 2.0)
    assert a.wav_bytes == b.wav_bytes
    assert a.metrics["clipping_fraction"] == 0
    channels, frames, samples = _wav(a.wav_bytes)
    assert channels == 2 and frames == 48_000
    assert not np.array_equal(samples[:, 0], samples[:, 1])
    assert np.all(samples[36_000:] == 0)


def test_stereo_peak_metrics_do_not_downmix_antiphase():
    samples = np.array([[0.9, -0.9], [-0.9, 0.9]], dtype=np.float32)
    stats = metrics_channels(samples)
    assert stats["peak_dbfs"] == pytest.approx(
        round(20 * math.log10(float(samples.max())), 3)
    )
    assert stats["clipping_fraction"] == 0.0
    channels, frames, encoded = _wav(encode_wav_channels(samples, 24_000))
    assert (channels, frames) == (2, 2)
    assert encoded[0, 0] > 0 and encoded[0, 1] < 0
    assert encoded[1, 0] < 0 and encoded[1, 1] > 0


@pytest.mark.parametrize("bad", [
    np.zeros(10), np.zeros((3, 10)), np.zeros((2, 0)),
    np.array([[float("nan")], [1.0]]),
    np.array([[float("inf")], [0.0]]),
])
def test_multichannel_encoder_rejects_unsafe_shapes_and_samples(bad):
    with pytest.raises(ValueError):
        encode_wav_channels(bad, 24_000)
    with pytest.raises(ValueError):
        metrics_channels(bad)


@pytest.mark.parametrize("pan", [-1.01, 1.01, float("nan"), float("inf"), True, "left"])
def test_scene_rejects_invalid_pan_without_rendering(pan):
    with pytest.raises(GenerationError):
        compose_scene([SceneLayer("rain", duration_seconds=0.5, pan=pan)], 1)


def test_cli_scene_pan_produces_stereo_wav(tmp_path):
    source = tmp_path / "scene.json"
    destination = tmp_path / "out.wav"
    source.write_text(json.dumps({
        "duration_seconds": 1,
        "layers": [{"prompt": "rain", "duration_seconds": 0.5,
                    "pan": -1.0, "seed": 3}],
    }), encoding="utf-8")
    assert main(["scene", str(source), "--output", str(destination)]) == 0
    assert _wav(destination.read_bytes())[0] == 2


def test_api_scene_pan_roundtrip_and_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    client = TestClient(api.app)
    payload = {
        "duration_seconds": 1,
        "layers": [{"prompt": "wind", "duration_seconds": 0.5,
                    "pan": 1.0, "seed": 5}],
    }
    response = client.post("/v1/scenes", json=payload)
    assert response.status_code == 201, response.text
    assert response.json()["channels"] == 2
    wav = client.get(response.json()["audio_url"])
    assert wav.status_code == 200
    assert _wav(wav.content)[0] == 2
    payload["layers"][0]["pan"] = 1.5
    assert client.post("/v1/scenes", json=payload).status_code == 422


def test_stereo_scene_master_gain_preserves_ceiling():
    layers = [SceneLayer("rain", duration_seconds=1, seed=n,
                         gain_db=12, pan=-1 if n % 2 else 1)
              for n in range(8)]
    scene = compose_scene(layers, 1)
    assert scene.result.samples.ndim == 2
    assert np.max(np.abs(scene.result.samples)) <= 0.951
    assert scene.metrics["clipping_fraction"] == 0.0
    assert scene.metrics["master_gain"] < 1.0
