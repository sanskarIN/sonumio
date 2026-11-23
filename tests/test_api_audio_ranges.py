"""Full local API checks for WAV byte ranges, seeking and HEAD metadata."""
from __future__ import annotations

from io import BytesIO
import wave

import pytest
from fastapi.testclient import TestClient

from sonumio import api


def _wav(frames: int = 64) -> bytes:
    stream = BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(b"\x01\x02" * frames)
    return stream.getvalue()


@pytest.fixture
def preview(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    store = api._store_for(tmp_path)
    audio = _wav()
    identifier = store.write(audio)
    with TestClient(api.app) as client:
        yield client, f"/v1/generations/{identifier}/audio", audio


def test_full_download_preserves_backward_compatibility(preview):
    client, url, audio = preview
    response = client.get(url)
    assert response.status_code == 200
    assert response.content == audio
    assert response.headers["content-length"] == str(len(audio))
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"] == "audio/wav"
    assert 'filename="sonumio.wav"' in response.headers["content-disposition"]


@pytest.mark.parametrize("header, offset, finish", [
    ("bytes=0-0", 0, 0),
    ("bytes=4-15", 4, 15),
    ("bytes=-10", -10, None),
    ("bytes=10-", 10, None),
    ("bytes=0-999999", 0, None),
])
def test_single_ranges_are_exact(preview, header, offset, finish):
    client, url, audio = preview
    response = client.get(url, headers={"Range": header})
    assert response.status_code == 206
    assert response.headers["accept-ranges"] == "bytes"
    assert response.content == (audio[offset:] if finish is None else audio[offset:finish+1])
    assert response.headers["content-length"] == str(len(response.content))
    lower = len(audio) + offset if offset < 0 else offset
    upper = len(audio) - 1 if finish is None else finish
    assert response.headers["content-range"] == f"bytes {lower}-{upper}/{len(audio)}"


@pytest.mark.parametrize("header", ["bytes=999999-", "bytes=-0", "bytes=20-10"])
def test_unsatisfiable_ranges_have_bounded_416_response(preview, header):
    client, url, audio = preview
    response = client.get(url, headers={"Range": header})
    assert response.status_code == 416
    assert response.content == b""
    assert response.headers["content-range"] == f"bytes */{len(audio)}"
    assert response.headers["content-length"] == "0"
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("header", [
    "bytes=bad", "bytes=0-1,8-9", "items=0-1",
    "bytes=+1-2", "bytes=" + "1"*150 + "-",
])
def test_malformed_and_multipart_ranges_fallback_to_200(preview, header):
    client, url, audio = preview
    response = client.get(url, headers={"Range": header})
    assert response.status_code == 200
    assert response.content == audio
    assert "content-range" not in response.headers


def test_if_range_without_server_validator_never_serves_stale_partial(preview):
    client, url, audio = preview
    response = client.get(url, headers={"Range": "bytes=5-10", "If-Range": '"stale"'})
    assert response.status_code == 200
    assert response.content == audio
    assert "content-range" not in response.headers


def test_head_reports_correct_lengths_without_body(preview):
    client, url, audio = preview
    full = client.head(url)
    assert full.status_code == 200
    assert full.content == b""
    assert full.headers["content-length"] == str(len(audio))
    partial = client.head(url, headers={"Range": "bytes=5-11"})
    assert partial.status_code == 200
    assert partial.content == b""
    assert "content-range" not in partial.headers
    assert partial.headers["content-length"] == str(len(audio))


def test_missing_audio_is_not_exposed_by_range_or_head(preview):
    client, _, _ = preview
    url = "/v1/generations/00000000-0000-0000-0000-000000000000/audio"
    assert client.get(url, headers={"Range": "bytes=0-2"}).status_code == 404
    assert client.head(url).status_code == 404
