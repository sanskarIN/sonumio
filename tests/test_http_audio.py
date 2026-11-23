"""Deterministic HTTP range protocol regression tests (no generation needed)."""
import pytest

from sonumio.http_audio import wav_http_response

AUDIO = b"RIFF" + bytes(range(96))


@pytest.mark.parametrize(("header", "status", "expected"), [
    ("bytes=0-0", 206, AUDIO[:1]),
    ("bytes=4-11", 206, AUDIO[4:12]),
    ("bytes=98-", 206, AUDIO[98:]),
    ("bytes=-1", 206, AUDIO[-1:]),
    ("bytes=-1000", 206, AUDIO),
    ("bytes=0-999", 206, AUDIO),
    ("BYTES=1-2", 206, AUDIO[1:3]),
    ("bytes=100-", 416, b""),
    ("bytes=-0", 416, b""),
    ("bytes=9-3", 416, b""),
    ("bytes=0-0,2-3", 200, AUDIO),
    ("nibbles=0-1", 200, AUDIO),
    ("bytes=wat", 200, AUDIO),
    ("bytes=0-+2", 200, AUDIO),
    ("bytes=1-", 206, AUDIO[1:]),
    ("bytes=" + "9"*150 + "-", 200, AUDIO),
])
def test_range_response(header, status, expected):
    response = wav_http_response(AUDIO, range_header=header)
    assert response.status_code == status
    assert response.body == expected
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-length"] == str(len(expected))
    assert response.headers["content-type"] == "audio/wav"
    if status == 206:
        assert response.headers["content-range"].endswith("/100")
    if status == 416:
        assert response.headers["content-range"] == "bytes */100"


def test_if_range_without_validator_falls_back_to_entire_file():
    response = wav_http_response(
        AUDIO, range_header="bytes=1-4", if_range='"unknown-etag"'
    )
    assert response.status_code == 200
    assert response.body == AUDIO
    assert "content-range" not in response.headers


def test_head_has_length_of_selected_representation_without_body():
    response = wav_http_response(AUDIO, range_header="bytes=6-9", head=True)
    assert response.status_code == 200
    assert response.body == b""
    assert response.headers["content-length"] == str(len(AUDIO))
    assert "content-range" not in response.headers


def test_full_response_keeps_existing_download_headers():
    response = wav_http_response(AUDIO)
    assert response.status_code == 200
    assert response.body == AUDIO
    assert "sonumio.wav" in response.headers["content-disposition"]
