"""Strict audio core ownership, clock, structure and fingerprint guarantees."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from sonumio.audio_core.contracts import (
    AudioBuffer, AudioLayout, AudioSpec, as_channel_first, check_audio,
    check_rate, make_buffer, validate_allocation,
)
from sonumio.audio_core.errors import (
    AudioContractError, AudioIntegrityError, AudioLimitError,
)
from sonumio.audio_core.hashing import (
    canonical_json_bytes, checksum_record, digest_audio, digest_file,
    parse_canonical_json, strict_sha256, verify_record,
)
from sonumio.audio_core.timebase import (
    FrameInterval, FramePosition, TimeGrid, frames_from_seconds,
    seconds_from_frames, split_intervals,
)


@pytest.fixture
def mono():
    return make_buffer(np.linspace(-.5, .5, 128, dtype=np.float32), 24_000)


@pytest.fixture
def stereo():
    return make_buffer(np.zeros((2, 128), dtype=np.float32), 24_000)


def test_channel_first_mono_and_stereo_buffer_metadata(mono, stereo):
    assert mono.samples.shape == (1, 128)
    assert mono.spec.dtype == "float32"
    assert mono.spec.layout.name == "mono"
    assert mono.frames == 128 and mono.channels == 1
    assert mono.seconds == 128 / 24_000
    assert mono.spec.byte_count == 512
    assert stereo.samples.shape == (2, 128)
    assert stereo.spec.layout.name == "stereo"
    assert stereo.spec.byte_count == 1024
    assert mono.spec.same_clock(stereo.spec)
    assert not mono.spec.same_shape(stereo.spec)


def test_memory_ownership_and_readonly_sample_contract():
    original = np.arange(30, dtype=np.float32)
    owned = make_buffer(original, 44_100)
    original[:] = 0
    assert owned.samples[0, 8] == 8
    with pytest.raises(ValueError):
        owned.samples[0, 0] = 77
    duplicate = owned.mutable_copy()
    duplicate[:] = -7
    assert owned.samples[0, 0] == 0
    assert owned.samples[0, 8] == 8
    with pytest.raises(FrozenInstanceError):
        owned.spec = AudioSpec(44_100, AudioLayout.for_channels(1), 30)
    assert not np.shares_memory(duplicate, owned.samples)


def test_channel_last_requires_explicit_switch():
    data = np.arange(30, dtype=np.float32).reshape(10, 3)
    with pytest.raises(AudioLimitError):
        as_channel_first(data)
    transformed = as_channel_first(data, channels_last=True)
    assert transformed.shape == (3, 10)
    np.testing.assert_array_equal(transformed[1], data[:, 1])


@pytest.mark.parametrize("rate", [0, 7999, 192001, True, None, 24000.0, -1])
def test_wrong_sample_rate_fails_closed(rate):
    with pytest.raises(AudioContractError):
        check_rate(rate)


def test_high_rate_and_overrange_handling_is_explicit():
    source = np.array([0.0, -1.2, 1.5], dtype=np.float32)
    check_audio(source)
    with pytest.raises(AudioContractError, match="exceeds"):
        check_audio(source, allow_overrange=False)
    with pytest.raises(AudioContractError):
        check_audio(source.astype(np.float64), allow_float64=False)


def test_nonfinite_object_int_scalar_and_empty_samples_rejected():
    for bad in (
        np.array([], dtype=np.float32),
        np.array([1.0, np.nan], dtype=np.float32),
        np.array([1.0, np.inf], dtype=np.float64),
        np.array([1, 2], dtype=np.int32),
        np.array(["sound"], dtype=object),
        np.array(4.0, dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
    ):
        with pytest.raises(AudioContractError):
            check_audio(bad)


def test_layout_requires_explicit_geometry_and_unique_labels():
    assert AudioLayout.for_channels(1).labels == ("M",)
    assert AudioLayout.for_channels(2).labels == ("L", "R")
    with pytest.raises(AudioContractError):
        AudioLayout(2, "mono")
    with pytest.raises(AudioContractError):
        AudioLayout(1, "stereo")
    with pytest.raises(AudioContractError):
        AudioLayout(3, "discrete", ("A", "A", "B"))
    with pytest.raises(AudioContractError):
        AudioLayout(2, "discrete", ("A",))
    with pytest.raises(AudioContractError):
        AudioLayout(9)


def test_shape_allocation_and_duration_limits_are_enforced():
    assert validate_allocation(2, 100) == 800
    for args in ((9, 1), (1, 0), (1, 1, 0)):
        with pytest.raises(AudioContractError):
            validate_allocation(*args)
    with pytest.raises(AudioLimitError):
        validate_allocation(2, 100_000_000)
    with pytest.raises(AudioLimitError):
        AudioSpec(24_000, AudioLayout.for_channels(1), 24_000 * 121)
    with pytest.raises(AudioContractError):
        AudioSpec(24_000, AudioLayout.for_channels(1), 2, "int16")


def test_copy_with_and_frame_slices_preserve_sample_clock(mono):
    changed = mono.copy_with(np.ones((1, 20), dtype=np.float32))
    assert changed.frames == 20
    assert changed.sample_rate == mono.sample_rate
    assert changed.spec.layout == mono.spec.layout
    piece = mono.sliced(2, 18)
    assert piece.frames == 16
    np.testing.assert_array_equal(piece.samples[0], mono.samples[0, 2:18])
    for a, b in ((-1, 2), (1, 1), (0, 129), (False, 10)):
        with pytest.raises(AudioContractError):
            mono.sliced(a, b)


def test_frame_rounding_half_even_and_exact_fraction_clock():
    assert frames_from_seconds(0, 8_000) == 0
    assert frames_from_seconds(Decimal("0.0000625"), 8_000) == 0
    assert frames_from_seconds(Decimal("0.0001875"), 8_000) == 2
    assert frames_from_seconds(Fraction(1, 4), 8_000) == 2000
    assert seconds_from_frames(48000, 48000) == Fraction(1)
    with pytest.raises(AudioContractError):
        frames_from_seconds(0, 24000, allow_zero=False)
    with pytest.raises(AudioLimitError):
        frames_from_seconds(-1, 24000)
    with pytest.raises(AudioContractError):
        frames_from_seconds(float("inf"), 24000)


def test_frame_positions_and_half_open_intersections():
    p = FramePosition(5, 24000)
    q = p.advance(10)
    assert q.value == 15 and q.distance(p) == 10
    assert p.seconds == Fraction(5, 24000)
    a = FrameInterval(0, 10, 24000)
    b = FrameInterval(10, 20, 24000)
    c = FrameInterval(5, 15, 24000)
    assert not a.overlaps(b)
    assert a.touches(b)
    assert a.union(b) == FrameInterval(0, 20, 24000)
    assert a.intersection(b) is None
    assert a.intersection(c) == FrameInterval(5, 10, 24000)
    assert c.shifted(3) == FrameInterval(8, 18, 24000)
    with pytest.raises(AudioContractError):
        a.overlaps(FrameInterval(0, 2, 44100))


def test_timegrid_clips_and_overlap_chunking():
    grid = TimeGrid(24000, 1000)
    assert grid.duration == Fraction(1000, 24000)
    assert grid.at(Fraction(1, 48)).value == 500
    interval = FrameInterval(200, 400, 24000)
    grid.validate_interval(interval)
    assert [x.frames for x in grid.partition(256)] == [256, 256, 256, 232]
    groups = list(split_intervals(1000, 24000, 400, overlap=100))
    assert [(x.start, x.stop) for x in groups] == [
        (0, 400), (300, 700), (600, 1000)
    ]
    with pytest.raises(AudioContractError):
        grid.validate_interval(FrameInterval(0, 1001, 24000))


def test_canonical_json_is_order_independent_and_rejects_nonfinite():
    a = {"name": "雨", "source": [1, 2], "valid": True}
    b = {"valid": True, "source": [1, 2], "name": "雨"}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    assert parse_canonical_json(canonical_json_bytes(a)) == a
    with pytest.raises(AudioContractError):
        canonical_json_bytes({"x": float("nan")})
    with pytest.raises(AudioContractError):
        canonical_json_bytes({"x": {3: 1}})
    with pytest.raises(AudioContractError):
        parse_canonical_json(b'{"a":1,"a":2}')


def test_audio_sha_changes_with_pcm_sample_and_sample_clock(mono):
    a = digest_audio(mono)
    assert len(a) == 64
    assert digest_audio(mono) == a
    edited = mono.mutable_copy()
    edited[0, 4] += .01
    assert digest_audio(mono.copy_with(edited)) != a
    other_clock = make_buffer(mono.mutable_copy(), 44_100)
    assert digest_audio(other_clock) != a
    assert digest_audio(mono.samples, sample_rate=24000) == a


def test_provenance_checksum_tampering_and_payload_hash_enforcement(mono):
    payload = digest_audio(mono)
    record = checksum_record({"creator": "research-test", "rights": "not-reviewed"},
                             payload_sha256=payload)
    assert verify_record(record)["creator"] == "research-test"
    assert verify_record(record, expected_payload=payload)["payload_sha256"] == payload
    forgery = dict(record, creator="different")
    with pytest.raises(AudioIntegrityError):
        verify_record(forgery)
    with pytest.raises(AudioIntegrityError):
        verify_record(record, expected_payload="0" * 64)
    with pytest.raises(AudioContractError):
        strict_sha256("F" * 64)


def test_file_digest_uses_bounded_snapshot_and_rejects_symlink(tmp_path):
    data = tmp_path / "owned.bin"
    data.write_bytes(b"test-case")
    assert digest_file(data) == hashlib.sha256(b"test-case").hexdigest()
    with pytest.raises(AudioLimitError):
        digest_file(data, max_bytes=2)
    link = tmp_path / "alias.bin"
    link.symlink_to(data)
    with pytest.raises(AudioContractError):
        digest_file(link)
