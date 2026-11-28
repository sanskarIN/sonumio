"""Integrity, geometry, provenance and publication tests for music stem ZIPs."""
from __future__ import annotations

import hashlib
import io
import json
import struct
import wave
import zipfile

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan, TRACKS, arrangement_digest, compose_arrangement
from sonumio.audio import SAMPLE_RATE
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.music_score_io import encode_score
from sonumio.music_stem_pack import (
    MANIFEST_NAME, SCORE_NAME, MAX_STEM_PACK_BYTES, PEAK_CEILING,
    create_stem_pack, verify_stem_pack, select_stem, publish_stem_pack,
    read_stem_pack, _pack, _info, _inspect_stem,
)


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4, bpm=160, seed=53))


@pytest.fixture(scope="module")
def bundle(score):
    return create_stem_pack(score)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unpack(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        return (archive.read(MANIFEST_NAME), archive.read(SCORE_NAME),
                {track: archive.read("stems/" + track + ".wav") for track in TRACKS})


def rewrite(raw, transform):
    sink = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as reader:
        with zipfile.ZipFile(sink, "w", allowZip64=False) as writer:
            for info in reader.infolist():
                name, data = info.filename, reader.read(info)
                result = transform(name, data)
                if result is not None:
                    writer.writestr(_info(result[0]), result[1])
    return sink.getvalue()


def test_archive_canonical_order_score_and_geometry(score, bundle):
    raw, receipt = bundle
    assert receipt.archive_sha256 == sha(raw)
    assert receipt.archive_bytes == len(raw)
    assert receipt.manifest["score_sha256"] == arrangement_digest(score)
    assert receipt.manifest["score_bytes_sha256"] == sha(encode_score(score))
    assert receipt.manifest["frames"] == score.plan.frame(score.plan.ticks)
    assert receipt.replay_verified is False
    assert receipt.as_dict()["signature_verified"] is False
    assert 0 < receipt.manifest["common_attenuation"] <= 1
    assert receipt.manifest["room_effects_applied"] is False
    assert receipt.manifest["static_mix_gains_applied"] is False
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert archive.namelist() == [
            MANIFEST_NAME, SCORE_NAME,
            *["stems/" + track + ".wav" for track in sorted(TRACKS)],
        ]
        assert archive.read(SCORE_NAME) == encode_score(score)
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        for row in receipt.manifest["stems"]:
            member = archive.read(row["member"])
            assert row["wav_sha256"] == sha(member)
            assert row["bytes"] == len(member)
            with wave.open(io.BytesIO(member)) as audio:
                assert audio.getnchannels() == 2
                assert audio.getsampwidth() == 2
                assert audio.getframerate() == SAMPLE_RATE
                assert audio.getnframes() == receipt.manifest["frames"]
                pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
                pcm = pcm.reshape(-1, 2).astype(np.int64)
            assert row["peak_pcm16"] == np.max(np.abs(pcm), axis=0).tolist()
            assert row["sample_sum"] == np.sum(pcm, axis=0).tolist()
            assert row["sample_square_sum"] == np.sum(pcm * pcm, axis=0).tolist()
            assert row["nonzero_samples"] == np.count_nonzero(pcm, axis=0).tolist()
            assert max(row["peak_pcm16"]) <= round(PEAK_CEILING * 32767)


def test_same_runtime_replay_and_exact_selected_member(score, bundle):
    raw, original = bundle
    assert create_stem_pack(score)[0] == raw
    checked = verify_stem_pack(raw, expected_sha256=sha(raw), replay=True)
    assert checked.replay_verified
    assert checked.manifest == original.manifest
    assert checked.as_dict()["source_peak_replayed"]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for track in TRACKS:
            wav, receipt = select_stem(raw, track, expected_sha256=sha(raw))
            assert wav == archive.read("stems/" + track + ".wav")
            assert receipt.archive_sha256 == sha(raw)


def test_declared_source_peak_requires_replay_for_provenance(bundle):
    raw, _ = bundle
    manifest_raw, score_raw, members = unpack(raw)
    manifest = json.loads(manifest_raw)
    # A malicious publisher can rewrite internally consistent metadata.
    # Integrity alone does not prove actual rendering.
    manifest["declared_dry_source_peak"] = 100.
    manifest["common_attenuation"] = PEAK_CEILING / 100.
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    modified = _pack(canonical, score_raw, members)
    assert verify_stem_pack(modified).replay_verified is False
    with pytest.raises(GenerationError, match="replay mismatch"):
        verify_stem_pack(modified, replay=True)


def test_common_scale_preserves_relative_levels(score, monkeypatch):
    frames = score.plan.frame(score.plan.ticks)
    values = {track: (index + 1) / 4. for index, track in enumerate(TRACKS)}

    def fake_render(_score):
        return {track: np.full((2, frames), values[track], dtype=np.float32)
                for track in TRACKS}

    monkeypatch.setattr("sonumio.music_stem_pack.render_stems", fake_render)
    _, receipt = create_stem_pack(score)
    assert receipt.manifest["declared_dry_source_peak"] == 2.
    assert receipt.manifest["common_attenuation"] == .45
    for row in receipt.manifest["stems"]:
        expected = round(values[row["track"]] * .45 * 32767)
        assert row["peak_pcm16"] == [expected, expected]


def test_declared_peak_type_and_nonfinite_rejected(bundle):
    raw, _ = bundle
    manifest_raw, score_raw, members = unpack(raw)
    for peak in (True, -1, "2", None, float("inf")):
        manifest = json.loads(manifest_raw)
        manifest["declared_dry_source_peak"] = peak
        modified = _pack(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
                         score_raw, members)
        with pytest.raises(ValueError):
            verify_stem_pack(modified)


@pytest.mark.parametrize("mutation", ["pcm", "geometry", "missing", "rename", "score", "manifest"])
def test_all_members_checked_before_selection(bundle, mutation):
    raw, _ = bundle

    def transform(name, data):
        if mutation == "pcm" and name == "stems/hat.wav":
            data = data[:-1] + bytes([data[-1] ^ 1])
        if mutation == "geometry" and name == "stems/hat.wav":
            changed = bytearray(data)
            struct.pack_into("<I", changed, 24, 16000)
            struct.pack_into("<I", changed, 28, 64000)
            data = bytes(changed)
        if mutation == "missing" and name == "stems/hat.wav":
            return None
        if mutation == "rename" and name == "stems/hat.wav":
            name = "stems/../hat.wav"
        if mutation == "score" and name == SCORE_NAME:
            data += b"\n"
        if mutation == "manifest" and name == MANIFEST_NAME:
            data += b"\n"
        return name, data

    modified = rewrite(raw, transform)
    with pytest.raises(ValueError):
        # Selecting lead still checks a tampered hat or score.
        select_stem(modified, "lead")


def test_noncanonical_archive_trailing_data_comment_and_compression(bundle):
    raw, _ = bundle
    with pytest.raises(GenerationError, match="canonical ZIP"):
        verify_stem_pack(raw + b"trailing")
    for mode in ("comment", "compressed", "timestamp"):
        sink = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(raw)) as reader:
            with zipfile.ZipFile(sink, "w") as writer:
                if mode == "comment":
                    writer.comment = b"untrusted"
                for old in reader.infolist():
                    info = _info(old.filename)
                    if mode == "compressed":
                        info.compress_type = zipfile.ZIP_DEFLATED
                    if mode == "timestamp":
                        info.date_time = (2000, 1, 1, 0, 0, 0)
                    writer.writestr(info, reader.read(old))
        with pytest.raises(GenerationError):
            verify_stem_pack(sink.getvalue())


def test_malformed_zip_and_invalid_digest(bundle):
    raw, _ = bundle
    for bad in (b"", b"not a zip", raw[:100], bytearray(raw)):
        with pytest.raises(ValueError):
            verify_stem_pack(bad)
    for digest in ("0" * 64, "A" * 64, "short"):
        with pytest.raises(ValueError):
            verify_stem_pack(raw, expected_sha256=digest)
    with pytest.raises(GenerationError):
        verify_stem_pack(raw, replay=1)
    with pytest.raises(GenerationError):
        select_stem(raw, "../lead")


def test_oversize_score_refused_before_render(monkeypatch):
    long_score = compose_arrangement(ArrangementPlan(bars=32, bpm=80))
    assert long_score.plan.frame(long_score.plan.ticks) > 60 * SAMPLE_RATE

    def forbidden(*args):
        pytest.fail("Oversize score must not synthesize stems")

    monkeypatch.setattr("sonumio.music_stem_pack.render_stems", forbidden)
    with pytest.raises(GenerationError, match="60-second"):
        create_stem_pack(long_score)


def test_fullscale_pcm_rejected(bundle):
    raw, _ = bundle
    _, _, members = unpack(raw)
    member = bytearray(members["lead"])
    struct.pack_into("<h", member, 44, -32768)
    with pytest.raises(GenerationError, match="ceiling"):
        _inspect_stem("lead", bytes(member), (len(member) - 44) // 4)


def test_publication_no_clobber_or_wrong_digest(bundle, tmp_path):
    raw, receipt = bundle
    target = tmp_path / "stems.zip"
    result = publish_stem_pack(target, raw, expected_sha256=sha(raw))
    assert target.read_bytes() == raw
    assert result["output_sha256"] == receipt.archive_sha256
    with pytest.raises(ValueError):
        publish_stem_pack(target, raw)
    with pytest.raises(ValueError):
        publish_stem_pack(target, raw, overwrite=True, expected_sha256="0" * 64)
    with pytest.raises(ValueError):
        publish_stem_pack(target, raw + b"x", overwrite=True)
    assert target.read_bytes() == raw
    assert read_stem_pack(target, expected_sha256=sha(raw)).manifest == receipt.manifest
    assert not list(tmp_path.glob("*.partial"))


def test_cli_workflow(score, tmp_path, capsys):
    score_file = tmp_path / "score.json"
    archive_file = tmp_path / "stems.zip"
    selected = tmp_path / "lead.wav"
    score_file.write_bytes(encode_score(score))
    create = ["music-stems", str(score_file), "--output", str(archive_file),
              "--expected-score-sha256", arrangement_digest(score)]
    assert main(create) == 0
    report = json.loads(capsys.readouterr().out)
    digest = report["archive_sha256"]
    assert digest == sha(archive_file.read_bytes())
    assert main(["music-stems-verify", str(archive_file),
                 "--expected-pack-sha256", digest, "--replay"]) == 0
    assert json.loads(capsys.readouterr().out)["recipe_replay_verified"]
    args = ["music-stem-select", str(archive_file), "--track", "lead",
            "--expected-pack-sha256", digest, "--output", str(selected)]
    assert main(args) == 0
    chosen = json.loads(capsys.readouterr().out)
    assert chosen["output_sha256"] == sha(selected.read_bytes())
    assert main(args) == 2
    assert main(args + ["--overwrite", "--expected-output-sha256", "0" * 64]) == 2
    assert main(["music-stems-verify", str(archive_file),
                 "--expected-pack-sha256", "0" * 64]) == 2
    assert chosen["output_sha256"] == sha(selected.read_bytes())


def test_cli_input_alias_not_replaced(score, tmp_path):
    score_file = tmp_path / "score.zip"
    original = encode_score(score)
    score_file.write_bytes(original)
    assert main(["music-stems", str(score_file), "--output", str(score_file),
                 "--overwrite"]) == 2
    assert score_file.read_bytes() == original
