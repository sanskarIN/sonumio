"""Private corpus cache round trips, bounded batches and corruption rejection."""
import hashlib
import io
import json
import os
import struct
import wave

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.audio_dataset import ClipProvenance, LoadedSplit
from research.corpus_archive import (
    CorpusArchiveReader, _write_archive, build_archive, main,
)
from research.corpus_archive_format import (
    ArchiveError, ArchiveIndex, ArchiveRow, ShardDescriptor, MAGIC, FOOTER,
    FOOTER_MAGIC, MAX_BATCH_CLIPS, MAX_INDEX_BYTES, MAX_ROWS,
    canonical_json, parse_canonical, encode_shard, decode_shard, digest,
)
from research.wave_diffusion import CLIP_FRAMES, SAMPLE_RATE


def fixture_splits():
    rng = np.random.default_rng(71)
    splits = {}
    for number, (name, count) in enumerate((("train", 5), ("validation", 3), ("test", 2))):
        clips = rng.normal(0, .1, (count, 1, CLIP_FRAMES)).astype(np.float32)
        labels = np.zeros(count, dtype=np.int64)
        captions = ("rain",) * count
        provenance = tuple(
            ClipProvenance(f"asset-{name}-{i}", f"group-{name}-{i}", "en",
                           number * 20 + i + 1, 1)
            for i in range(count)
        )
        splits[name] = LoadedSplit(clips, labels, count, str(number+1)*64,
                                  captions, provenance, name)
    report = {
        "wav_bytes_verified": True, "declaration": {"manifest_fingerprint": "a"*64},
        "split_fingerprints": {s: v.fingerprint for s, v in splits.items()},
        "clip_counts": {s: len(v.labels) for s, v in splits.items()},
    }
    return splits, report


def make_archive(tmp_path, shard_clips=2):
    splits, ingestion = fixture_splits()
    path = tmp_path / "data.scorpus"
    result = _write_archive(splits, ingestion, path, shard_clips=shard_clips)
    return path, splits, result


def unpack_index(path):
    data = path.read_bytes()
    size, magic = FOOTER.unpack(data[-FOOTER.size:])
    assert magic == FOOTER_MAGIC
    offset = len(data) - FOOTER.size - size
    return data[:offset], json.loads(data[offset:offset+size])


def replace_index(path, payload, index):
    encoded = canonical_json(index)
    path.write_bytes(payload + encoded + FOOTER.pack(len(encoded), FOOTER_MAGIC))


@pytest.mark.parametrize("shard_clips", [1, 2, 5, 256])
def test_archive_round_trip_every_split(tmp_path, shard_clips):
    path, splits, result = make_archive(tmp_path, shard_clips)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result.archive_sha256
    assert path.stat().st_size == result.archive_bytes
    with CorpusArchiveReader(path, expected_sha256=result.archive_sha256) as reader:
        assert reader.report()["all_shards_verified"]
        assert reader.report()["split_counts"] == {"train": 5, "validation": 3, "test": 2}
        for name, original in splits.items():
            shards = list(reader.iter_shards(name))
            merged = np.concatenate([x.clips for x in shards])
            np.testing.assert_array_equal(merged, original.clips)
            for i in range(len(original.labels)):
                row, clip = reader.read_clip(name, i)
                assert row.index == i and row.caption == original.captions[i]
                assert row.provenance == original.provenance[i]
                np.testing.assert_array_equal(clip, original.clips[i])
                assert not clip.flags.writeable
                with pytest.raises(ValueError):
                    clip.setflags(write=True)
    with pytest.raises(ArchiveError, match="closed"):
        reader.report()


def test_training_batch_order_and_duplicate_indices(tmp_path, monkeypatch):
    path, splits, _ = make_archive(tmp_path)
    with CorpusArchiveReader(path) as reader:
        original = reader._read_shard
        calls = []
        def track(descriptor):
            calls.append(descriptor.start)
            return original(descriptor)
        monkeypatch.setattr(reader, "_read_shard", track)
        requested = [4, 0, 1, 4, 2, 0]
        batch = reader.read_train_batch(requested)
        assert sorted(calls) == [0, 2, 4]
        np.testing.assert_array_equal(batch.clips, splits["train"].clips[requested])
        np.testing.assert_array_equal(batch.labels, splits["train"].labels[requested])
        assert [row.index for row in batch.rows] == requested
        batch.clips[:] = 0
        _, original_clip = reader.read_clip("train", 4)
        np.testing.assert_array_equal(original_clip, splits["train"].clips[4])


@pytest.mark.parametrize("bad", [[], [True], [-1], [5], [1.5], [0]*(MAX_BATCH_CLIPS+1)])
def test_training_batch_rejects_bad_indices(tmp_path, bad):
    path, _, _ = make_archive(tmp_path)
    with CorpusArchiveReader(path) as reader:
        with pytest.raises(ArchiveError):
            reader.read_train_batch(bad)


def test_no_test_or_validation_rows_in_training_batch(tmp_path):
    path, _, _ = make_archive(tmp_path)
    with CorpusArchiveReader(path) as reader:
        batch = reader.read_train_batch([0, 1, 2, 3, 4])
        assert all(row.asset_id.startswith("asset-train-") for row in batch.rows)
        with pytest.raises(ArchiveError):
            reader.read_clip("unknown", 0)
        with pytest.raises(ArchiveError):
            reader.read_clip("test", 2)


def test_archive_build_deterministic_and_no_clobber(tmp_path):
    splits, ingestion = fixture_splits()
    a, b = tmp_path / "a.scorpus", tmp_path / "b.scorpus"
    first = _write_archive(splits, ingestion, a, shard_clips=2)
    second = _write_archive(splits, ingestion, b, shard_clips=2,
                            expected_sha256=first.archive_sha256)
    assert first.archive_sha256 == second.archive_sha256
    assert a.read_bytes() == b.read_bytes()
    before = a.read_bytes()
    with pytest.raises(ArchiveError, match="already exists"):
        _write_archive(splits, ingestion, a, shard_clips=2)
    assert a.read_bytes() == before
    assert not list(tmp_path.glob(".sonumio-corpus-*"))


def test_output_digest_gate_no_partial_archive(tmp_path):
    splits, ingestion = fixture_splits()
    path = tmp_path / "data.scorpus"
    with pytest.raises(ArchiveError, match="output SHA256"):
        _write_archive(splits, ingestion, path, shard_clips=2, expected_sha256="0"*64)
    assert not path.exists()
    assert not list(tmp_path.glob(".sonumio-corpus-*"))


def test_reader_expected_digest_mismatch(tmp_path):
    path, _, _ = make_archive(tmp_path)
    with pytest.raises(ArchiveError, match="SHA256 mismatch"):
        CorpusArchiveReader(path, expected_sha256="0"*64)


@pytest.mark.parametrize("change", ["magic", "footer", "payload", "truncate"])
def test_corrupt_archive_rejected(tmp_path, change):
    path, _, _ = make_archive(tmp_path)
    data = bytearray(path.read_bytes())
    if change == "magic":
        data[0] ^= 1
    elif change == "footer":
        data[-1] ^= 1
    elif change == "payload":
        data[len(MAGIC)+10] ^= 1
    else:
        del data[-10:]
    path.write_bytes(data)
    with pytest.raises(ArchiveError):
        CorpusArchiveReader(path)


@pytest.mark.parametrize("change", [
    "gap", "overlap", "wrong_count", "oversized_count", "wrong_clock", "wrong_protocol",
    "bad_digest", "bad_start", "reordered", "unknown_key", "wrong_split",
])
def test_index_corruption_rejected_even_with_well_formed_json(tmp_path, change):
    path, _, _ = make_archive(tmp_path)
    payload, index = unpack_index(path)
    if change == "gap":
        index["shards"][0]["offset"] += 1
    elif change == "overlap":
        index["shards"][1]["offset"] -= 1
    elif change == "wrong_count":
        index["split_counts"]["train"] += 1
    elif change == "oversized_count":
        index["split_counts"]["train"] = MAX_ROWS+1
    elif change == "wrong_clock":
        index["sample_rate"] = SAMPLE_RATE+1
    elif change == "wrong_protocol":
        index["protocol"] = "unknown"
    elif change == "bad_digest":
        index["shards"][0]["sha256"] = "x"*64
    elif change == "bad_start":
        index["shards"][0]["start"] = 1
    elif change == "reordered":
        index["shards"][0], index["shards"][1] = index["shards"][1], index["shards"][0]
    elif change == "wrong_split":
        index["shards"][0]["split"] = "other"
    else:
        index["extra"] = "unknown"
    replace_index(path, payload, index)
    with pytest.raises(ArchiveError):
        CorpusArchiveReader(path)


def test_footer_claim_cannot_allocate_huge_index(tmp_path):
    path, _, _ = make_archive(tmp_path)
    data = path.read_bytes()
    path.write_bytes(data[:-FOOTER.size] + FOOTER.pack(MAX_INDEX_BYTES+1, FOOTER_MAGIC))
    with pytest.raises(ArchiveError, match="footer"):
        CorpusArchiveReader(path)


@pytest.mark.parametrize("bad", [
    b'{"a":1,"a":1}', b'{"a":NaN}', b'{ "a":1}', b'{"a":"\\ud800"}',
])
def test_noncanonical_json_is_rejected(bad):
    with pytest.raises(ArchiveError):
        parse_canonical(bad)


def test_read_after_file_edit_is_rejected(tmp_path):
    path, _, _ = make_archive(tmp_path)
    with CorpusArchiveReader(path) as reader:
        original = path.stat()
        with path.open("r+b") as fp:
            fp.seek(len(MAGIC)+12)
            byte = fp.read(1)
            fp.seek(-1, os.SEEK_CUR)
            fp.write(bytes([byte[0] ^ 1]))
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 2_000_000_000))
        with pytest.raises(ArchiveError, match="changed"):
            reader.read_clip("train", 0)


def test_leaf_symlink_is_rejected(tmp_path):
    path, _, _ = make_archive(tmp_path)
    link = tmp_path / "link.scorpus"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ArchiveError, match="ordinary"):
        CorpusArchiveReader(link)


def test_writer_rejects_unverified_or_wrong_ingestion_report(tmp_path):
    splits, ingestion = fixture_splits()
    path = tmp_path / "data.scorpus"
    with pytest.raises(ArchiveError, match="verified ingestion"):
        _write_archive(splits, {}, path, shard_clips=2)
    ingestion["split_fingerprints"]["train"] = "b"*64
    with pytest.raises(ArchiveError, match="differs"):
        _write_archive(splits, ingestion, path, shard_clips=2)
    assert not path.exists()


def test_writer_rejects_exact_crop_cross_split_leakage(tmp_path):
    splits, ingestion = fixture_splits()
    splits["test"].clips[0] = splits["train"].clips[0]
    path = tmp_path / "data.scorpus"
    with pytest.raises(ArchiveError, match="crosses splits"):
        _write_archive(splits, ingestion, path, shard_clips=2)
    assert not path.exists()


def test_writer_rejects_group_cross_split_leakage(tmp_path):
    from dataclasses import replace
    splits, ingestion = fixture_splits()
    test = splits["test"]
    changed = replace(test.provenance[0], group_id=splits["train"].provenance[0].group_id)
    splits["test"] = replace(test, provenance=(changed, *test.provenance[1:]))
    with pytest.raises(ArchiveError, match="group crosses splits"):
        _write_archive(splits, ingestion, tmp_path / "data.scorpus", shard_clips=2)


def test_shard_nonfinite_pcm_rejected_even_after_rehash():
    row = ArchiveRow(0, 0, "rain", "a", "g", "en", 1, 1)
    samples = np.zeros((1, 1, CLIP_FRAMES), dtype=np.float32)
    payload = bytearray(encode_shard((row,), samples))
    payload[-4:] = struct.pack("<f", float("nan"))
    raw = bytes(payload)
    descriptor = ShardDescriptor("train", 0, 1, len(MAGIC), len(raw), digest(raw))
    with pytest.raises(ArchiveError, match="Nonfinite"):
        decode_shard(descriptor, raw)


def test_archive_build_from_actual_local_manifest(tmp_path, capsys):
    root = tmp_path / "recordings"
    root.mkdir()
    rows = []
    for i, split in enumerate(("train", "validation")):
        rate = 8000
        rng = np.random.default_rng(300+i)
        pcm = rng.integers(-10000, 10000, 4096, dtype=np.int16).astype("<i2")
        name = f"clip-{i}.wav"
        with wave.open(str(root / name), "wb") as fp:
            fp.setnchannels(1)
            fp.setsampwidth(2)
            fp.setframerate(rate)
            fp.writeframes(pcm.tobytes())
        rows.append({
            "asset_id": f"a{i}", "source_url": "https://example.org/local-test",
            "local_path": name, "sha256": digest((root/name).read_bytes()),
            "caption": "rain", "language": "en", "group_id": f"g{i}", "split": split,
            "license": "CC0-1.0", "source_kind": "cc0", "rights_reviewed": True,
            "creator_permission": False, "voice_present": False, "sample_rate": rate,
            "duration_seconds": len(pcm)/rate,
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")
    path = tmp_path / "private.scorpus"
    result = build_archive(manifest, root, path, clips_per_asset=1, shard_clips=1)
    assert main(["inspect", str(path), "--expected-sha256", result.archive_sha256]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["split_counts"] == {"train": 1, "validation": 1, "test": 0}
    assert "caption" not in report and "asset_id" not in report
    assert report["rights_independently_verified"] is False
    assert report["contains_private_audio_and_provenance"] is True
