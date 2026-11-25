"""Local WAV ingestion: audio bytes, provenance and split-integrity tests."""
import hashlib
import io
import json
from pathlib import Path
import wave

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.audio_dataset import caption_category, load_reviewed_splits
from research.dataset_manifest import ManifestError
from research.wave_diffusion import train


def wav_pcm(*, seconds=0.3, rate=24000, channels=1):
    frames = int(seconds * rate)
    t = np.arange(frames) / rate
    signal = (np.sin(2 * np.pi * 340 * t) * 9000).astype('<i2')
    audio = np.column_stack((signal, -signal // 2)).astype('<i2') if channels == 2 else signal
    with io.BytesIO() as output:
        with wave.open(output, 'wb') as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(audio.tobytes())
        return output.getvalue()


def make_dataset(tmp_path, *, train_bytes=None, validation_bytes=None, change=None):
    root = tmp_path / 'recordings'
    root.mkdir()
    rows = []
    for idx, (split, prompt, content) in enumerate((('train', 'rain', train_bytes or wav_pcm()),
                                                  ('validation', 'हवा', validation_bytes or wav_pcm(rate=16000)))):
        filename = f'clip-{idx}.wav'
        (root / filename).write_bytes(content)
        with wave.open(io.BytesIO(content), 'rb') as wav:
            rate, duration = wav.getframerate(), wav.getnframes() / wav.getframerate()
        rows.append({
            'asset_id': f'a{idx}', 'source_url': 'https://example.org/owned',
            'local_path': filename, 'sha256': hashlib.sha256(content).hexdigest(),
            'caption': prompt, 'language': 'hi' if idx else 'en',
            'group_id': f'session-{idx}', 'split': split, 'license': 'CC0-1.0',
            'source_kind': 'cc0', 'rights_reviewed': True, 'creator_permission': False,
            'voice_present': False, 'sample_rate': rate, 'duration_seconds': duration,
        })
    if change:
        rows[0].update(change)
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
    return manifest, root


def test_load_verified_wav_and_deterministic_crops(tmp_path):
    manifest, root = make_dataset(tmp_path)
    split, info = load_reviewed_splits(manifest, root, clips_per_asset=3)
    other, again = load_reviewed_splits(manifest, root, clips_per_asset=3)
    assert split['train'].clips.shape == (3, 1, 2048)
    assert split['validation'].labels.tolist() == [2] * 3
    assert split['train'].labels.tolist() == [0] * 3
    assert np.array_equal(split['train'].clips, other['train'].clips)
    assert np.isfinite(split['validation'].clips).all()
    assert info['wav_bytes_verified'] and not info['voice_detection_performed']
    assert info['split_fingerprints'] == again['split_fingerprints']


def test_stereo_pcm_and_mono_mix(tmp_path):
    manifest, root = make_dataset(tmp_path, train_bytes=wav_pcm(channels=2))
    split, _ = load_reviewed_splits(manifest, root)
    assert 0 < np.max(np.abs(split['train'].clips)) < 1


@pytest.mark.parametrize('edit', [
    {'local_path': '../secret.wav'}, {'local_path': '/etc/secret.wav'},
    {'local_path': 'C:/secret.wav'}, {'local_path': 'missing.wav'},
    {'local_path': 'clip-0.mp3'}, {'sha256': 'b' * 64},
    {'caption': 'rain and thunder'}, {'caption': 'sound of a car'},
    {'sample_rate': 8000}, {'duration_seconds': 1.0},
    {'rights_reviewed': False}, {'voice_present': True},
])
def test_reject_untrusted_manifest_and_audio(tmp_path, edit):
    manifest, root = make_dataset(tmp_path, change=edit)
    with pytest.raises(ManifestError):
        load_reviewed_splits(manifest, root)


def test_reject_escape_via_symlink(tmp_path):
    manifest, root = make_dataset(tmp_path)
    elsewhere = tmp_path / 'outside.wav'
    elsewhere.write_bytes((root / 'clip-0.wav').read_bytes())
    (root / 'clip-0.wav').unlink()
    try:
        (root / 'clip-0.wav').symlink_to(elsewhere)
    except OSError:
        pytest.skip('Symlinks disabled on this operating system')
    with pytest.raises(ManifestError, match='outside'):
        load_reviewed_splits(manifest, root)


def test_reject_dangerous_or_unsupported_audio(tmp_path):
    manifest, root = make_dataset(tmp_path)
    (root / 'clip-0.wav').write_bytes(b'not a wav')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]['sha256'] = hashlib.sha256(b'not a wav').hexdigest()
    manifest.write_text('\n'.join(map(json.dumps, rows)) + '\n')
    with pytest.raises(ManifestError):
        load_reviewed_splits(manifest, root)


def test_validation_required(tmp_path):
    manifest, root = make_dataset(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[1]['split'] = 'test'
    manifest.write_text('\n'.join(map(json.dumps, rows)) + '\n')
    with pytest.raises(ManifestError, match='validation'):
        load_reviewed_splits(manifest, root)


def test_caption_category_is_not_unbounded_text_understanding():
    assert caption_category('rain') == 0
    assert caption_category('हवा') == 2
    assert caption_category('gentle ocean waves') == 4
    with pytest.raises(ManifestError):
        caption_category('rain with ocean waves')


def test_training_with_reviewed_files_isolation_and_checkpoint(tmp_path):
    manifest, root = make_dataset(tmp_path)
    output = tmp_path / 'private-checkpoint'
    report = train(steps=2, channels=16, diffusion_steps=8,
                   manifest=manifest, audio_root=root, output=output)
    assert report['data_source'] == 'rights-declared-local-wav'
    assert report['data_ingestion']['asset_counts'] == {'train': 1, 'validation': 1, 'test': 0}
    assert report['training_examples'] == 2 and report['heldout_examples'] == 2
    assert report['weights_published'] is False
    assert report['training_fingerprint'] != report['heldout_fingerprint']
    assert (output / 'weights.npz').exists()


def test_training_requires_both_manifest_and_root(tmp_path):
    with pytest.raises(ValueError, match='together'):
        train(steps=1, manifest=tmp_path / 'manifest.jsonl')


def test_caption_changes_affect_fingerprint_without_changing_audio(tmp_path):
    manifest, root = make_dataset(tmp_path)
    first, report = load_reviewed_splits(manifest, root)
    assert first["train"].captions == ("rain", "rain")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["caption"] = "gentle rain"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    revised, new_report = load_reviewed_splits(manifest, root)
    assert np.array_equal(first["train"].clips, revised["train"].clips)
    assert revised["train"].labels.tolist() == first["train"].labels.tolist()
    assert revised["train"].captions == ("gentle rain", "gentle rain")
    assert first["train"].fingerprint != revised["train"].fingerprint
    assert report["declaration"]["manifest_fingerprint"] != new_report["declaration"]["manifest_fingerprint"]


def test_ingestion_uses_single_snapshot_even_if_manifest_changes(tmp_path, monkeypatch):
    from research import audio_dataset
    manifest, root = make_dataset(tmp_path)
    original = audio_dataset.load_manifest_snapshot
    calls = []

    def change_manifest_after_snapshot(path):
        result = original(path)
        calls.append(True)
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        rows[0]["caption"] = "thunder instead"
        manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(audio_dataset, "load_manifest_snapshot", change_manifest_after_snapshot)
    loaded, _ = load_reviewed_splits(manifest, root)
    assert len(calls) == 1
    assert loaded["train"].captions == ("rain", "rain")
    assert loaded["train"].labels.tolist() == [0, 0]


def test_reject_identical_decoded_audio_with_distinct_wav_file_hashes(tmp_path):
    """Metadata-only RIFF changes must not leak the same audio across splits."""
    base = wav_pcm(rate=24000)
    modified_header = base + b"JUNK:trailing-metadata"
    assert hashlib.sha256(base).hexdigest() != hashlib.sha256(modified_header).hexdigest()
    manifest, root = make_dataset(
        tmp_path, train_bytes=base, validation_bytes=modified_header,
    )
    # The declared file hashes differ and the recording groups are distinct.
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["sha256"] != rows[1]["sha256"]
    assert rows[0]["group_id"] != rows[1]["group_id"]
    with pytest.raises(ManifestError, match="line 2: decoded PCM duplicates line 1"):
        load_reviewed_splits(manifest, root)


def test_reject_duplicate_decoded_audio_in_same_split(tmp_path):
    base = wav_pcm(rate=24000)
    manifest, root = make_dataset(
        tmp_path, train_bytes=base, validation_bytes=wav_pcm(rate=16000),
    )
    duplicate = base + b"RIFF-metadata-only"
    (root / "clip-2.wav").write_bytes(duplicate)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    extra = dict(rows[0])
    extra.update(
        asset_id="a2", local_path="clip-2.wav",
        sha256=hashlib.sha256(duplicate).hexdigest(), group_id="session-2",
    )
    rows.append(extra)
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="line 3: decoded PCM duplicates line 1"):
        load_reviewed_splits(manifest, root)


def test_actual_decoded_audio_differences_are_allowed(tmp_path):
    manifest, root = make_dataset(tmp_path)
    loaded, report = load_reviewed_splits(manifest, root)
    assert len(loaded["train"].labels) == 2
    assert len(loaded["validation"].labels) == 2
    assert report["decoded_audio_distinct_recordings"] == 2
    assert report["decoded_audio_duplicate_gate"] == "exact mono PCM at identical sample rate"


@pytest.mark.parametrize("seconds", [0.1, 0.2, 0.255])
def test_short_recordings_cannot_be_silently_zero_padded(tmp_path, seconds):
    manifest, root = make_dataset(tmp_path, train_bytes=wav_pcm(seconds=seconds))
    with pytest.raises(ManifestError, match="shorter than one complete training clip"):
        load_reviewed_splits(manifest, root)


def test_minimum_256ms_clip_is_supported_without_artificial_padding(tmp_path):
    manifest, root = make_dataset(tmp_path, train_bytes=wav_pcm(seconds=0.256))
    loaded, report = load_reviewed_splits(manifest, root, clips_per_asset=3)
    assert loaded["train"].clips.shape == (3, 1, 2048)
    assert report["complete_clips_without_zero_padding"] is True
    # Every crop must retain audible content through the last 128 frames.
    assert np.all(np.max(np.abs(loaded["train"].clips[:, :, -128:]), axis=(1, 2)) > 0.005)
    assert loaded["validation"].clips.shape == (3, 1, 2048)


def test_same_fingerprint_and_wav_samples_for_existing_complete_clips(tmp_path):
    manifest, root = make_dataset(tmp_path)
    a, ar = load_reviewed_splits(manifest, root, clips_per_asset=2)
    b, br = load_reviewed_splits(manifest, root, clips_per_asset=2)
    for split in ("train", "validation"):
        np.testing.assert_array_equal(a[split].clips, b[split].clips)
        assert a[split].fingerprint == b[split].fingerprint
    assert ar["split_fingerprints"] == br["split_fingerprints"]


def test_corpus_pcm_preflight_rejects_before_opening_any_audio(tmp_path, monkeypatch):
    from research import audio_dataset
    manifest, root = make_dataset(tmp_path)
    monkeypatch.setattr(audio_dataset, "MAX_CORPUS_PCM_BYTES", 8192)
    def unexpected(*args, **kwargs):
        pytest.fail("audio was opened despite failed corpus budget")
    monkeypatch.setattr(audio_dataset, "read_local_snapshot", unexpected)
    with pytest.raises(ManifestError, match="bounded PCM allocation"):
        load_reviewed_splits(manifest, root, clips_per_asset=2)


def test_preallocated_model_inputs_match_declared_byte_plan(tmp_path):
    manifest, root = make_dataset(tmp_path)
    loaded, report = load_reviewed_splits(manifest, root, clips_per_asset=3)
    assert report["planned_model_input_pcm_bytes"] == sum(
        split.clips.nbytes for split in loaded.values()
    )
    assert report["planned_model_input_pcm_bytes"] == 6*2048*4
    assert report["local_file_snapshot_checks"] is True
    assert loaded["test"].clips.shape == (0, 1, 2048)
    assert loaded["train"].clips.flags.c_contiguous


def test_reject_inside_root_leaf_symlink(tmp_path):
    manifest, root = make_dataset(tmp_path)
    original = root / "clip-0.wav"
    other = root / "actual.wav"
    original.rename(other)
    try:
        original.symlink_to(other)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ManifestError, match="ordinary"):
        load_reviewed_splits(manifest, root)


def test_wav_snapshot_rejects_special_file(tmp_path):
    import os
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO unavailable")
    manifest, root = make_dataset(tmp_path)
    target = root / "clip-0.wav"
    target.unlink()
    os.mkfifo(target)
    with pytest.raises(ManifestError, match="ordinary"):
        load_reviewed_splits(manifest, root)
