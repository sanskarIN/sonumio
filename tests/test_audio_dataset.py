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
