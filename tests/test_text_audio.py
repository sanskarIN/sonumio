"""Research-only tests: learned text representations, sound generation, rights gates."""
import hashlib
import io
import json
import wave

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from research.text_audio import (
    BYTE_VOCAB, MAX_TOKENS, ByteTextTokenizer, TextEncoder, TextWaveDenoiser,
    synthetic_captioned_dataset, reviewed_captioned_dataset, text_noise_loss,
    DiffusionSchedule, sample_text, train, load_checkpoint, caption_alignment_probe,
)


def test_multilingual_utf8_and_normalization():
    t = ByteTextTokenizer()
    ids = t.encode(['Rain   and   thunder', 'बारिश और गरज', '  rain\t and\n wind  '])
    assert ids.shape == (3, MAX_TOKENS)
    assert ids.dtype == torch.long
    assert int(ids[0, 0]) == 1
    assert int(ids[0, ids[0].nonzero()[-1]]) == 2
    assert torch.equal(ids[0], t.encode(['Rain and thunder'])[0])
    assert torch.equal(ids[2], t.encode(['rain and wind'])[0])
    assert ids[1, 2] > 128  # UTF-8 Devanagari bytes, no ASCII-only dictionary
    assert ids.max() < BYTE_VOCAB
    assert not torch.equal(ids[0], ids[1])


@pytest.mark.parametrize('caption', ['', '   ', 'a' * 129, '\x01rain',
                                   'clone the voice of my friend',
                                   'speech about rain',
                                   'आवाज़ की नकल बारिश'])
def test_bad_text_is_rejected_without_truncation(caption):
    with pytest.raises(ValueError):
        ByteTextTokenizer().encode([caption])


def test_english_hindi_dataset_pairing_and_grouped_seed_holdout():
    ds = synthetic_captioned_dataset((0, 1))
    again = synthetic_captioned_dataset((0, 1))
    heldout = synthetic_captioned_dataset((101,))
    assert ds.examples == 84
    assert ds.clips.shape == (84, 1, 2048)
    assert ds.tokens.shape == (84, MAX_TOKENS)
    assert torch.equal(ds.clips, again.clips)
    assert torch.equal(ds.tokens, again.tokens)
    assert ds.fingerprint == again.fingerprint != heldout.fingerprint
    # English/Hindi captions for the same recording are aligned by construction.
    assert torch.equal(ds.clips[0], ds.clips[1])
    assert not torch.equal(ds.tokens[0], ds.tokens[1])
    assert torch.isfinite(ds.clips).all()
    assert ds.clips.abs().max() <= 1


@pytest.mark.parametrize('seeds', [(), (1, 1), (-1,), (2**32,), (True,)])
def test_synthetic_dataset_rejects_bad_seed_groups(seeds):
    with pytest.raises(ValueError):
        synthetic_captioned_dataset(seeds)


def test_learned_text_features_and_null_conditioning_are_differentiable():
    torch.manual_seed(7)
    model = TextEncoder(channels=16)
    tokens = ByteTextTokenizer().encode(['gentle rain', 'rain and wind'])
    features = model(tokens)
    null_features = model(torch.zeros_like(tokens))
    assert features.shape == null_features.shape == (2, 16)
    assert torch.isfinite(features).all()
    assert torch.equal(null_features[0], null_features[1])
    assert not torch.allclose(features[0], features[1])
    features.sum().backward()
    assert model.tokens.weight.grad is not None
    assert model.tokens.weight.grad[ord('r') + 3].abs().sum() > 0
    assert torch.isfinite(model.tokens.weight.grad).all()
    with pytest.raises(ValueError):
        model(torch.ones(2, 32, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.full((2, MAX_TOKENS), BYTE_VOCAB, dtype=torch.long))


def test_end_to_end_text_conditioned_denoising_gradients_and_dropout():
    torch.manual_seed(8)
    model = TextWaveDenoiser(channels=16)
    schedule = DiffusionSchedule(8)
    tokens = ByteTextTokenizer().encode(['rain', 'समुद्र'])
    waveforms = torch.randn(2, 1, 256) * .1
    loss = text_noise_loss(model, schedule, waveforms, tokens, drop_condition=0.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.text_encoder.tokens.weight.grad.abs().sum() > 0
    zero = text_noise_loss(model, schedule, waveforms, tokens, drop_condition=1.0)
    assert torch.isfinite(zero)
    with pytest.raises(ValueError):
        model(waveforms, torch.tensor([0]), tokens)
    with pytest.raises(ValueError):
        text_noise_loss(model, schedule, waveforms, tokens, drop_condition=float('nan'))


def test_deterministic_text_to_waveform_sampling_and_guidance():
    torch.manual_seed(23)
    model, diffusion = TextWaveDenoiser(channels=16), DiffusionSchedule(8)
    one = sample_text(model, diffusion, 'rain and wind', seed=12, guidance=1.4)
    same = sample_text(model, diffusion, 'rain and wind', seed=12, guidance=1.4)
    other = sample_text(model, diffusion, 'बारिश और हवा', seed=12, guidance=1.4)
    assert one.shape == (2048,)
    assert torch.equal(one, same)
    assert not torch.equal(one, other)
    assert torch.isfinite(one).all()
    assert one.abs().max() <= 1
    with pytest.raises(ValueError):
        sample_text(model, diffusion, 'rain', frames=1024)
    with pytest.raises(ValueError):
        sample_text(model, diffusion, 'rain', guidance=7.0)


def test_smoke_train_checkpoint_safe_load_and_checksum(tmp_path):
    checkpoint = tmp_path / 'private-model'
    report = train(steps=2, seed=19, channels=16, diffusion_steps=8, output=checkpoint)
    assert report['conditioning_mode'] == 'learned-byte-text'
    assert report['training_examples'] == 168
    assert report['heldout_examples'] == 84
    assert report['training_fingerprint'] != report['heldout_fingerprint']
    assert report['weights_published'] is False
    assert np.isfinite(report['heldout_noise_mse'])
    assert report['parameter_count'] > 5000
    model, schedule, manifest = load_checkpoint(checkpoint)
    assert manifest['model_generation'] is None
    assert manifest['weights_publication_authorized'] is False
    assert schedule.steps == 8
    assert sample_text(model, schedule, 'wind', seed=5).shape == (2048,)
    with (checkpoint / 'text_weights.npz').open('ab') as destination:
        destination.write(b'tamper')
    with pytest.raises(ValueError, match='SHA256'):
        load_checkpoint(checkpoint)


def test_corrupt_tensor_name_is_rejected_even_with_valid_hash(tmp_path):
    report = train(steps=1, seed=4, channels=16, diffusion_steps=8, output=tmp_path)
    assert not report['weights_published']
    file = tmp_path / 'text_weights.npz'
    arrays = dict(np.load(file, allow_pickle=False))
    arrays['unknown-tensor'] = np.ones((1,), dtype=np.float32)
    np.savez_compressed(file, **arrays)
    manifest = tmp_path / 'text_manifest.json'
    data = json.loads(manifest.read_text())
    data['weights_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Unexpected text checkpoint tensor names'):
        load_checkpoint(tmp_path)


def _wav_bytes(rate=24000, frequency=420):
    time = np.arange(rate // 2, dtype=np.float64) / rate
    signal = (np.sin(2 * np.pi * frequency * time) * 12000).astype('<i2')
    with io.BytesIO() as out:
        with wave.open(out, 'wb') as recorder:
            recorder.setnchannels(1)
            recorder.setsampwidth(2)
            recorder.setframerate(rate)
            recorder.writeframes(signal.tobytes())
        return out.getvalue()


def _rights_fixture(tmp_path):
    root = tmp_path / 'recordings'
    root.mkdir()
    rows = []
    for i, split in enumerate(('train', 'validation')):
        name = f'{i}.wav'
        wav = _wav_bytes(frequency=420 + i * 100)
        (root / name).write_bytes(wav)
        rows.append({
            'asset_id': f'local-{i}', 'source_url': 'https://example.org/owned-recording',
            'local_path': name, 'sha256': hashlib.sha256(wav).hexdigest(),
            'caption': 'Gentle rain falling' if i == 0 else 'बारिश हो रही है',
            'language': 'en' if i == 0 else 'hi', 'group_id': f'recording-{i}',
            'split': split, 'license': 'CC0-1.0', 'source_kind': 'cc0',
            'rights_reviewed': True, 'creator_permission': False,
            'voice_present': False, 'sample_rate': 24000, 'duration_seconds': 0.5,
        })
    path = tmp_path / 'manifest.jsonl'
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
    return path, root


def test_rights_declaration_loader_preserves_recording_captions(tmp_path):
    manifest, root = _rights_fixture(tmp_path)
    splits, declaration = reviewed_captioned_dataset(manifest, root, clips_per_asset=2)
    assert splits['train'].examples == 2
    assert splits['validation'].examples == 2
    assert splits['train'].tokens.shape == (2, MAX_TOKENS)
    assert splits['train'].fingerprint != splits['validation'].fingerprint
    assert declaration['human_rights_review_remains_required'] is True
    assert declaration['voice_detection_performed'] is False
    info = train(steps=1, seed=3, channels=16, diffusion_steps=8,
                 manifest=manifest, audio_root=root)
    assert info['data_source'] == 'rights-declared-local-wav'
    assert info['manifest_review']['human_rights_review_remains_required'] is True
    assert info['weights_published'] is False


def test_training_requires_manifest_and_root_together(tmp_path):
    with pytest.raises(ValueError, match='together'):
        train(steps=1, manifest=tmp_path / 'missing.jsonl')
    with pytest.raises(ValueError, match='steps'):
        train(steps=0)


def test_caption_alignment_probe_uses_matching_noisy_inputs():
    torch.manual_seed(28)
    model, schedule = TextWaveDenoiser(channels=16), DiffusionSchedule(8)
    audio = torch.randn(3, 1, 256) * 0.1
    tokens = ByteTextTokenizer().encode(['rain', 'hindi sound बारिश', 'storm wind'])
    one = caption_alignment_probe(model, schedule, audio, tokens, seed=77)
    two = caption_alignment_probe(model, schedule, audio, tokens, seed=77)
    assert one == two
    assert one['probe_examples'] == 3
    assert one['paired_minus_shuffled_mse'] == pytest.approx(
        one['paired_noise_mse'] - one['shuffled_noise_mse'])
    assert all(np.isfinite(val) for val in one.values())
    with pytest.raises(ValueError):
        caption_alignment_probe(model, schedule, audio[:1], tokens[:1])

def test_nonstring_prompt_fails_cleanly():
    with pytest.raises(ValueError, match='Prompt must be text'):
        ByteTextTokenizer().encode([None])
