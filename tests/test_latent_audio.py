"""CPU-only tests for independent longer-than-256ms latent research experiment."""
import json
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from research.latent_audio import (
    FRAMES, LATENT_FRAMES, LATENT_CHANNELS, TinyAudioAutoencoder,
    LatentResearchModel, TextLatentDenoiser, reconstruction_loss,
    synthetic_bilingual_waves, latent_noise_loss, DiffusionSchedule,
    ByteTextTokenizer, sample_latent, train, load_checkpoint,
)


def test_autoencoder_shapes_stable_and_gradients():
    torch.manual_seed(7)
    codec = TinyAudioAutoencoder()
    wave = torch.randn(2, 1, FRAMES) * 0.1
    latent = codec.encode(wave)
    assert latent.shape == (2, LATENT_CHANNELS, LATENT_FRAMES)
    reconstructed = codec.decode(latent)
    assert reconstructed.shape == wave.shape
    assert reconstructed.abs().max() <= 1
    loss = reconstruction_loss(wave, reconstructed)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in codec.parameters())
    with pytest.raises(ValueError):
        codec.encode(wave[:, :, :2048])
    with pytest.raises(ValueError):
        codec.decode(latent[:, :, :-1])
    with pytest.raises(ValueError):
        reconstruction_loss(wave, wave[:, :, :2048])


def test_synthetic_bilingual_full_length_is_reproducible_and_seed_separated():
    one = synthetic_bilingual_waves((0,))
    again = synthetic_bilingual_waves((0,))
    heldout = synthetic_bilingual_waves((101,))
    assert one.examples == 42 and one.waves.shape == (42, 1, FRAMES)
    assert one.tokens.shape[0] == 42
    assert torch.equal(one.waves, again.waves)
    assert torch.equal(one.tokens, again.tokens)
    assert one.fingerprint == again.fingerprint != heldout.fingerprint
    assert torch.equal(one.waves[0], one.waves[1])
    assert not torch.equal(one.tokens[0], one.tokens[1])
    assert torch.isfinite(one.waves).all()
    with pytest.raises(ValueError):
        synthetic_bilingual_waves((0, 0))
    with pytest.raises(ValueError):
        synthetic_bilingual_waves((True,))


def test_text_latent_diffusion_backward_and_unconditioned():
    torch.manual_seed(9)
    denoiser = TextLatentDenoiser(16)
    schedule = DiffusionSchedule(8)
    latent = torch.randn(2, LATENT_CHANNELS, LATENT_FRAMES) * 0.1
    tokens = ByteTextTokenizer().encode(['rain and wind', 'बारिश और हवा'])
    loss = latent_noise_loss(denoiser, schedule, latent, tokens, drop_condition=0)
    loss.backward()
    assert torch.isfinite(loss)
    assert denoiser.text_encoder.tokens.weight.grad.abs().sum() > 0
    assert torch.isfinite(latent_noise_loss(denoiser, schedule, latent, tokens,
                                           drop_condition=1))
    with pytest.raises(ValueError):
        latent_noise_loss(denoiser, schedule, latent, tokens, drop_condition=float('nan'))
    with pytest.raises(ValueError):
        denoiser(latent[:, :, :-1], torch.zeros(2, dtype=torch.long), tokens)


def test_seeded_sampling_and_validation():
    torch.manual_seed(3)
    model = LatentResearchModel(16)
    schedule = DiffusionSchedule(8)
    one = sample_latent(model, schedule, 'rain', seed=14, guidance=1)
    same = sample_latent(model, schedule, 'rain', seed=14, guidance=1)
    different = sample_latent(model, schedule, 'rain', seed=15, guidance=1)
    assert one.shape == (FRAMES,) and torch.equal(one, same)
    assert not torch.equal(one, different)
    assert torch.isfinite(one).all() and one.abs().max() <= 1
    with pytest.raises(ValueError):
        sample_latent(model, schedule, 'rain', seed=-1)
    with pytest.raises(ValueError):
        sample_latent(model, schedule, 'rain', guidance=float('inf'))
    with pytest.raises(ValueError):
        sample_latent(model, schedule, '')


def test_two_stage_training_private_checkpoint_and_integrity(tmp_path):
    output = tmp_path / 'private-model'
    report = train(ae_steps=2, diffusion_steps=2, schedule_steps=8,
                   channels=16, seed=12, output=output)
    assert report['model_generation'] is None
    assert report['weights_published'] is False
    assert report['audio_frames'] == FRAMES and report['latent_frames'] == FRAMES // 64
    assert report['train_fingerprint'] != report['validation_fingerprint']
    assert report['train_examples'] == 84
    assert np.isfinite(report['heldout_noise_mse'])
    assert np.isfinite(report['heldout_codec_reconstruction_loss'])
    model, schedule, manifest = load_checkpoint(output)
    assert schedule.steps == 8 and manifest['weights_publication_authorized'] is False
    assert sample_latent(model, schedule, 'जंगल में चिड़िया', seed=11).shape == (FRAMES,)
    with (output / 'latent_weights.npz').open('ab') as fp:
        fp.write(b'tamper')
    with pytest.raises(ValueError, match='SHA256'):
        load_checkpoint(output)


def test_reject_checkpoint_manifest_model_dimensions_before_allocation(tmp_path):
    train(ae_steps=1, diffusion_steps=1, schedule_steps=8,
          seed=5, output=tmp_path)
    f = tmp_path / 'latent_manifest.json'
    data = json.loads(f.read_text(encoding='utf-8'))
    data['channels'] = 1_000_000
    f.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='Unsupported or unsafe'):
        load_checkpoint(tmp_path)


def test_rejects_invalid_training_parameters():
    for values in [dict(ae_steps=0), dict(diffusion_steps=True),
                   dict(schedule_steps=2), dict(channels=64), dict(seed=2**32)]:
        with pytest.raises(ValueError):
            train(**values)
