"""Multi-sound prompt research is intentionally finite vocabulary and tiny clips."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from research.multi_condition import parse_sound_prompt, mixed_dataset, sample_mixed
from research.wave_diffusion import TinyWaveDenoiser, DiffusionSchedule, train, load_checkpoint, training_step


def test_parser_supports_hindi_english_and_two_class_soundscapes():
    a = parse_sound_prompt('rain and thunder')
    assert a.tolist() == [1, 1, 0, 0, 0, 0]
    assert torch.equal(parse_sound_prompt('बारिश और गरज'), a)
    assert parse_sound_prompt('ocean waves and soft wind').tolist() == [0, 0, 1, 0, 1, 0]


@pytest.mark.parametrize('prompt', [
    '', 'unknown noisy city street', 'rain thunder wind', 'voice clone rain',
    'नैरेशन बारिश', ' '.join(['a'] * 801),
])
def test_parser_refuses_unsupported_requests(prompt):
    with pytest.raises(ValueError):
        parse_sound_prompt(prompt)


def test_deterministic_mixed_dataset_and_nonleaking_seeds():
    audio, labels, fingerprint = mixed_dataset((1, 2))
    other_audio, other_labels, same = mixed_dataset((1, 2))
    _, _, heldout = mixed_dataset((900,))
    assert audio.shape == (42, 1, 2048)
    assert labels.shape == (42, 6)
    assert labels.sum(dim=1).tolist().count(2) == 30
    assert torch.equal(audio, other_audio) and torch.equal(labels, other_labels)
    assert fingerprint == same and fingerprint != heldout
    assert torch.isfinite(audio).all()


def test_multi_conditional_denoiser_training_gradients_and_dropout():
    torch.manual_seed(4)
    model, schedule = TinyWaveDenoiser(channels=16), DiffusionSchedule(8)
    clips = torch.randn(2, 1, 256)
    conditions = torch.tensor([[1., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 1.]])
    loss = training_step(model, schedule, clips, conditions, drop_condition=1)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()
    with pytest.raises(ValueError):
        model(clips, torch.tensor([1, 1]), torch.full((2, 6), float('nan')))
    with pytest.raises(ValueError):
        model(clips, torch.tensor([1, 1]), torch.ones((2, 6)))


def test_mixed_sampling_is_repeatable_with_cfg():
    torch.manual_seed(17)
    model, schedule = TinyWaveDenoiser(channels=16), DiffusionSchedule(8)
    a = sample_mixed(schedule, model, 'rain and thunder', frames=256, seed=42, guidance=1.5)
    b = sample_mixed(schedule, model, 'rain and thunder', frames=256, seed=42, guidance=1.5)
    c = sample_mixed(schedule, model, 'rain and thunder', frames=256, seed=41, guidance=1.5)
    assert a.shape == (256,) and torch.isfinite(a).all()
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.abs().max() <= 1
    with pytest.raises(ValueError):
        sample_mixed(schedule, model, 'rain', guidance=50)


def test_mixed_training_checkpoint_is_explicitly_unpublished(tmp_path):
    report = train(steps=2, channels=16, diffusion_steps=8, mixed_synthetic=True, output=tmp_path)
    assert report['conditioning_mode'] == 'multi-label'
    assert report['data_source'] == 'sonumio-mixed-synthetic-dsp'
    assert report['training_examples'] == 84
    assert np.isfinite(report['heldout_noise_mse'])
    assert not report['weights_published']
    loaded, _, info = load_checkpoint(tmp_path)
    assert info['training_report']['conditioning_mode'] == 'multi-label'
    assert loaded.classes == 6


def test_mixed_training_cannot_relabel_reviewed_wav_assets(tmp_path):
    with pytest.raises(ValueError, match='cannot be combined'):
        train(steps=1, mixed_synthetic=True, manifest=tmp_path/'m.jsonl', audio_root=tmp_path)
