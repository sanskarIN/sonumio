"""Tests of the independent waveform research architecture and paired study."""
import json

import pytest

torch = pytest.importorskip("torch")

from research import heldout_caption_panel as lexical
from research import multiscale_waveform as ms
from research import multiscale_waveform_study as study
from research.latent_audio import FRAMES
from research.text_audio import ByteTextTokenizer
from research.velocity_prediction import velocity_target
from research.wave_diffusion import DiffusionSchedule


def test_pack_is_exact_and_preserves_sample_order():
    audio = torch.arange(2 * FRAMES, dtype=torch.float32).reshape(2, 1, FRAMES)
    packed = ms.pack_waveform(audio)
    assert packed.shape == (2, 8, 1024)
    assert torch.equal(packed[0, :, 1], audio[0, 0, 8:16])
    assert torch.equal(ms.unpack_waveform(packed), audio)


@pytest.mark.parametrize("audio", [
    torch.zeros(1, 2, FRAMES), torch.zeros(1, 1, FRAMES - 1),
    torch.zeros(0, 1, FRAMES), torch.zeros(17, 1, FRAMES),
    torch.zeros(1, 1, FRAMES, dtype=torch.float64),
    torch.full((1, 1, FRAMES), float("nan")),
])
def test_reject_invalid_waveforms(audio):
    with pytest.raises(ValueError, match="waveform batch"):
        ms.pack_waveform(audio)


@pytest.mark.parametrize("packed", [
    torch.zeros(1, 7, 1024), torch.zeros(1, 8, 1023),
    torch.full((1, 8, 1024), float("inf")),
])
def test_reject_invalid_patches(packed):
    with pytest.raises(ValueError, match="waveform patches"):
        ms.unpack_waveform(packed)


def test_model_shape_gradients_and_null_condition():
    model = ms.MultiscaleWaveDenoiser()
    audio = torch.randn(2, 1, FRAMES)
    tokens = ByteTextTokenizer().encode(["rain", "wind"])
    tokens[1] = 0
    output = model(audio, torch.tensor([0, 7]), tokens)
    assert output.shape == audio.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_prompt_and_timestep_can_change_prediction():
    model = ms.MultiscaleWaveDenoiser().eval()
    audio = torch.randn(1, 1, FRAMES)
    tokens = ByteTextTokenizer().encode(["rain", "wind"])
    with torch.no_grad():
        rain = model(audio, torch.tensor([3]), tokens[:1])
        wind = model(audio, torch.tensor([3]), tokens[1:])
        other_time = model(audio, torch.tensor([6]), tokens[:1])
    assert not torch.equal(rain, wind)
    assert not torch.equal(rain, other_time)


def test_invalid_timestep_and_caption_shapes():
    model = ms.MultiscaleWaveDenoiser()
    audio = torch.zeros(1, 1, FRAMES)
    tokens = ByteTextTokenizer().encode(["rain"])
    for time in (torch.tensor([-1]), torch.tensor([256]), torch.tensor([1.0]), torch.tensor([1, 2])):
        with pytest.raises(ValueError, match="time or caption"):
            model(audio, time, tokens)
    with pytest.raises(ValueError, match="time or caption"):
        model(audio, torch.tensor([1]), tokens[:, :-1])


def test_spectral_loss_is_small_for_exact_velocity_and_backpropagates():
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    clean = torch.randn(2, 1, FRAMES) * 0.1
    noise = torch.randn_like(clean)
    timestep = torch.tensor([3, 7])
    noisy = schedule.corrupt(clean, timestep, noise)
    target = velocity_target(clean, noise, timestep, schedule)
    oracle = ms.spectral_clean_loss(noisy, target, clean, timestep, schedule)
    estimate = (target + 0.2 * torch.randn_like(target)).requires_grad_()
    loss = ms.spectral_clean_loss(noisy, estimate, clean, timestep, schedule)
    assert oracle < 1e-5
    assert loss > oracle
    loss.backward()
    assert torch.isfinite(estimate.grad).all()
    assert estimate.grad.abs().sum() > 0


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"train_steps": 0}, {"train_steps": 257},
    {"evaluations": 9}, {"sampling_seeds": (301, 301)}, {"spectral_weight": float("nan")},
    {"spectral_weight": 0}, {"spectral_weight": True},
])
def test_invalid_study_configuration(kwargs):
    with pytest.raises(ValueError, match="bounded multiscale"):
        study.run_study(**kwargs)


def test_heldout_caption_tokens_are_distinct_from_training():
    lexical.validate_caption_panel()
    assert all(len(values) == 6 for values in lexical.PARAPHRASES.values())


def test_actual_paired_training_and_generation():
    original_rng = torch.random.get_rng_state().clone()
    report = study.run_study(train_steps=1, evaluations=2)
    assert torch.equal(original_rng, torch.random.get_rng_state())
    assert report["same_training_input_trace"]
    assert report["same_multiscale_initialization"]
    assert not report["equal_parameter_comparison"]
    assert not report["model_readiness_established"]
    assert not report["weights_or_audio_saved"]
    assert report["unique_training_recordings"] == report["unique_validation_recordings"] == 42
    arms = report["arms"]
    assert set(arms) == set(study.ARMS)
    assert len({arm["training"]["input_trace_sha256"] for arm in arms.values()}) == 1
    assert arms["multiscale_velocity"]["training"]["initial_state_sha256"] == arms["multiscale_spectral"]["training"]["initial_state_sha256"]
    assert arms["multiscale_velocity"]["training"]["mean_spectral_auxiliary"] == 0
    assert arms["multiscale_spectral"]["training"]["mean_spectral_auxiliary"] > 0
    for arm in arms.values():
        assert arm["heldout"]["caption_rows_per_timestep"] == 84
        assert arm["heldout"]["unique_recordings"] == 42
        assert set(arm["heldout"]["by_timestep"]) == {"0", "4", "7"}
        if report["reference_probe"]["gate"]["passed"]:
            assert len(arm["generated"]["cases"]) == 12
            assert arm["generated"]["null_checks"] == [{"seed": 301, "caption_invariant": True}]
            panel = arm["heldout_caption_panel"]
            assert panel["zero_guidance_caption_invariant"]
            assert panel["alternative_templates_token_disjoint_from_training"]
            assert not panel["general_open_vocabulary_quality_established"]
            for group in panel["by_language"].values():
                assert len(group["cases"]) == 6
                assert group["summary"]["reference_assignment"]["permutations"] == 720
                assert group["summary"]["alternative_assignment"]["permutations"] == 720
        else:
            assert arm["generated"] is None
            assert arm["heldout_caption_panel"] is None
    json.dumps(report, allow_nan=False)
