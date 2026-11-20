"""Research reconstruction baselines: prevent silent evaluation regression."""
from __future__ import annotations

import math
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from research import latent_quality
from research.latent_audio import FRAMES, LatentResearchModel, DiffusionSchedule, train
from research.latent_quality import (
    waveform_metrics, evaluate_codec, generation_health, _moving_average,
)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x.clone()


class Silent(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def reference():
    frames = np.arange(FRAMES, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 240 * frames / 8000)).astype(np.float32)


def test_waveform_metrics_exact_identity_and_silence_controls():
    original = reference()
    identical = waveform_metrics(original, original)
    empty = waveform_metrics(original, np.zeros_like(original))
    averaged = waveform_metrics(original, _moving_average(original))
    for key in ('mae', 'rmse', 'log_spectral_l1', 'spectral_convergence'):
        assert identical[key] == pytest.approx(0, abs=1e-12)
        assert empty[key] > identical[key]
    assert identical['snr_db'] > 100
    assert empty['snr_db'] == pytest.approx(0, abs=1e-6)
    assert averaged['mae'] > identical['mae']
    assert all(math.isfinite(value) for value in averaged.values())


@pytest.mark.parametrize('bad', [
    np.zeros(FRAMES - 1, dtype=np.float32),
    np.ones((1, FRAMES), dtype=np.float32),
    np.full(FRAMES, float('nan'), dtype=np.float32),
    np.ones(FRAMES, dtype=np.int16),
    np.full(FRAMES, 1.1, dtype=np.float32),
])
def test_waveform_metrics_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        waveform_metrics(reference(), bad)


def test_silent_reference_cannot_hide_bad_relative_metrics():
    with pytest.raises(ValueError, match='Silence-only reference'):
        waveform_metrics(np.zeros(FRAMES, dtype=np.float32),
                         np.zeros(FRAMES, dtype=np.float32))


def test_objective_codec_evaluation_checks_every_unique_waveform_and_reproducibility():
    kwargs = dict(seeds=(101, 102), combinations=((0,), (0, 1)),
                  bootstrap_iterations=32, bootstrap_seed=16)
    model = Identity()
    model.train()
    rng_before = torch.random.get_rng_state().clone()
    report = evaluate_codec(model, **kwargs)
    again = evaluate_codec(model, **kwargs)
    assert model.training
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert report == again
    assert report['recordings_checked'] == 4
    assert report['independent_recordings'] == 4
    assert report['exact_training_waveform_overlaps'] == 0
    assert report['evaluation_sha256'] != ''
    assert report['by_source_count']['1']['codec']['mae'] == 0
    assert report['by_source_count']['2']['codec']['mae'] == 0
    assert report['overall']['codec']['log_spectral_l1'] == pytest.approx(0.0)
    assert report['overall']['silence']['mae'] > 0
    assert report['exploratory_intervals']['available']
    for baseline in ('silence', 'moving_average'):
        interval = report['exploratory_intervals']['comparisons'][baseline]['mae']
        assert interval['codec_minus_baseline'] < 0
        assert len(interval['exploratory_cluster_bootstrap_95pct']) == 2
    assert report['same_waveform_for_bilingual_captions_counted_once']


def test_codec_benchmark_reports_failed_untrained_silence_against_silence_baseline():
    report = evaluate_codec(Silent(), seeds=(101,),
                            combinations=((0,), (0, 1)),
                            bootstrap_iterations=16)
    assert report['overall']['codec']['mae'] == pytest.approx(
        report['overall']['silence']['mae'])
    assert report['overall']['codec']['snr_db'] == pytest.approx(0, abs=1e-6)


def test_reject_training_seed_as_evaluation_and_invalid_controls():
    for opts in (
        dict(seeds=(0,)), dict(seeds=(101, 101)), dict(seeds=(True,)),
        dict(combinations=((10,),)), dict(combinations=((0,), (0,))),
        dict(bootstrap_iterations=0), dict(bootstrap_iterations=2001),
        dict(bootstrap_seed=-1),
    ):
        with pytest.raises(ValueError):
            evaluate_codec(Identity(), combinations=((0,),), **{
                k: v for k, v in opts.items() if k != 'combinations'
            }) if 'combinations' not in opts else evaluate_codec(Identity(), **opts)


def test_detect_exact_train_eval_leakage_even_if_seed_ids_differ(monkeypatch):
    real = latent_quality.render_seeded_waveform

    def same_recording(members, seed):
        return real(members, 0)

    monkeypatch.setattr(latent_quality, 'render_seeded_waveform', same_recording)
    with pytest.raises(ValueError, match='Exact training/evaluation audio overlap'):
        evaluate_codec(Identity(), seeds=(101,), combinations=((0,),))


def test_generated_waveform_health_only_not_audio_quality():
    torch.manual_seed(4)
    model = LatentResearchModel(16)
    schedule = DiffusionSchedule(8)
    result = generation_health(model, schedule, prompts=('rain', 'बारिश'), seed=8)
    again = generation_health(model, schedule, prompts=('rain', 'बारिश'), seed=8)
    assert result == again
    assert result['clip_seconds'] == FRAMES / 8000
    assert len(result['samples']) == 2
    assert all(case['frames'] == FRAMES and 0 <= case['rms'] <= 1
               and 0 <= case['peak'] <= 1 and 0 <= case['fraction_at_clipping_boundary'] <= 1
               for case in result['samples'])
    assert result['samples'][0]['sample_sha256'] != ''
    assert 'not perceptual' in result['warning']
    with pytest.raises(ValueError):
        generation_health(model, schedule, prompts=())


def test_training_report_covers_all_unique_validation_recordings():
    report = train(ae_steps=1, diffusion_steps=1, schedule_steps=8, seed=6)
    assert report['validation_examples'] == 84
    assert report['heldout_unique_recordings'] == 42
    assert report['heldout_caption_noise_examples'] == 84
    assert math.isfinite(report['heldout_codec_reconstruction_loss'])
    assert math.isfinite(report['heldout_noise_mse'])
    assert report['weights_published'] is False
