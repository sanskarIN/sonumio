"""Bounded multi-segment Sonumio v-DDIM and deterministic crossfade tests."""
from __future__ import annotations

import copy
import hashlib
import math

import pytest
torch = pytest.importorskip("torch")
from torch import nn

from research.longform_ddim import (
    FRAMES, MAX_OVERLAP, MAX_SEGMENTS, PROTOCOL,
    overlap_add, sample_overlapped_velocity_ddim,
)
from research.latent_audio import LatentResearchModel
from research.text_audio import TextWaveDenoiser
from research.wave_diffusion import DiffusionSchedule
from research.weight_ema import model_state_sha256


def test_complementary_fade_preserves_constant_level_and_length():
    first = torch.ones(FRAMES, dtype=torch.float32) * 0.25
    waves = (first, first.clone(), first.clone())
    before_hash = hashlib.sha256(first.numpy().tobytes()).hexdigest()
    result, joins = overlap_add(waves, overlap_frames=1024)
    assert result.shape == (FRAMES + 2 * (FRAMES - 1024),)
    assert torch.equal(result, torch.full_like(result, 0.25))
    assert len(joins) == 2
    assert all(row["hard_concatenation_sample_jump"] == 0 for row in joins)
    assert all(row["blend_start_sample_jump"] == 0 for row in joins)
    assert all(row["blend_end_sample_jump"] == 0 for row in joins)
    assert hashlib.sha256(first.numpy().tobytes()).hexdigest() == before_hash


@pytest.mark.parametrize("overlap", [128, 256, 512, 1024, 2048, 4096])
def test_zero_to_one_crossfade_is_monotone_with_smaller_splice_jumps(overlap):
    zero = torch.zeros(FRAMES, dtype=torch.float32)
    one = torch.ones(FRAMES, dtype=torch.float32)
    generated, joins = overlap_add((zero, one), overlap_frames=overlap)
    assert generated.shape == (2 * FRAMES - overlap,)
    start = FRAMES - overlap
    values = generated[start:FRAMES]
    assert torch.all(values >= 0) and torch.all(values <= 1)
    assert torch.all(values[1:] >= values[:-1])
    assert 0 < values[0] < 0.01
    assert 0.99 < values[-1] < 1
    assert generated[start - 1] == 0
    assert generated[FRAMES] == 1
    assert joins[0]["blend_starts_at_frame"] == start
    assert joins[0]["blend_ends_before_frame"] == FRAMES
    assert joins[0]["hard_concatenation_sample_jump"] == 1
    assert joins[0]["blend_start_sample_jump"] < 0.01
    assert joins[0]["blend_end_sample_jump"] < 0.01
    assert joins[0]["mean_overlap_waveform_difference"] == 1
    assert joins[0]["perceptual_click_or_semantic_quality_measured"] is False


@pytest.mark.parametrize("bad", [
    0, 1, -1, 127, 129, 2047, 4097, True, 512.0,
])
def test_invalid_overlap_rejected(bad):
    with pytest.raises(ValueError):
        overlap_add((torch.zeros(FRAMES), torch.ones(FRAMES)),
                    overlap_frames=bad)


@pytest.mark.parametrize("bad", [
    (), (torch.zeros(FRAMES),), [torch.zeros(FRAMES)] * 9,
    [torch.zeros(FRAMES), torch.zeros(FRAMES - 1)],
    [torch.zeros(FRAMES), torch.ones(FRAMES, dtype=torch.float64)],
    [torch.zeros(FRAMES), torch.full((FRAMES,), float("nan"))],
    [torch.zeros(FRAMES), torch.ones(FRAMES) * 1e13],
    [torch.zeros(FRAMES), torch.ones(FRAMES, requires_grad=True)],
])
def test_invalid_segment_shapes_dtype_nonfinite_and_bounds_rejected(bad):
    with pytest.raises(ValueError):
        overlap_add(bad, overlap_frames=512)


def test_raw_network_full_pipeline_reproducible_and_isolated_from_global_rng():
    torch.manual_seed(11)
    raw = TextWaveDenoiser(16)
    raw.train()
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    rng_before = torch.random.get_rng_state().clone()
    model_before = model_state_sha256(raw)
    config = dict(segments=2, overlap_frames=512, evaluations=4, seed=17)
    a, report = sample_overlapped_velocity_ddim(raw, schedule, "rain", **config)
    again, repeated_report = sample_overlapped_velocity_ddim(
        raw, schedule, "rain", **config,
    )
    other, _ = sample_overlapped_velocity_ddim(
        raw, schedule, "rain", **(config | {"seed": 18}),
    )
    assert torch.equal(a, again)
    assert not torch.equal(a, other)
    assert a.shape == (2 * FRAMES - 512,)
    assert torch.isfinite(a).all()
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert model_before == model_state_sha256(raw)
    assert raw.training
    assert report == repeated_report
    assert report["protocol"] == PROTOCOL
    assert report["sample_rate"] == 8000
    assert report["model_evaluations_per_clip"] == 4
    assert report["duration_seconds"] == (2 * FRAMES - 512) / 8000
    assert report["seed"] == 17
    assert [part["sampling_seed"] for part in report["per_clip"]] == [17, 18]
    assert report["output_sha256"] == hashlib.sha256(
        a.numpy().tobytes(),
    ).hexdigest()
    assert report["limiter_or_normalization_applied"] is False
    assert report["model_state_preserved"] is True
    assert report["global_rng_preserved"] is True
    assert report["audio_files_or_model_weights_saved"] is False
    assert report["model_publication_authorized"] is False
    assert report["listening_or_multisegment_semantic_continuity_evaluated"] is False
    assert len(report["joins"]) == 1
    assert 0 <= report["fraction_above_unit_audio_range"] <= 1
    assert math.isfinite(report["unclipped_peak"])


def test_latent_network_bilingual_segment_prompts_and_pretanh_diagnostics():
    torch.manual_seed(12)
    model = LatentResearchModel(16)
    model.denoiser.train()
    model.codec.train()
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    prior_rng = torch.random.get_rng_state().clone()
    before = (
        model_state_sha256(model.denoiser),
        model_state_sha256(model.codec),
    )
    wave, report = sample_overlapped_velocity_ddim(
        model.denoiser, schedule, ("बारिश", "wind"), segments=2,
        overlap_frames=1024, evaluations=3, guidance=1.5, seed=21,
        latent=True, codec=model.codec,
    )
    assert wave.shape == (FRAMES * 2 - 1024,)
    assert torch.isfinite(wave).all() and wave.abs().max() <= 1
    assert report["latent"] is True
    assert report["model_evaluations_per_clip"] == 6
    assert [p["caption"] for p in report["per_clip"]] == ["बारिश", "wind"]
    assert [p["sampling_seed"] for p in report["per_clip"]] == [21, 22]
    for item in report["per_clip"]:
        assert item["prelimiter_signal_source"] == "latent_codec_decoder_before_tanh"
        assert 0 <= item["pre_tanh_saturation_fraction"] <= 1
    assert report["audio_files_or_model_weights_saved"] is False
    assert torch.equal(prior_rng, torch.random.get_rng_state())
    assert before == (
        model_state_sha256(model.denoiser),
        model_state_sha256(model.codec),
    )
    assert model.denoiser.training and model.codec.training


@pytest.mark.parametrize("changes", [
    {"segments": 1}, {"segments": 9}, {"segments": True},
    {"overlap_frames": 0}, {"overlap_frames": 65},
    {"evaluations": 1}, {"evaluations": 9}, {"evaluations": True},
    {"seed": -1}, {"seed": True}, {"seed": 2**32 - 1},
    {"guidance": -1}, {"guidance": 4.5},
    {"guidance": float("nan")}, {"guidance": True},
    {"latent": "yes"}, {"latent": True},
    {"codec": nn.Linear(1, 1)},
])
def test_stitched_sampler_refuses_invalid_configuration(changes):
    model = TextWaveDenoiser(16)
    schedule = DiffusionSchedule(8)
    config = dict(segments=2, overlap_frames=512, evaluations=3, seed=0)
    with pytest.raises(ValueError):
        sample_overlapped_velocity_ddim(
            model, schedule, "rain", **(config | changes),
        )


@pytest.mark.parametrize("captions", [
    (), ("rain",), ("rain", "wind", "storm"),
    ("rain", ""), ("rain", 3), 42, "  ",
])
def test_stitched_sampler_rejects_invalid_caption_layout(captions):
    model = TextWaveDenoiser(16)
    with pytest.raises(ValueError):
        sample_overlapped_velocity_ddim(
            model, DiffusionSchedule(8), captions,
            segments=2, overlap_frames=512, evaluations=3, seed=0,
        )


def test_valid_final_unsigned_seed_can_be_used_by_last_segment():
    model = TextWaveDenoiser(16)
    result, report = sample_overlapped_velocity_ddim(
        model, DiffusionSchedule(8), "wind",
        segments=2, overlap_frames=512, evaluations=2, seed=2**32 - 2,
    )
    assert result.shape == (2 * FRAMES - 512,)
    assert [item["sampling_seed"] for item in report["per_clip"]] == [
        2**32 - 2, 2**32 - 1,
    ]


def test_stitched_sampler_no_files_written_or_model_weights_saved(tmp_path):
    assert not list(tmp_path.iterdir())
    model = TextWaveDenoiser(16)
    result, report = sample_overlapped_velocity_ddim(
        model, DiffusionSchedule(8), "rain",
        segments=2, overlap_frames=1024, evaluations=2, seed=44,
    )
    assert result.ndim == 1
    assert report["audio_files_or_model_weights_saved"] is False
    assert not list(tmp_path.iterdir())
