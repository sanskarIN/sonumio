"""Endpoint search against analytic, random and adversarial sources."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.loop_analysis import (
    LoopSearchPlan, LoopScoreWeights, search_loop, measure_loop_seam,
)


def periodic():
    t = np.arange(8192)
    x = .23*np.sin(2*np.pi*t/64) + .11*np.sin(4*np.pi*t/64+.4)
    return make_buffer(x.astype(np.float32), 24000)


def test_search_improves_misaligned_nominal_and_matches_direct_score():
    audio = periodic()
    plan = LoopSearchPlan(100, 1003, start_radius=0, stop_radius=32,
                          context_frames=64, stride=8, results=8)
    nominal = measure_loop_seam(audio, 100, 1003, context_frames=64)
    found = search_loop(audio, plan)
    assert found.best.score < nominal.score
    assert found.best.start_frame == 100
    assert 971 <= found.best.stop_frame <= 1035
    assert found.best.period_frames % 64 == 0
    direct = measure_loop_seam(audio, found.best.start_frame, found.best.stop_frame,
                               context_frames=64)
    assert direct.metrics == found.best.metrics
    assert direct.score == found.best.score
    assert not found.as_dict()["perceptual_seamlessness_established"]
    assert found.coarse_pairs > 0 and found.refined_pairs > 0


def test_refinement_matches_exhaustive_near_best():
    audio = periodic()
    coarse = search_loop(audio, LoopSearchPlan(128, 1039, start_radius=0,
        stop_radius=31, context_frames=64, stride=8, refine_count=0))
    refined = search_loop(audio, LoopSearchPlan(128, 1039, start_radius=0,
        stop_radius=31, context_frames=64, stride=8, refine_count=4))
    full = search_loop(audio, LoopSearchPlan(128, 1039, start_radius=0,
        stop_radius=31, context_frames=64, stride=1, refine_count=0))
    assert refined.best.score <= coarse.best.score
    assert refined.best.score == full.best.score
    assert refined.best.stop_frame == full.best.stop_frame


def test_antiphase_stereo_preserves_shape_matching():
    mono = periodic()
    stereo = make_buffer(np.concatenate((mono.samples, -mono.samples), axis=0), 24000)
    plan = LoopSearchPlan(128, 1030, start_radius=8, stop_radius=16,
                          context_frames=64, stride=4)
    a = search_loop(mono, plan)
    b = search_loop(stereo, plan)
    assert (a.best.start_frame, a.best.stop_frame) == (b.best.start_frame, b.best.stop_frame)
    assert a.best.score == pytest.approx(b.best.score, abs=1e-12)
    assert a.best.metrics.shape_correlation == pytest.approx(b.best.metrics.shape_correlation)


def test_silence_tie_prefers_nominal_without_nan():
    audio = make_buffer(np.zeros(2048, dtype=np.float32), 24000)
    plan = LoopSearchPlan(128, 1024, start_radius=8, stop_radius=8, stride=4)
    found = search_loop(audio, plan)
    assert (found.best.start_frame, found.best.stop_frame) == (128, 1024)
    assert found.best.metrics.silence_penalty == 1.
    assert found.best.metrics.shape_correlation == 1.
    assert np.isfinite(found.best.score)


def test_period_constraints_include_crossfade_overlap():
    audio = periodic()
    plan = LoopSearchPlan(100, 1000, start_radius=10, stop_radius=10,
        blend_frames=64, context_frames=64, stride=4,
        minimum_period=831, maximum_period=832)
    found = search_loop(audio, plan)
    assert all(831 <= c.period_frames <= 832 for c in found.candidates)
    for c in found.candidates:
        assert c.stop_frame-c.start_frame-c.period_frames == 64


def test_edges_and_no_search():
    audio = periodic()
    plan = LoopSearchPlan(0, audio.frames, start_radius=0, stop_radius=0,
                           context_frames=64)
    result = search_loop(audio, plan)
    assert result.coarse_pairs == 1 and result.refined_pairs == 0
    assert len(result.candidates) == 1
    assert result.best.start_frame == 0
    assert result.best.stop_frame == audio.frames


def test_source_immutability_and_stable_rank():
    rng = np.random.default_rng(42)
    audio = make_buffer(rng.normal(size=(2, 4096)).astype(np.float32)*.1, 24000)
    before = audio.samples.tobytes()
    plan = LoopSearchPlan(400, 3400, start_radius=20, stop_radius=20, stride=5)
    a = search_loop(audio, plan)
    b = search_loop(audio, plan)
    assert a.as_dict() == b.as_dict()
    assert audio.samples.tobytes() == before
    assert len({(x.start_frame, x.stop_frame) for x in a.candidates}) == len(a.candidates)
    keys = [(x.score, x.offset_distance, x.start_frame, x.stop_frame) for x in a.candidates]
    assert keys == sorted(keys)


@pytest.mark.parametrize("kwargs", [
    {"start_frame": True}, {"start_frame": 1000}, {"stop_frame": 50},
    {"start_radius": -1}, {"stop_radius": 99999}, {"context_frames": 2},
    {"stride": 0}, {"refine_count": 20}, {"results": 0},
    {"minimum_period": 1}, {"maximum_period": 1}, {"blend_frames": 1},
    {"weights": object()},
])
def test_invalid_search_plan(kwargs):
    config = {"start_frame": 100, "stop_frame": 1000}
    config.update(kwargs)
    with pytest.raises(AudioContractError):
        LoopSearchPlan(**config)


@pytest.mark.parametrize("weights", [
    {"sample_jump": True}, {"rms_mismatch": -1}, {"shape_mismatch": float("inf")},
    dict(sample_jump=0, slope_jump=0, rms_mismatch=0, shape_mismatch=0,
         spectrum_mismatch=0, silence_penalty=0),
])
def test_invalid_weights(weights):
    with pytest.raises(AudioContractError):
        LoopScoreWeights(**weights)


def test_invalid_or_impossible_span():
    audio = periodic()
    for plan in (LoopSearchPlan(0, audio.frames+1),
                 LoopSearchPlan(100, 110, start_radius=0, stop_radius=0),
                 LoopSearchPlan(100, 1000, start_radius=0, stop_radius=0,
                                minimum_period=2000)):
        with pytest.raises(AudioContractError):
            search_loop(audio, plan)


@pytest.mark.parametrize("constant, match", [
    ("MAX_GRID_PAIRS", "grid"), ("MAX_PAIR_WORK", "work"),
    ("MAX_WINDOWS_BYTES", "cache"),
])
def test_search_work_caps(monkeypatch, constant, match):
    import sonumio.audio_core.loop_analysis as module
    monkeypatch.setattr(module, constant, 1)
    with pytest.raises(AudioLimitError, match=match):
        search_loop(periodic(), LoopSearchPlan(100, 1000, context_frames=64))
