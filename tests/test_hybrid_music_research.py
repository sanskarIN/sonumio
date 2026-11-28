"""End-to-end score-level tests for research-only learned instruments."""
from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.hybrid_music_render import (
    ARMS, NoteRoute, compare_hybrid_score, compare_stems, describe_routes,
    render_hybrid_stems, route_score,
)
from research.hybrid_music_study import HybridStudyPlan, ScoreCase, prepare_panel, run_study
from research.instrument_harmonic_model import initialize_model
from research.instrument_harmonic_study import StudyPlan
from research.instrument_note_data import GridPlan, INSTRUMENTS
from research.weight_ema import model_state_sha256
from sonumio.arrangement import ArrangementPlan, TRACKS, compose_arrangement
from sonumio.errors import GenerationError
from sonumio.music_mix import _render_stems, render_stems


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(style="rhythmic", mode="minor",
                                               root=57, bpm=160, bars=4, seed=131))


@pytest.fixture(scope="module")
def original(score):
    return render_stems(score)


def test_routes_partition_every_original_note(score):
    routes = route_score(score)
    summary = describe_routes(routes)
    assert tuple(r.event for r in routes) == score.events
    assert summary["eligible_events"] > 0
    assert summary["eligible_events"] < summary["events"]
    assert summary["eligible_samples"] < summary["event_samples"]
    assert sum(summary["fallback_reasons"].values()) == summary["events"] - summary["eligible_events"]
    for route in routes:
        assert route.start == score.plan.frame(route.event.tick)
        assert route.stop == score.plan.frame(route.event.tick + route.event.length)
        if route.spec is not None:
            assert route.fallback is None
            assert route.spec.frames == route.frames
            assert route.spec.pitch == route.event.pitch
        elif route.event.track not in INSTRUMENTS:
            assert route.fallback == "percussion"


def test_exact_training_note_exclusions(score):
    routes = route_score(score)
    eligible = next(r for r in routes if r.spec is not None)
    excluded = frozenset((eligible.spec.identity,))
    blocked = route_score(score, excluded=excluded)
    assert sum(r.fallback == "training_overlap" for r in blocked) == sum(
        r.spec is not None and r.spec.identity == eligible.spec.identity for r in routes
    )
    assert all(r.spec is None or r.spec.identity not in excluded for r in blocked)


@pytest.mark.parametrize("exclusions", [set(), frozenset(("not-a-digest",)), frozenset(("A"*64,))])
def test_invalid_exclusion_contract(score, exclusions):
    with pytest.raises(ValueError):
        route_score(score, excluded=exclusions)


def test_route_rejects_inconsistent_geometry(score):
    route = next(r for r in route_score(score) if r.spec is not None)
    with pytest.raises(ValueError):
        replace(route, stop=route.stop + 1)
    with pytest.raises(ValueError):
        replace(route, fallback="duration")
    with pytest.raises(ValueError):
        NoteRoute(route.event, route.start, route.stop, None, None)


def test_procedural_control_is_sample_exact(score, original):
    candidate, coverage = render_hybrid_stems(score, arm="procedural")
    assert coverage["substituted_events"] == 0
    assert coverage["procedural_events"] == len(score.events)
    for track in TRACKS:
        np.testing.assert_array_equal(candidate[track], original[track])
        assert not candidate[track].flags.writeable


def test_sine_changes_only_eligible_tracks(score, original):
    candidate, coverage = render_hybrid_stems(score, arm="fixed_sine")
    report = compare_stems(candidate, original, coverage)
    assert coverage["substituted_events"] == coverage["eligible_events"]
    assert coverage["substituted_samples"] == coverage["eligible_samples"]
    for track in TRACKS:
        if coverage["tracks"][track]["eligible_events"] == 0:
            assert report[track]["exact_pcm_match"]
    assert any(not item["exact_pcm_match"] for item in report.values())


def test_matched_and_wrong_labels_preserve_state_and_fallback(score, original):
    model = initialize_model(32, 17)
    before = model_state_sha256(model)
    model.train()
    a, ar = render_hybrid_stems(score, arm="matched", model=model)
    b, br = render_hybrid_stems(score, arm="wrong_instrument", model=model, wrong_offset=2)
    assert model.training
    assert model_state_sha256(model) == before == ar["model_sha256"] == br["model_sha256"]
    assert ar["eligible_events"] == br["eligible_events"]
    assert br["wrong_offset"] == 2
    for track in TRACKS:
        if ar["tracks"][track]["eligible_events"] == 0:
            np.testing.assert_array_equal(a[track], original[track])
            np.testing.assert_array_equal(b[track], original[track])
    assert any(not np.array_equal(a[track], b[track]) for track in INSTRUMENTS)


@pytest.mark.parametrize("kwargs", [
    {"arm": "invalid"}, {"arm": "matched"}, {"arm": "wrong_instrument"},
    {"arm": "fixed_sine", "model": object()},
    {"arm": "procedural", "wrong_offset": 0},
])
def test_invalid_arm_and_model(score, kwargs):
    with pytest.raises(ValueError):
        render_hybrid_stems(score, **kwargs)


def test_excluding_all_eligible_notes_recovers_original(score, original):
    excluded = frozenset(r.spec.identity for r in route_score(score) if r.spec)
    candidate, coverage = render_hybrid_stems(score, arm="matched",
                                               model=initialize_model(32, 29), excluded=excluded)
    assert coverage["eligible_events"] == 0
    assert coverage["substituted_events"] == 0
    for track in TRACKS:
        np.testing.assert_array_equal(candidate[track], original[track])


def test_shared_render_rejects_invalid_waveforms(score):
    for renderer in (
        lambda e, n, s: np.zeros(n+1),
        lambda e, n, s: np.zeros(n, dtype=np.int16),
        lambda e, n, s: np.full(n, np.nan),
        lambda e, n, s: np.full(n, 2.0),
    ):
        with pytest.raises(GenerationError, match="note renderer"):
            _render_stems(score, renderer)


def test_score_budget_rejects_long_score():
    score = compose_arrangement(ArrangementPlan(bars=16, bpm=60))
    with pytest.raises(ValueError, match="budget"):
        route_score(score)


def test_complete_comparison_report(score):
    report = compare_hybrid_score(score, arm="fixed_sine")
    assert report["candidate_wav_sha256"] != report["reference_wav_sha256"]
    assert report["sample_rate"] == 24000
    assert report["frames"] == score.plan.frame(score.plan.ticks)
    assert report["mix_comparison"]["relative_wave_mse"] > 0
    assert report["learned_events_used"] is False
    assert report["generated_audio_written"] is False
    assert report["perceptual_quality_established"] is False
    json.dumps(report, allow_nan=False)


def test_exact_procedural_comparison(score):
    report = compare_hybrid_score(score, arm="procedural")
    assert report["candidate_wav_sha256"] == report["reference_wav_sha256"]
    assert report["mix_comparison"]["relative_wave_mse"] == 0
    assert report["mix_comparison"]["exact_pcm_match"]


def test_study_small_replication_runs_actual_training_and_all_controls():
    grid = GridPlan(training_pitches=(60,), evaluation_pitches=(66,),
                    training_velocities=(500,), evaluation_velocities=(750,),
                    training_frames=(4096,), evaluation_frames=(6144,))
    training = StudyPlan(model_seeds=(17, 29), sampler_seed=11,
                         width=32, updates=2, batch_size=2, grid=grid)
    case = ScoreCase("rhythmic", "minor", 57, 160, 131)
    plan = HybridStudyPlan(training=training, cases=(case,))
    panel = prepare_panel(plan, frozenset())
    assert panel[0][2]["eligible_events"] > 0
    report = run_study(plan)
    assert report["score_cases"] == 1
    assert report["exact_training_control_identities_excluded"] == 5
    assert len(report["replicates"]) == 2
    for replicate in report["replicates"]:
        assert replicate["training"]["optimizer_updates"] == 2
        assert len(replicate["cases"][0]["wrong_instrument"]) == 4
        assert replicate["cases"][0]["trained"]["learned_events_used"]
    assert report["evaluation_used_for_training"] is False
    json.dumps(report, allow_nan=False)
