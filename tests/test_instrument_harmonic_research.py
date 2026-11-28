"""Actual gradients, split isolation, waveform controls and study coverage."""
from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.instrument_note_data import (
    GridPlan, NoteSpec, NoteExample, NoteCorpus, PANELS, INSTRUMENTS,
    build_corpus, render_example, collate, training_schedule, evaluation_batches,
)
from research.instrument_harmonic_model import (
    HarmonicInstrument, initialize_model, fixed_sine, instrument_loss, note_metrics,
    VOICES, ANCHORS, NYQUIST_GUARD,
)
from research.instrument_harmonic_study import (
    StudyPlan, evaluate_panel, run_study, train_model, EVALUATION_PANELS,
    ARMS, ERROR_METRICS,
)
from research.weight_ema import model_state_sha256


def small_grid():
    return GridPlan(training_pitches=(48,), evaluation_pitches=(54,),
                    training_velocities=(500,), evaluation_velocities=(750,),
                    training_frames=(2048,), evaluation_frames=(3072,))


@pytest.fixture(scope="module")
def corpus():
    return build_corpus(small_grid())


def batch_for(corpus):
    return collate(corpus.panel("train")[:3])


def test_source_recipe_replays_exact_original_waveform():
    spec = NoteSpec("lead", 60, 750, 4096)
    a = render_example(spec)
    b = render_example(spec)
    assert a == b
    assert a.waveform_sha256 == b.waveform_sha256
    assert a.spec.identity == b.spec.identity
    assert len(a.pcm_le_f32) == spec.frames * 4
    assert a.samples.shape == (spec.frames,)
    assert a.samples.flags.writeable is False
    assert np.max(np.abs(a.samples)) <= 1.
    assert np.any(a.samples)
    assert a.spec.identity != replace(spec, pitch=61).identity
    assert a.waveform_sha256 != render_example(replace(spec, pitch=61)).waveform_sha256


@pytest.mark.parametrize("kwargs", [
    {"instrument": "snare"}, {"instrument": "unknown"},
    {"pitch": True}, {"pitch": 35}, {"pitch": 89}, {"pitch": 60.5},
    {"velocity_milli": False}, {"velocity_milli": 0}, {"velocity_milli": 1001},
    {"frames": 2047}, {"frames": 8193}, {"frames": 2048.},
])
def test_invalid_note_controls(kwargs):
    params = dict(instrument="lead", pitch=60, velocity_milli=500, frames=4096)
    params.update(kwargs)
    with pytest.raises(ValueError):
        NoteSpec(**params)


def test_invalid_note_bytes():
    spec = NoteSpec("keys", 60, 500, 2048)
    for raw in (b"", bytearray(4 * spec.frames), bytes(4 * spec.frames),
                np.full(spec.frames, np.nan, dtype="<f4").tobytes(),
                np.full(spec.frames, 2., dtype="<f4").tobytes()):
        with pytest.raises(ValueError):
            NoteExample(spec, raw)


@pytest.mark.parametrize("kwargs", [
    {"training_pitches": (48, 48)},
    {"training_pitches": (60, 48)},
    {"evaluation_pitches": (48,)},
    {"evaluation_velocities": (500,)},
    {"evaluation_frames": (2048,)},
    {"training_frames": [2048]},
    {"training_pitches": ()},
    {"training_velocities": (True,)},
])
def test_grid_rejects_overlapping_or_noncanonical_axes(kwargs):
    with pytest.raises(ValueError):
        replace(small_grid(), **kwargs)


def test_grid_refuses_cartesian_overflow():
    with pytest.raises(ValueError, match="budget"):
        GridPlan(training_pitches=tuple(range(36, 52)),
                 evaluation_pitches=tuple(range(52, 68)),
                 training_velocities=tuple(range(100, 116)),
                 evaluation_velocities=tuple(range(200, 216)),
                 training_frames=tuple(range(2048, 2064)),
                 evaluation_frames=tuple(range(4096, 4112)))


def test_panel_grid_and_disjoint_exact_waveforms(corpus):
    report = corpus.report()
    assert tuple(report["panels"]) == PANELS
    assert not report["cross_split_exact_waveform_overlap"]
    assert not report["cross_split_note_identity_overlap"]
    assert not report["external_recordings_used"]
    train = corpus.panel("train")
    train_ids = {x.spec.identity for x in train}
    train_pcm = {x.waveform_sha256 for x in train}
    assert len(train) == len(INSTRUMENTS)
    for name in PANELS[1:]:
        rows = corpus.panel(name)
        assert len(rows) == len(INSTRUMENTS)
        assert not train_ids & {x.spec.identity for x in rows}
        assert not train_pcm & {x.waveform_sha256 for x in rows}
        assert {x.spec.instrument for x in rows} == set(INSTRUMENTS)
        assert tuple(x.spec for x in rows) == corpus.plan.specifications(name)
    assert build_corpus(small_grid()).report() == report


def test_waveform_leakage_rejected_even_if_metadata_differs(corpus):
    changed = list(corpus.panels)
    rows = list(changed[1])
    rows[0] = NoteExample(rows[0].spec, corpus.panel("train")[0].pcm_le_f32)
    changed[1] = tuple(rows)
    with pytest.raises(ValueError, match="waveform crosses"):
        NoteCorpus(corpus.plan, tuple(changed))


def test_schedule_cycles_durations_and_is_reproducible():
    plan = replace(small_grid(), training_frames=(2048, 4096), evaluation_frames=(3072,))
    rows = build_corpus(plan).panel("train")
    a, digest = training_schedule(rows, updates=7, batch_size=2, seed=11)
    b, second = training_schedule(rows, updates=7, batch_size=2, seed=11)
    assert a == b and digest == second
    assert len(digest) == 64
    assert len(a) == 7
    for step, batch in enumerate(a):
        assert len(batch) == 2
        assert {rows[i].spec.frames for i in batch} == {(2048, 4096)[step % 2]}
    grouped = tuple(evaluation_batches(rows, 3))
    assert sorted(i for batch in grouped for i in batch) == list(range(len(rows)))
    assert all(len({rows[i].spec.frames for i in batch}) == 1 for batch in grouped)


def test_collate_owns_writable_tensor_and_preserves_source(corpus):
    rows = corpus.panel("train")[:2]
    batch = collate(rows)
    original = rows[0].pcm_le_f32
    assert batch.targets.shape == (2, 2048)
    assert batch.targets.dtype == torch.float32
    assert batch.instruments.dtype == torch.long
    assert batch.identities == tuple(x.spec.identity for x in rows)
    batch.targets.zero_()
    assert rows[0].pcm_le_f32 == original
    with pytest.raises(ValueError, match="share exact"):
        collate((rows[0], corpus.panel("duration")[0]))


def test_model_initialization_isolated_and_reproducible():
    before = torch.random.get_rng_state().clone()
    a = initialize_model(32, 17)
    assert torch.equal(before, torch.random.get_rng_state())
    b = initialize_model(32, 17)
    c = initialize_model(32, 29)
    assert model_state_sha256(a) == model_state_sha256(b)
    assert model_state_sha256(a) != model_state_sha256(c)
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(ValueError):
        HarmonicInstrument(31)
    with pytest.raises(ValueError):
        initialize_model(32, True)


def test_conditioned_parameters_bounded_and_output_shape(corpus):
    batch = batch_for(corpus)
    model = initialize_model(32, 17)
    params = model.predict_parameters(batch.instruments, batch.pitches, batch.velocities, batch.frames)
    assert params.amplitudes.shape == (3, VOICES, ANCHORS)
    assert torch.all(params.amplitudes >= 0)
    assert torch.all(params.amplitudes.sum(dim=1) <= 1.000001)
    assert torch.all(params.detune_cents[:, 0] == 0)
    assert torch.all(params.detune_cents.abs() <= 8.)
    assert torch.all((params.inharmonicity >= 0) & (params.inharmonicity <= .0003))
    assert torch.all((params.vibrato_depth >= 0) & (params.vibrato_depth <= .15))
    assert torch.all((params.attack_fraction > 0) & (params.attack_fraction < .5))
    assert torch.all((params.release_fraction > 0) & (params.release_fraction < .5))
    predicted = model(batch.instruments, batch.pitches, batch.velocities, batch.frames)
    assert predicted.shape == batch.targets.shape
    assert torch.isfinite(predicted).all()
    assert torch.all(predicted.abs().amax(dim=1) <= batch.velocities.float() / 1000. + 1e-6)
    assert torch.count_nonzero(predicted[:, 0]) == 0
    assert torch.count_nonzero(predicted[:, -1]) == 0


@pytest.mark.parametrize("mutation", ["instrument", "pitch", "velocity", "shape", "dtype", "frames"])
def test_invalid_model_inputs(corpus, mutation):
    batch = batch_for(corpus)
    args = [batch.instruments, batch.pitches, batch.velocities, batch.frames]
    if mutation == "instrument":
        args[0] = torch.full_like(args[0], len(INSTRUMENTS))
    if mutation == "pitch":
        args[1] = torch.full_like(args[1], 100)
    if mutation == "velocity":
        args[2] = torch.zeros_like(args[2])
    if mutation == "shape":
        args[0] = args[0][:, None]
    if mutation == "dtype":
        args[1] = args[1].float()
    if mutation == "frames":
        args[3] = 2047
    with pytest.raises(ValueError):
        initialize_model(32, 17)(*args)


def test_fixed_sine_is_label_independent_and_learned_output_is_conditioned(corpus):
    batch = batch_for(corpus)
    wrong = (batch.instruments + 1) % len(INSTRUMENTS)
    sine = fixed_sine(batch.instruments, batch.pitches, batch.velocities, batch.frames)
    alternate_sine = fixed_sine(wrong, batch.pitches, batch.velocities, batch.frames)
    torch.testing.assert_close(sine, alternate_sine, rtol=0, atol=0)
    model = initialize_model(32, 17)
    matched = model(batch.instruments, batch.pitches, batch.velocities, batch.frames)
    mismatched = model(wrong, batch.pitches, batch.velocities, batch.frames)
    assert not torch.equal(matched, mismatched)


def test_waveform_objective_and_actual_gradient(corpus):
    batch = batch_for(corpus)
    model = initialize_model(32, 17)
    predicted = model(batch.instruments, batch.pitches, batch.velocities, batch.frames)
    loss = instrument_loss(predicted, batch.targets)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad) > 0 for p in model.parameters())
    reference = note_metrics(batch.targets, batch.targets)
    assert torch.count_nonzero(reference["relative_wave_mse"]) == 0
    assert torch.count_nonzero(reference["relative_spectral_error"]) == 0
    torch.testing.assert_close(reference["rms_ratio"], torch.ones(3))
    torch.testing.assert_close(reference["waveform_cosine"], torch.ones(3))
    with pytest.raises(ValueError):
        instrument_loss(predicted[:, :-1], batch.targets)


def test_wrong_condition_exhausts_all_four_labels_without_mutating_model(corpus, monkeypatch):
    rows = corpus.panel("joint")
    model = initialize_model(32, 17)
    state = model_state_sha256(model)
    original_forward = model.forward
    calls = []

    def recorded(labels, pitches, velocities, frames):
        calls.append((labels.tolist(), pitches.tolist(), velocities.tolist(), frames))
        return original_forward(labels, pitches, velocities, frames)

    monkeypatch.setattr(model, "forward", recorded)
    report = evaluate_panel(rows, len(INSTRUMENTS), model=model, arm="wrong_instrument")
    original = [row.spec.instrument_id for row in rows]
    assert len(calls) == len(INSTRUMENTS) - 1
    assert [call[0] for call in calls] == [
        [(value + offset) % len(INSTRUMENTS) for value in original]
        for offset in range(1, len(INSTRUMENTS))
    ]
    assert all(call[1] == [row.spec.pitch for row in rows] for call in calls)
    assert all(call[2] == [row.spec.velocity_milli for row in rows] for call in calls)
    assert all(call[3] == rows[0].spec.frames for call in calls)
    assert report["wrong_conditions_per_note"] == len(INSTRUMENTS) - 1
    assert report["notes"] == len(rows)
    assert model_state_sha256(model) == state
    assert model.training


def test_training_requires_prescribed_schedule_and_changes_parameters(corpus):
    plan = StudyPlan(model_seeds=(17, 29), width=32, updates=2, batch_size=2,
                     evaluation_batch_size=5, grid=corpus.plan)
    rows = corpus.panel("train")
    schedule, digest = training_schedule(rows, updates=2, batch_size=2, seed=plan.sampler_seed)
    model = initialize_model(32, 17)
    before = model_state_sha256(model)
    with pytest.raises(ValueError, match="schedule"):
        train_model(model, corpus, plan, tuple(reversed(schedule)))
    assert model_state_sha256(model) == before
    report = train_model(model, corpus, plan, schedule)
    assert report["optimizer_updates"] == 2
    assert report["note_examples_seen"] == 4
    assert report["waveform_samples_seen"] == 4 * 2048
    assert report["schedule_sha256"] == digest
    assert report["initial_model_sha256"] == before
    assert report["trained_model_sha256"] != before


def test_small_replicated_study_has_complete_controls_and_no_publication():
    plan = StudyPlan(model_seeds=(17, 29), width=32, updates=2, batch_size=2,
                     evaluation_batch_size=5, grid=small_grid())
    report = run_study(plan)
    json.dumps(report, allow_nan=False)
    assert len(report["results"]) == 2
    assert report["independent_initialization_replicates"] == 2
    assert report["wrong_instrument_conditions"] == 4
    for key in ("evaluation_used_for_training", "checkpoint_written", "generated_audio_written",
                "model_weights_published", "perceptual_quality_established",
                "open_vocabulary_text_conditioning", "statistical_significance_claimed"):
        assert report[key] is False
    for result in report["results"]:
        training = result["training"]
        assert training["schedule_sha256"] == report["shared_training_schedule_sha256"]
        assert training["trained_model_sha256"] != training["initial_model_sha256"]
        assert set(result["evaluation"]) == set(ARMS)
        for panel in EVALUATION_PANELS:
            for arm in ARMS:
                row = result["evaluation"][arm][panel]
                assert row["notes"] == len(INSTRUMENTS)
                assert set(row["by_instrument"]) == set(INSTRUMENTS)
                assert all(math.isfinite(v) for v in row["instrument_macro"].values())
            for metric in ERROR_METRICS:
                effects = result["paired_effects"][panel][metric]
                assert all(math.isfinite(v) for v in effects.values())
    assert report["results"][0]["training"]["initial_model_sha256"] != report["results"][1]["training"]["initial_model_sha256"]
    assert set(report["seed_summary"]) == set(EVALUATION_PANELS)


@pytest.mark.parametrize("changes", [
    {"model_seeds": (17,)}, {"model_seeds": (17, 17)}, {"width": 16},
    {"batch_size": 17}, {"updates": 0}, {"evaluation_batch_size": True},
    {"model_seeds": (1, 2, 3, 4, 5), "updates": 512, "batch_size": 16},
])
def test_study_rejects_invalid_or_excessive_budget(changes):
    with pytest.raises(ValueError):
        replace(StudyPlan(), **changes)
