"""Numerical temporal continuation oracle, controls, audio grouping and guards."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.adjacent_audio import adjacent_bilingual_dataset
from research.latent_audio import FRAMES
from research.temporal_continuity_benchmark import (
    PROTOCOL, WAVE_PROTOCOL, ARMS, ERROR_METRICS, SeamEvaluationConfig,
    score_continuation, benchmark_adjacent_dataset,
    compare_continuations, _controls, _group_summary,
)


def adjacent(overlap=512, *, seed=101):
    return adjacent_bilingual_dataset(
        (seed,), prefix_frames=overlap,
    )


@pytest.fixture(scope="module")
def dataset():
    return adjacent()


@pytest.fixture(scope="module")
def result(dataset):
    return benchmark_adjacent_dataset(
        dataset, config=SeamEvaluationConfig(prefix_frames=512),
    )


def _sample(dataset, index=0):
    return dataset.previous[index, 0], dataset.future[index, 0]


@pytest.mark.parametrize("width", [128, 256, 512, 1024, 4096])
def test_exact_source_continuation_zero_signal_error(width):
    dataset = adjacent(width)
    before, future = _sample(dataset)
    record = score_continuation(
        before, future, future.clone(),
        config=SeamEvaluationConfig(prefix_frames=width),
    )
    assert record["protocol"] == WAVE_PROTOCOL
    assert record["prefix_frames"] == width
    assert record["unknown_future_frames"] == FRAMES - width
    assert record["known_prefix_is_bitwise_equal"] is True
    assert record["candidate_matches_true_future_exactly"] is True
    assert record["previous_sha256"] == hashlib.sha256(
        before.numpy().tobytes()
    ).hexdigest()
    assert record["true_next_sha256"] == record["candidate_next_sha256"]
    assert all(record["metrics"][metric] == 0 for metric in ERROR_METRICS)
    assert record["reference_first_unknown_sample_jump"] == pytest.approx(
        record["candidate_first_unknown_sample_jump"],
    )
    assert record["model_weights_or_audio_written"] is False
    assert record["human_listening_or_semantic_continuity_measured"] is False


def test_real_silent_and_wrong_source_controls_share_exact_known_prefix(dataset):
    prior, future = _sample(dataset)
    arms = _controls(dataset, 0)
    assert set(arms) == set(ARMS)
    for wave in arms.values():
        assert torch.equal(wave[:512], prior[-512:])
        assert wave.shape == (FRAMES,)
        assert wave.dtype == torch.float32
        assert torch.isfinite(wave).all()
    assert torch.equal(arms["true_adjacent_future"], future)
    assert torch.count_nonzero(arms["silence_after_prefix"][512:]) == 0
    assert not torch.equal(arms["unrelated_future"][512:], future[512:])
    oracle = score_continuation(
        prior, future, arms["true_adjacent_future"],
        config=SeamEvaluationConfig(prefix_frames=512),
    )
    silent = score_continuation(
        prior, future, arms["silence_after_prefix"],
        config=SeamEvaluationConfig(prefix_frames=512),
    )
    wrong = score_continuation(
        prior, future, arms["unrelated_future"],
        config=SeamEvaluationConfig(prefix_frames=512),
    )
    assert all(oracle["metrics"][key] == 0 for key in ERROR_METRICS)
    assert silent["metrics"]["future_sample_mae"] > 0
    assert wrong["metrics"]["future_sample_mae"] > 0
    assert silent["candidate_matches_true_future_exactly"] is False
    assert wrong["candidate_matches_true_future_exactly"] is False


def test_corrupt_one_distant_future_sample_increases_error():
    dataset = adjacent(512)
    prior, future = _sample(dataset)
    altered = future.clone()
    altered[FRAMES - 10] += 0.03
    score = score_continuation(
        prior, future, altered,
        config=SeamEvaluationConfig(prefix_frames=512),
    )
    assert score["metrics"]["future_sample_mae"] > 0
    assert score["metrics"]["future_sample_rmse"] > 0
    assert score["metrics"]["future_first_difference_mae"] > 0
    assert score["metrics"]["first_new_sample_jump_absolute_error"] == 0
    assert score["metrics"]["first_new_sample_slope_change_absolute_error"] == 0
    assert score["candidate_matches_true_future_exactly"] is False


def test_different_first_future_sample_changes_jump_and_slope():
    dataset = adjacent(1024)
    previous, future = _sample(dataset)
    disturbed = future.clone()
    disturbed[1024] += 0.2
    score = score_continuation(
        previous, future, disturbed,
        config=SeamEvaluationConfig(prefix_frames=1024),
    )
    assert score["metrics"]["first_new_sample_jump_absolute_error"] == pytest.approx(0.2, abs=1e-5)
    assert score["metrics"]["first_new_sample_slope_change_absolute_error"] == pytest.approx(0.4, abs=1e-5)
    assert score["metrics"]["future_sample_rmse"] > 0


def test_eight_independent_numerical_error_components_are_nonnegative(result):
    assert result["protocol"] == PROTOCOL
    assert len(result["configuration_sha256"]) == 64
    assert result["configuration"]["prefix_frames"] == 512
    assert result["configuration"]["source_recipe"] == "sonumio-adjacent-dsp-window-pairs-v1"
    source = result["source"]
    assert source["independent_recordings"] == 21
    assert source["bilingual_caption_examples"] == 42
    assert source["known_prefix_exactly_overlapped_in_source"] is True
    assert source["english_hindi_audio_counted_once_per_acoustic_group"] is True
    assert source["related_procedural_source_families"] is True
    assert len(source["source_sha256"]) == 21
    assert len(set(source["source_sha256"])) == 21
    assert len(result["per_recording"]) == 21
    assert len(set(x["acoustic_recording_group_sha256"] for x in result["per_recording"])) == 21
    assert result["ground_truth_oracle_zero_error_verified"] is True
    assert result["deliberately_mismatched_future_negative_control_verified"] is True
    assert result["zero_future_negative_control_verified"] is True
    assert result["same_prefix_for_all_three_arms"] is True
    assert result["no_pretrained_or_perceptual_model_used"] is True
    assert result["human_quality_or_audio_coherence_claim"] is False
    assert result["no_model_weights_or_wav_written"] is True
    assert result["model_publication_authorized"] is False
    all_scores = result["group_descriptive_error"]
    assert set(all_scores) == set(ARMS)
    for arm, values in all_scores.items():
        assert set(values) == set(ERROR_METRICS)
        for component in ERROR_METRICS:
            score = values[component]
            assert score["count"] == 21
            assert score["min"] <= score["mean"] <= score["max"]
            assert score["positive_groups"] + score["zero_groups"] == 21
            assert all(math.isfinite(score[field])
                       for field in ("mean", "min", "max"))
            assert score["min"] >= 0
            if arm == "true_adjacent_future":
                assert score["mean"] == 0
                assert score["zero_groups"] == 21
    assert all_scores["silence_after_prefix"]["future_sample_mae"]["mean"] > 0
    assert all_scores["unrelated_future"]["future_sample_mae"]["mean"] > 0


def test_each_independent_acoustic_recording_has_three_paired_scores(result):
    for row in result["per_recording"]:
        assert len(row["acoustic_recording_group_sha256"]) == 64
        assert len(row["source_full_recording_sha256"]) == 64
        assert row["recording_seed"] == 101
        assert row["caption_language_provenance"] == "same_audio_en_and_hi"
        measurements = row["candidate_scores"]
        assert set(measurements) == set(ARMS)
        for name in ARMS:
            values = measurements[name]
            assert set(values["metrics"]) == set(ERROR_METRICS)
            assert values["protocol"] == WAVE_PROTOCOL
            assert values["prefix_frames"] == 512
            assert values["known_prefix_is_bitwise_equal"] is True
            assert values["model_weights_or_audio_written"] is False
            assert values["sample_level_metric_only"] is True
            assert values["human_listening_or_semantic_continuity_measured"] is False
            assert all(math.isfinite(item) and item >= 0
                       for item in values["metrics"].values())
        assert measurements["true_adjacent_future"][
            "candidate_matches_true_future_exactly"] is True
        assert measurements["unrelated_future"][
            "candidate_matches_true_future_exactly"] is False
    assert len(json.dumps(result, allow_nan=False, ensure_ascii=False)) < 240_000


def test_direct_model_delta_comparison_preserves_reference_context(dataset):
    prior, future = _sample(dataset)
    controls = _controls(dataset, 0)
    score = compare_continuations(
        future, controls["silence_after_prefix"],
        controls["true_adjacent_future"],
        prior, prefix_frames=512,
    )
    assert score["protocol"] == PROTOCOL
    assert score["same_exact_previous_clip_and_true_future"] is True
    assert score["unclipped_source_and_candidate_inputs"] is True
    assert score["perceptual_audio_quality_or_long_context_semantics_measured"] is False
    for component, value in score["candidate_minus_reference_error"].items():
        assert component in ERROR_METRICS
        assert value <= 1e-8
        assert value == pytest.approx(-score["reference_error"][component])


@pytest.mark.parametrize("changes", [
    {"prefix_frames": 0}, {"prefix_frames": 127},
    {"prefix_frames": 129}, {"prefix_frames": 4097},
    {"prefix_frames": True}, {"prefix_frames": 512.0},
    {"fft_sizes": ()}, {"fft_sizes": (512, 256)},
    {"fft_sizes": (256, 256)}, {"fft_sizes": (123,)},
    {"fft_sizes": [256]}, {"fft_sizes": (True,)},
    {"highband_cutoff_hz": 200}, {"highband_cutoff_hz": 4000},
    {"highband_cutoff_hz": True},
    {"early_future_frames": 511}, {"early_future_frames": 1100},
    {"early_future_frames": True},
])
def test_bad_evaluation_plan_rejected(changes):
    with pytest.raises(ValueError):
        SeamEvaluationConfig(**changes)


@pytest.mark.parametrize("bad", [
    torch.zeros(FRAMES - 1),
    torch.zeros(FRAMES + 1),
    torch.zeros(FRAMES, dtype=torch.float64),
    torch.zeros((1, FRAMES)),
    torch.full((FRAMES,), float("nan")),
    torch.full((FRAMES,), float("inf")),
    torch.ones(FRAMES) * 1e5,
    torch.zeros(FRAMES, requires_grad=True),
])
def test_bad_candidate_waveforms_fail_before_metrics(dataset, bad):
    before, future = _sample(dataset)
    with pytest.raises(ValueError):
        score_continuation(
            before, future, bad,
            config=SeamEvaluationConfig(prefix_frames=512),
        )


def test_candidate_cannot_forge_known_prefix_even_when_suffix_is_right(dataset):
    previous, true = _sample(dataset)
    changed = true.clone()
    changed[1] += 0.1
    with pytest.raises(ValueError, match="prefix"):
        score_continuation(
            previous, true, changed,
            config=SeamEvaluationConfig(prefix_frames=512),
        )


def test_wrong_dataset_overlap_width_cannot_mix_into_source_report(dataset):
    with pytest.raises(ValueError, match="overlap"):
        benchmark_adjacent_dataset(
            dataset, config=SeamEvaluationConfig(prefix_frames=1024),
        )


def test_nonindependent_or_single_source_dataset_rejected():
    single = adjacent_bilingual_dataset(
        (101,), prefix_frames=512,
        combinations=((0,),),
    )
    with pytest.raises(ValueError):
        benchmark_adjacent_dataset(
            single, config=SeamEvaluationConfig(prefix_frames=512),
        )


def test_group_metric_aggregator_rejects_nonfinite_or_empty():
    for bad in ([], [float("nan")], [float("inf")], [1.0, float("nan")]):
        with pytest.raises(ValueError):
            _group_summary(bad)


def test_all_metrics_are_stable_under_identical_repeat_source(dataset):
    first = benchmark_adjacent_dataset(
        dataset, config=SeamEvaluationConfig(prefix_frames=512),
    )
    next_result = benchmark_adjacent_dataset(
        dataset, config=SeamEvaluationConfig(prefix_frames=512),
    )
    assert first["configuration_sha256"] == next_result["configuration_sha256"]
    assert first["source"]["source_sha256"] == next_result["source"]["source_sha256"]
    assert first["group_descriptive_error"] == next_result["group_descriptive_error"]
    assert [x["acoustic_recording_group_sha256"] for x in first["per_recording"]] == [
        x["acoustic_recording_group_sha256"] for x in next_result["per_recording"]
    ]
