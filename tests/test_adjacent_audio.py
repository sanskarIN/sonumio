"""Provenance, adjacency and leakage regression tests for first-party DSP pairs."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.adjacent_audio import (
    FRAMES, RECIPE, AdjacentDataset, audit_adjacent_split,
    adjacent_bilingual_dataset, paired_continuity_probes,
    render_adjacent_recording,
)
from research.text_audio import SYNTHETIC_COMBINATIONS


@pytest.fixture(scope="module")
def training_pairs():
    return adjacent_bilingual_dataset((0, 1), prefix_frames=1024)


@pytest.fixture(scope="module")
def evaluation_pairs():
    return adjacent_bilingual_dataset((101, 102), prefix_frames=1024)


@pytest.mark.parametrize("overlap", [128, 512, 1024, 4096])
@pytest.mark.parametrize("members", [(0,), (3,), (5,), (0, 5)])
def test_adjacent_windows_reconstruct_exact_original_source(members, overlap):
    first = render_adjacent_recording(members, 17, prefix_frames=overlap)
    again = render_adjacent_recording(members, 17, prefix_frames=overlap)
    assert first.seed == again.seed
    assert first.offset == again.offset
    assert first.members == members
    assert first.prefix_frames == overlap
    assert first.previous.shape == first.future.shape == (FRAMES,)
    assert first.previous.dtype == first.future.dtype == np.float32
    assert np.array_equal(first.previous, again.previous)
    assert np.array_equal(first.future, again.future)
    assert first.source_sha256 == again.source_sha256
    assert first.join_sha256 == again.join_sha256
    assert first.new_frames == FRAMES - overlap
    assert first.reconstructed_pair_frames == 2 * FRAMES - overlap
    assert np.array_equal(first.previous[-overlap:], first.future[:overlap])
    assert np.isfinite(first.previous).all() and np.isfinite(first.future).all()
    stitched = np.concatenate((first.previous, first.future[overlap:]))
    assert stitched.shape == (2 * FRAMES - overlap,)
    assert hashlib.sha256(stitched.tobytes()).hexdigest() == first.join_sha256
    assert abs(first.previous).max() <= 1.0001
    assert abs(first.future).max() <= 1.0001


def test_same_family_different_seeds_have_different_recordings():
    for members in ((5,), (3,), (0, 5)):
        first = render_adjacent_recording(members, 0)
        second = render_adjacent_recording(members, 1)
        assert first.source_sha256 != second.source_sha256
        assert first.join_sha256 != second.join_sha256
        assert first.previous_sha256 != second.previous_sha256


@pytest.mark.parametrize("bad", [
    -1, 2**32, True, 0.0, None, "1",
])
def test_invalid_seed_rejected(bad):
    with pytest.raises(ValueError):
        render_adjacent_recording((0,), bad)


@pytest.mark.parametrize("bad", [
    (), (0, 0), (99,), (0, 1, 2), [0], 0, None,
])
def test_invalid_sound_family_rejected(bad):
    with pytest.raises(ValueError):
        render_adjacent_recording(bad, 1)


@pytest.mark.parametrize("bad", [
    0, 127, 129, 4097, True, 512.0, -512,
])
def test_invalid_prefix_width_rejected(bad):
    with pytest.raises(ValueError):
        render_adjacent_recording((0,), 1, prefix_frames=bad)


@pytest.mark.parametrize("bad", [
    (), (0, 0), (True,), (-1,), (2**32,), "0", [0, 1],
])
def test_bad_seed_tuple_rejected_before_dsp(bad):
    with pytest.raises(ValueError):
        adjacent_bilingual_dataset(bad)


@pytest.mark.parametrize("bad", [
    (), ((),), ((0,), (0,)), ((99,),), ((0, 1, 2),), [(0,)],
])
def test_bad_combination_tuple_rejected_before_dsp(bad):
    with pytest.raises(ValueError):
        adjacent_bilingual_dataset((0,), combinations=bad)


def test_complete_training_dataset_is_grouped_by_recording(training_pairs):
    data = training_pairs
    data.check()
    assert data.examples == 84
    assert data.group_count == 42
    assert len(data.captions) == 84
    assert len(data.group_ids) == 84
    assert len(set(data.group_ids)) == 42
    assert len(set(data.source_hashes)) == 42
    assert data.tokens.shape == (84, 130)
    assert data.previous.shape == data.future.shape == (84, 1, FRAMES)
    assert data.prefix_frames == 1024
    assert len(data.fingerprint) == 64
    assert data.exact_adjacency_verified is True
    assert torch.equal(data.previous[:, :, -1024:], data.future[:, :, :1024])
    assert torch.equal(data.previous[::2], data.previous[1::2])
    assert torch.equal(data.future[::2], data.future[1::2])
    assert not torch.equal(data.tokens[::2], data.tokens[1::2])
    assert data.languages[::2] == ("en",) * 42
    assert data.languages[1::2] == ("hi",) * 42
    assert len(set(data.recording_seeds)) == 2
    assert data.captions[0] != data.captions[1]


def test_reproducible_dataset_with_identical_parameters(training_pairs):
    repeated = adjacent_bilingual_dataset((0, 1), prefix_frames=1024)
    assert repeated.fingerprint == training_pairs.fingerprint
    assert repeated.group_ids == training_pairs.group_ids
    assert repeated.source_hashes == training_pairs.source_hashes
    assert torch.equal(repeated.previous, training_pairs.previous)
    assert torch.equal(repeated.future, training_pairs.future)
    assert torch.equal(repeated.tokens, training_pairs.tokens)


def test_seed_disjoint_training_and_heldout_audit(training_pairs, evaluation_pairs):
    result = audit_adjacent_split(training_pairs, evaluation_pairs)
    assert result["protocol"] == RECIPE
    assert result["train_caption_examples"] == 84
    assert result["heldout_caption_examples"] == 84
    assert result["train_independent_recordings"] == 42
    assert result["heldout_independent_recordings"] == 42
    assert result["source_recording_groups_not_split_across_languages"] is True
    assert result["exact_audio_source_leakage_detected"] is False
    assert result["shared_seed_leakage_detected"] is False
    assert result["independent_acoustic_families_verified"] is False
    assert result["seeded_envelope_augmentation"] is True
    assert result["training_seeds"] == [0, 1]
    assert result["heldout_seeds"] == [101, 102]
    assert result["prefix_frames"] == 1024
    assert result["train_fingerprint"] != result["heldout_fingerprint"]


def test_train_heldout_seed_reuse_is_a_hard_error(training_pairs):
    with pytest.raises(ValueError, match="leakage"):
        audit_adjacent_split(training_pairs, training_pairs)


def test_multiple_prefix_lengths_are_incompatible(training_pairs):
    altered = adjacent_bilingual_dataset(
        (101,), prefix_frames=512, combinations=((0,),),
    )
    with pytest.raises(ValueError, match="overlap"):
        audit_adjacent_split(training_pairs, altered)


def test_broken_adjacency_is_rejected_even_with_true_flag(training_pairs):
    bad = training_pairs
    broken = bad.future.clone()
    broken[0, 0, 0] += 0.1
    replacement = AdjacentDataset(
        previous=bad.previous, future=broken,
        tokens=bad.tokens, captions=bad.captions,
        group_ids=bad.group_ids, source_hashes=bad.source_hashes,
        languages=bad.languages, recording_seeds=bad.recording_seeds,
        prefix_frames=bad.prefix_frames, group_count=bad.group_count,
        fingerprint=bad.fingerprint,
        exact_adjacency_verified=True,
    )
    with pytest.raises(ValueError, match="boundary"):
        replacement.check()


def test_bilingual_provenance_requires_same_audio_for_both_languages(training_pairs):
    bad = training_pairs
    previous = bad.previous.clone()
    previous[1, 0, 4096] += 0.1
    altered = AdjacentDataset(
        previous=previous, future=bad.future,
        tokens=bad.tokens, captions=bad.captions,
        group_ids=bad.group_ids, source_hashes=bad.source_hashes,
        languages=bad.languages, recording_seeds=bad.recording_seeds,
        prefix_frames=bad.prefix_frames, group_count=bad.group_count,
        fingerprint=bad.fingerprint,
        exact_adjacency_verified=True,
    )
    with pytest.raises(ValueError, match="Bilingual"):
        altered.check()


def test_exact_source_stitch_has_numerical_seam_controls(training_pairs):
    metrics = paired_continuity_probes(training_pairs)
    assert metrics["recordings_counted_once"] == 42
    assert metrics["caption_examples"] == 84
    assert metrics["prefix_frames"] == 1024
    assert metrics["ground_truth_contiguous_windows_not_generated"] is True
    assert metrics["perceptual_audio_quality_measured"] is False
    assert metrics["semantic_continuity_or_prompt_adherence_measured"] is False
    for key in ("hard_clip_concat_jump", "true_first_unknown_sample_jump"):
        value = metrics["mean_jump"][key]
        assert 0 <= value < 10
        assert np.isfinite(value)


@pytest.mark.parametrize("overlap", [128, 4096])
def test_one_seed_subset_does_not_conflate_caption_and_recording_count(overlap):
    data = adjacent_bilingual_dataset(
        (17,), prefix_frames=overlap, combinations=((0,), (5,)),
    )
    assert data.examples == 4
    assert data.group_count == 2
    assert data.languages == ("en", "hi", "en", "hi")
    assert torch.equal(data.previous[:, :, -overlap:], data.future[:, :, :overlap])
    data.check()
