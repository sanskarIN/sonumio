"""Rights-scoped acoustic-evaluation manifests, provenance and leakage gates."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError
from research.audio_eval.reference import (
    RightsDeclaration, AudioReference, EvaluationPair,
    EvaluationSet, blind_pair_labels, group_provenance_report,
)


@pytest.fixture
def legal_synthetic():
    return RightsDeclaration(
        "first-party-synthetic", "first-party-test-fixture",
        rights_reviewed=True, voice_present=False, creator_permission=True,
        attribution="Sonumio original synthetic test fixtures",
    )


@pytest.fixture
def source_wave():
    t = np.arange(4096, dtype=np.float32) / 24000
    return make_buffer((.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32), 24000)


def reference(asset_id, signal, rights, *, group="group_a", split="test",
              source_sha256=None):
    return AudioReference(
        asset_id=asset_id, buffer=signal, group_id=group,
        split=split, language="nonverbal", caption="Synthetic 400Hz tone",
        rights=rights, source_sha256=source_sha256,
    )


def pair(case_id, asset_id, signal, rights, *, group="group_a",
         split="test"):
    ref = reference(asset_id, signal, rights, group=group, split=split)
    return EvaluationPair(case_id, ref, signal, "synthetic-reference")


def test_metadata_records_rights_as_declarations_not_proof(source_wave, legal_synthetic):
    ref = reference("fixture_01", source_wave, legal_synthetic)
    assert ref.metadata()["rights"]["rights_reviewed"] is True
    assert ref.metadata()["rights"]["voice_present"] is False
    assert ref.metadata()["sample_rate"] == 24000
    assert ref.metadata()["source_frames"] == 4096
    assert len(ref.source_sha256) == 64
    assert ref.metadata()["source_sha256"] == ref.source_sha256


def test_reference_rejects_forged_pcm_hash(source_wave, legal_synthetic):
    with pytest.raises(AudioContractError, match="hash"):
        reference("fixture_01", source_wave, legal_synthetic,
                  source_sha256="0" * 64)
    assert source_wave.frames == 4096


def test_rights_must_be_finite_supported_and_human_declared():
    with pytest.raises(AudioContractError):
        RightsDeclaration("unknown", "CC0-1.0", True, False, True)
    with pytest.raises(AudioContractError):
        RightsDeclaration("owned", "made-up-license", True, False, True)
    with pytest.raises(AudioContractError):
        RightsDeclaration("owned", "CC0-1.0", "yes", False, True)
    with pytest.raises(AudioContractError):
        RightsDeclaration("direct-permission", "CC-BY-4.0", True, False, False)
    rights = RightsDeclaration("owned", "proprietary-with-permission",
                               True, False, True)
    assert rights.admissible_for_synthetic_acoustic_test()


def test_voice_present_flag_disqualifies_automatic_acoustic_test(source_wave):
    voice = RightsDeclaration("owned", "CC0-1.0", True, True, True)
    assert voice.admissible_for_synthetic_acoustic_test() is False
    entry = pair("case1", "a1", source_wave, voice)
    with pytest.raises(AudioContractError, match="rights declaration"):
        EvaluationSet((entry,))
    unsafe = EvaluationSet((entry,), strict_rights=False)
    assert unsafe.metadata()["declared_rights_strict"] is False
    assert unsafe.metadata()["external_rights_authentication_performed"] is False


def test_first_party_rights_do_not_permit_globally_unseen_claim(source_wave, legal_synthetic):
    sample = pair("a", "a", source_wave, legal_synthetic)
    result = EvaluationSet((sample,))
    metadata = result.metadata()
    assert metadata["claim_of_globally_unseen_neural_evaluation"] is False
    assert metadata["external_rights_authentication_performed"] is False
    assert metadata["split_sizes"] == {"test": 1}
    assert metadata["unique_groups"] == 1
    assert len(metadata["fingerprint"]) == 64
    assert result.fingerprint == result.fingerprint


def test_dataset_rejects_same_group_across_train_and_test(source_wave, legal_synthetic):
    p = pair("train1", "train1", source_wave, legal_synthetic,
             group="shared", split="train")
    q = pair("test1", "test1", source_wave, legal_synthetic,
             group="shared", split="test")
    with pytest.raises(AudioContractError, match="group"):
        EvaluationSet((p, q))


def test_dataset_rejects_same_pcm_across_train_and_test(source_wave, legal_synthetic):
    p = pair("train1", "train1", source_wave, legal_synthetic,
             group="group1", split="train")
    q = pair("test1", "test1", source_wave, legal_synthetic,
             group="group2", split="test")
    with pytest.raises(AudioContractError, match="PCM"):
        EvaluationSet((p, q))


def test_dataset_rejects_same_asset_across_split_even_when_waveform_changes(
    source_wave, legal_synthetic
):
    changed = source_wave.copy_with(source_wave.samples * .9)
    p = pair("train1", "same_asset", source_wave, legal_synthetic,
             group="group1", split="train")
    q = pair("test1", "same_asset", changed, legal_synthetic,
             group="group2", split="test")
    with pytest.raises(AudioContractError, match="asset"):
        EvaluationSet((p, q))


def test_same_group_is_valid_within_one_split(source_wave, legal_synthetic):
    modified = source_wave.copy_with(source_wave.samples * .8)
    p = pair("case1", "asset1", source_wave, legal_synthetic,
             group="g", split="test")
    q = pair("case2", "asset2", modified, legal_synthetic,
             group="g", split="test")
    evaluation = EvaluationSet((p, q))
    assert evaluation.metadata()["unique_groups"] == 1
    assert len(evaluation.groups()["g"]) == 2
    assert len(evaluation.splits()["test"]) == 2
    report = group_provenance_report(evaluation)
    assert report["group_sizes"] == {"g": 2}
    assert report["human_rights_review_still_required"] is True
    assert report["neural_training_not_performed"] is True


def test_duplicate_case_identifiers_are_rejected(source_wave, legal_synthetic):
    a = pair("case1", "a1", source_wave, legal_synthetic)
    b = pair("case1", "a2", source_wave, legal_synthetic)
    with pytest.raises(AudioContractError, match="Duplicate evaluation case"):
        EvaluationSet((a, b))


def test_pair_rejects_sample_rate_shape_and_channel_mismatch(
    source_wave, legal_synthetic
):
    ref = reference("a1", source_wave, legal_synthetic)
    other_clock = make_buffer(source_wave.samples, 44100)
    too_short = source_wave.sliced(0, 100)
    stereo = make_buffer(np.vstack((source_wave.samples, source_wave.samples)), 24000)
    for signal in (other_clock, too_short, stereo):
        with pytest.raises(AudioContractError):
            EvaluationPair("case1", ref, signal, "synthetic-reference")


def test_blind_ab_labels_are_seeded_and_avoid_quality_claims(
    source_wave, legal_synthetic
):
    test_case = pair("case1", "a1", source_wave, legal_synthetic)
    a = blind_pair_labels(test_case, seed=42)
    b = blind_pair_labels(test_case, seed=42)
    assert a == b
    assert {a["a"], a["b"]} == {"reference", "candidate"}
    assert a["listener_preference_recorded"] is False
    assert a["blinding_seed"] == 42
    with pytest.raises(AudioContractError):
        blind_pair_labels(test_case, seed=-1)


def test_language_and_caption_are_explicit_provenance_fields(
    source_wave, legal_synthetic
):
    with pytest.raises(AudioContractError):
        AudioReference("asset1", source_wave, "group1", "test",
                       "fr", "Rain", legal_synthetic)
    with pytest.raises(AudioContractError):
        AudioReference("asset1", source_wave, "group1", "test",
                       "hi", "", legal_synthetic)
    with pytest.raises(AudioContractError):
        AudioReference("invalid id", source_wave, "group1", "test",
                       "en", "Rain", legal_synthetic)


def test_snapshot_provenance_changes_after_candidate_pcm_edit(
    source_wave, legal_synthetic
):
    a = pair("case1", "asset1", source_wave, legal_synthetic)
    moved = source_wave.mutable_copy()
    moved[0, 200] += .1
    b = EvaluationPair("case1", a.reference,
                       source_wave.copy_with(moved), "synthetic-reference")
    assert EvaluationSet((a,)).fingerprint != EvaluationSet((b,)).fingerprint
    assert EvaluationSet((a,)).fingerprint == EvaluationSet((a,)).fingerprint
