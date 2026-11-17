"""End-to-end research-only full convolution byte reproducibility controls."""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("torch")

from research.full_convolution_candidate import (
    FRAMES, PROTOCOL, SEEDS, SOURCES, _aggregate,
    build_full_probe, compare_full_probes, full_scalar_convolution,
    probe_one_full, validate_full_probe,
)
from research.stable_convolution_candidate import selected_scalar_convolution


@pytest.mark.parametrize("taps", [1, 2, 12, 60, 140, 180])
@pytest.mark.parametrize("mode", ["same", "full"])
def test_full_frame_chunking_matches_unbroken_scalar_reference(taps, mode):
    rng = np.random.default_rng(91)
    signal = rng.normal(size=301).astype(np.float32)
    kernel = rng.uniform(-0.2, 0.2, size=taps).astype(np.float32)
    frames = len(signal) if mode == "same" else len(signal) + taps - 1
    actual = full_scalar_convolution(signal, kernel, mode=mode)
    expected = selected_scalar_convolution(
        signal, kernel, tuple(range(frames)), mode=mode
    )
    np.testing.assert_array_equal(actual, expected)


def test_chunk_boundary_does_not_change_output_bits():
    signal = np.asarray(
        [((i * 17) % 173 - 86) / 127.0 for i in range(2305)],
        dtype=np.float32,
    )
    kernel = np.asarray(
        [((i * 13) % 37 - 18) / 41.0 for i in range(140)],
        dtype=np.float32,
    )
    full = full_scalar_convolution(signal, kernel, mode="same")
    for start in (0, 1023, 1024, 2047, 2048, 2304):
        expected = selected_scalar_convolution(
            signal, kernel, (start,), mode="same"
        )
        assert full[start] == expected[0]


@pytest.mark.parametrize("source", SOURCES)
def test_complete_original_source_convolution_is_preserved(source):
    row = probe_one_full(source, SEEDS[0])
    assert row["source"] == source
    assert row["legacy_cropped_frames"] == FRAMES
    assert row["full_output_frames"] >= FRAMES
    assert row["max_full_output_absolute_error"] <= 2e-5
    assert row["rms_full_output_absolute_error"] <= row["max_full_output_absolute_error"]
    for key in ("input_sha256", "kernel_sha256", "candidate_full_output_sha256"):
        assert len(row[key]) == 64


@pytest.fixture(scope="module")
def full_report():
    return build_full_probe()


def test_all_sixteen_full_source_replays_are_hash_only(full_report):
    validate_full_probe(full_report)
    assert full_report["protocol"] == PROTOCOL
    assert full_report["original_dsp_unchanged"]
    assert full_report["scalar_candidate_only"]
    assert full_report["full_convolution_evaluated"]
    assert len(full_report["records"]) == 16
    assert not full_report["audio_files_saved"]
    assert not full_report["model_weights_saved"]
    same = compare_full_probes(full_report, copy.deepcopy(full_report))
    assert same["cases"] == 16
    assert same["same_inputs_and_kernels"]
    assert same["candidate_full_hashes_equal"]
    assert same["original_full_hashes_equal"]


def test_full_frame_validator_and_comparison_detect_tampering(full_report):
    changed = copy.deepcopy(full_report)
    changed["model_weights_saved"] = True
    with pytest.raises(ValueError, match="metadata"):
        validate_full_probe(changed)
    changed = copy.deepcopy(full_report)
    changed["records"][0]["full_output_frames"] += 1
    with pytest.raises(ValueError, match="dimensions"):
        validate_full_probe(changed)
    changed = copy.deepcopy(full_report)
    changed["records"][0]["max_full_output_absolute_error"] = float("nan")
    with pytest.raises(ValueError, match="metrics"):
        validate_full_probe(changed)
    changed = copy.deepcopy(full_report)
    changed["records"][0]["candidate_full_output_sha256"] = "f" * 64
    changed["source_aggregate"] = _aggregate(changed["records"])
    validate_full_probe(changed)
    status = compare_full_probes(full_report, changed)
    assert status["same_inputs_and_kernels"]
    assert status["original_full_hashes_equal"]
    assert not status["candidate_full_hashes_equal"]


def test_full_frame_input_guards_prevent_unbounded_allocation():
    x = np.ones(50001, dtype=np.float32)
    k = np.ones(5, dtype=np.float32)
    with pytest.raises(ValueError):
        full_scalar_convolution(x, k, mode="same")
    with pytest.raises(ValueError):
        full_scalar_convolution(np.ones(30, dtype=np.float32),
                                np.ones(31, dtype=np.float32), mode="full")
