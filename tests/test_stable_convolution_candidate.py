"""Research-only tests: scalar convolution candidate; original DSP is unchanged."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

pytest.importorskip("torch")

from research.stable_convolution_candidate import (
    FRAMES, PROTOCOL, SAMPLE_COUNT, SEEDS, SOURCES,
    _aggregate, _positions, _source_inputs,
    build_probe, compare_probes, probe_one,
    selected_scalar_convolution, validate_probe,
)


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 12, 60, 140, 180])
@pytest.mark.parametrize("mode", ["same", "full"])
def test_scalar_convolution_matches_double_precision_reference(kernel_size, mode):
    rng = np.random.default_rng(17 + kernel_size)
    samples = rng.normal(size=233).astype(np.float32)
    kernel = rng.uniform(-0.5, 0.5, size=kernel_size).astype(np.float32)
    size = len(samples) if mode == "same" else len(samples) + len(kernel) - 1
    positions = tuple(range(size))
    actual = selected_scalar_convolution(samples, kernel, positions, mode=mode)
    reference = np.convolve(
        samples.astype(np.float64), kernel.astype(np.float64), mode=mode
    ).astype(np.float32)
    assert actual.dtype == np.float32
    assert actual.shape == (size,)
    # The reference uses vectorized accumulation and may round differently.
    np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-6)


def test_single_tap_kernel_is_exact_and_full_mode_supports_tail():
    samples = np.asarray([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    kernel = np.asarray([0.25], dtype=np.float32)
    for mode in ("same", "full"):
        actual = selected_scalar_convolution(
            samples, kernel, (0, 1, 2, 3), mode=mode
        )
        np.testing.assert_array_equal(actual, samples * 0.25)
    kernel2 = np.asarray([1.0, -2.0], dtype=np.float32)
    actual = selected_scalar_convolution(
        samples, kernel2, (0, 1, 2, 3, 4), mode="full"
    )
    np.testing.assert_array_equal(actual, np.convolve(samples, kernel2, "full"))


def test_selected_positions_are_immutable_unique_and_cover_boundaries():
    points = _positions(FRAMES)
    assert isinstance(points, tuple)
    assert points == tuple(sorted(set(points)))
    assert len(points) >= SAMPLE_COUNT
    assert points[0] == 0 and points[-1] == FRAMES - 1
    assert 139 in points and FRAMES // 2 in points


@pytest.mark.parametrize("mutation", [
    lambda x, k, p: (x.astype(np.float64), k, p, "same"),
    lambda x, k, p: (x, k.astype(np.float64), p, "same"),
    lambda x, k, p: (x.reshape(2, 4), k, p, "same"),
    lambda x, k, p: (x, np.asarray([float("nan")], dtype="float32"), p, "same"),
    lambda x, k, p: (x, k, (True,), "same"),
    lambda x, k, p: (x, k, (0, 0), "same"),
    lambda x, k, p: (x, k, (-1,), "same"),
    lambda x, k, p: (x, k, (8,), "same"),
    lambda x, k, p: (x, k, [0], "same"),
    lambda x, k, p: (x, k, (0,), "valid"),
])
def test_untrusted_numerical_shapes_and_positions_are_rejected(mutation):
    x = np.arange(8, dtype=np.float32)
    k = np.asarray([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError):
        a, b, positions, mode = mutation(x, k, (0,))
        selected_scalar_convolution(a, b, positions, mode=mode)


def test_boundaries_and_oversize_inputs_rejected():
    x = np.ones(8, dtype=np.float32)
    k = np.ones(9, dtype=np.float32)
    with pytest.raises(ValueError):
        selected_scalar_convolution(x, k, (0,), mode="full")
    with pytest.raises(ValueError):
        selected_scalar_convolution(np.ones(50001, dtype=np.float32),
                                    x, (0,), mode="same")
    with pytest.raises(ValueError):
        selected_scalar_convolution(x, np.asarray([], dtype=np.float32),
                                    (0,), mode="same")


@pytest.mark.parametrize("source", SOURCES)
def test_recreated_source_inputs_are_original_and_numeric_close(source):
    record = probe_one(source, SEEDS[0])
    samples, kernel, mode = _source_inputs(source, SEEDS[0])
    assert samples.dtype == kernel.dtype == np.float32
    assert len(samples) == FRAMES
    assert mode == record["mode"]
    assert record["sampled_positions"] == len(_positions(FRAMES))
    assert math.isfinite(record["max_selected_absolute_error"])
    assert 0 <= record["max_selected_absolute_error"] <= 2e-5


@pytest.fixture(scope="module")
def report():
    return build_probe()


def test_full_candidate_probe_is_digest_only_and_reproducible(report):
    validate_probe(report)
    assert report["protocol"] == PROTOCOL
    assert report["original_dsp_unchanged"]
    assert report["candidate_is_experimental_only"]
    assert len(report["records"]) == len(SOURCES) * len(SEEDS)
    assert report["audio_files_saved"] is False
    assert report["weights_saved"] is False
    assert all(row["sampled_positions"] == len(_positions(FRAMES))
               for row in report["records"])
    summary = compare_probes(report, copy.deepcopy(report))
    assert summary["cases_compared"] == 16
    assert summary["same_input_and_kernel_hashes"]
    assert summary["candidate_sampled_hashes_equal"]
    assert summary["original_sampled_hashes_equal"]
    assert summary["not_a_proof_of_full_audio_reproducibility"]


def test_mutations_are_rejected_and_comparator_distinguishes_host_drift(report):
    changed = copy.deepcopy(report)
    changed["weights_saved"] = True
    with pytest.raises(ValueError, match="metadata"):
        validate_probe(changed)
    changed = copy.deepcopy(report)
    changed["records"][0]["seed"] = changed["records"][1]["seed"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_probe(changed)
    changed = copy.deepcopy(report)
    changed["records"][0]["max_selected_absolute_error"] = float("nan")
    with pytest.raises(ValueError, match="numeric"):
        validate_probe(changed)
    changed = copy.deepcopy(report)
    changed["records"][0]["candidate_selected_sha256"] = "a" * 64
    changed["source_aggregate"] = _aggregate(changed["records"])
    validate_probe(changed)
    result = compare_probes(report, changed)
    assert result["same_input_and_kernel_hashes"]
    assert result["original_sampled_hashes_equal"]
    assert not result["candidate_sampled_hashes_equal"]
