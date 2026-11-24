import json

import numpy as np
import pytest

pytest.importorskip("torch")

from research import transform_audio_codec as codec
from research import transform_codec_study as study
from research.latent_audio import FRAMES


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(21)
    waves = rng.normal(0, 0.1, (12, FRAMES))
    labels = np.repeat(np.arange(6), 2)
    return codec.TransformFit.fit(waves, labels)


@pytest.mark.parametrize("family", ["dct", "mdct"])
def test_full_transform_roundtrip(family):
    rng = np.random.default_rng(7)
    waves = rng.normal(0, 0.1, (3, FRAMES))
    coeff = codec.analysis(waves, family)
    assert coeff.shape == (3, 128 + (family == "mdct"), 64)
    np.testing.assert_allclose(codec.synthesis(coeff, family), waves, atol=1e-12, rtol=0)


@pytest.mark.parametrize("index", [0, 63, 64, 4096, FRAMES - 1])
def test_lapped_transform_preserves_boundary_impulses(index):
    wave = np.zeros((1, FRAMES), np.float64)
    wave[0, index] = 1
    np.testing.assert_allclose(codec.synthesis(codec.analysis(wave, "mdct"), "mdct"),
                               wave, atol=1e-12, rtol=0)


def test_bases_and_rankings_are_valid(fitted):
    matrix = codec.dct_basis()
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(64), atol=1e-12)
    np.testing.assert_allclose(fitted.pca_basis.T @ fitted.pca_basis, np.eye(64), atol=1e-12)
    assert sorted(fitted.dct_order) == sorted(fitted.mdct_order) == list(range(64))
    assert np.all(np.diff(fitted.pca_eigenvalues) <= 0)
    assert not fitted.pca_basis.flags.writeable


@pytest.mark.parametrize("method", codec.METHODS)
def test_full_budget_reconstruction_and_nested_error(fitted, method):
    wave = np.random.default_rng(9).normal(0, 0.1, (2, FRAMES))
    errors = []
    for budget in codec.BUDGETS:
        retained = fitted.encode(wave, method, budget)
        desc = fitted.describe(method, budget)
        assert retained.size == len(wave) * desc["retained_scalar_values"]
        assert desc["scalar_compression_ratio"] == FRAMES / desc["retained_scalar_values"]
        decoded = fitted.decode(retained, method, budget)
        errors.append(float(np.mean((decoded - wave) ** 2)))
    assert errors[-1] < 1e-24
    # DCT/PCA are orthogonal projections. Padded MDCT synthesis includes
    # boundary cropping, so strict monotonicity is not assumed for arbitrary data.
    if not method.startswith("mdct"):
        assert all(a >= b for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize("wave", [
    np.zeros((1, FRAMES - 1)), np.zeros((0, FRAMES)),
    np.zeros((1, FRAMES), dtype=np.int16), np.full((1, FRAMES), np.nan),
])
def test_invalid_waveforms(wave):
    with pytest.raises(ValueError):
        codec.analysis(wave, "dct")


def test_fit_requires_recordings_in_every_category():
    with pytest.raises(ValueError, match="all six"):
        codec.TransformFit.fit(np.zeros((12, FRAMES)), np.zeros(12, dtype=np.int64))
    with pytest.raises(ValueError, match="At least two"):
        codec.TransformFit.fit(np.zeros((6, FRAMES)), np.arange(6))


def test_fit_arrays_not_changed_by_heldout_encoding(fitted):
    before = fitted.pca_basis.copy()
    heldout = np.random.default_rng(23).normal(0.2, 0.1, (2, FRAMES))
    fitted.decode(fitted.encode(heldout, "pca", 16), "pca", 16)
    np.testing.assert_array_equal(fitted.pca_basis, before)


def test_silent_reference_metrics_are_finite_json():
    row = codec.reconstruction_metrics(np.zeros((1, FRAMES)), np.zeros((1, FRAMES)))[0]
    assert row["relative_squared_error"] is None
    assert row["snr_db_with_error_floor"] is None
    assert row["waveform_cosine"] is None
    json.dumps(row, allow_nan=False)


def test_actual_study_all_methods_with_full_and_fourfold_budgets():
    result = study.run_study(budgets=(16, 64))
    assert result["fit_recordings"] == 36
    assert result["validation_recordings"] == 12
    assert result["reference_recordings"] == 18
    assert not result["exact_split_overlap"]
    assert result["fit_only_basis_and_coordinate_selection"]
    assert not result["quantized_bitrate_established"]
    assert not result["audio_or_weights_saved"]
    assert len(result["experiments"]) == 10
    for row in result["experiments"]:
        assert row["validation"]["recordings"] == 12
        assert row["probe_holdout"]["recordings"] == 18
        if row["coefficients_per_frame"] == 64:
            assert row["probe_holdout"]["numerical"]["mse"] < 1e-12
            assert row["probe_holdout"]["reconstructed_correct_accepted"] == row["probe_holdout"]["original_correct_accepted"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("options", [
    {"budgets": (16, 16)}, {"budgets": (True,)}, {"methods": ("unknown",)},
])
def test_invalid_grid(options):
    with pytest.raises(ValueError, match="study grid"):
        study.run_study(**options)
