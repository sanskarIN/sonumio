"""Canonical coefficient packing and independent quantized selection tests."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from research import quantized_transform_study as study
from research import transform_quantization as quant


@pytest.mark.parametrize("bits", quant.BIT_WIDTHS)
@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 3, 3), (2, 129, 16)])
def test_signed_payload_roundtrip(bits, shape):
    bound = (1 << (bits - 1)) - 1
    rng = np.random.default_rng(33)
    codes = rng.integers(-bound, bound + 1, size=shape, dtype=np.int16)
    codes.flat[0] = -bound
    codes.flat[-1] = bound
    payload = quant.pack_codes(codes, bits)
    assert len(payload) == quant.payload_size(shape, bits)
    np.testing.assert_array_equal(quant.unpack_codes(payload, shape, bits), codes)
    assert quant.pack_codes(quant.unpack_codes(payload, shape, bits), bits) == payload


def test_known_payload_vectors():
    codes = np.array([[[1, -1, 7, -7]]], dtype=np.int16)
    assert quant.pack_codes(codes, 4) == bytes.fromhex("f197")
    assert quant.pack_codes(codes, 8) == bytes.fromhex("01ff07f9")
    assert quant.pack_codes(np.array([[[0x123, -1]]], np.int16), 12) == bytes.fromhex("23f1ff")
    assert quant.pack_codes(np.array([[[0x1234, -1]]], np.int16), 16) == bytes.fromhex("3412ffff")


@pytest.mark.parametrize("bits", quant.BIT_WIDTHS)
def test_reserved_signed_minimum_and_length_rejection(bits):
    shape = (1, 1, 1)
    reserved = (1 << (bits - 1)).to_bytes(quant.payload_size(shape, bits), "little")
    with pytest.raises(ValueError, match="Reserved"):
        quant.unpack_codes(reserved, shape, bits)
    with pytest.raises(ValueError, match="reserved"):
        quant.pack_codes(np.array([[[-(1 << (bits - 1))]]], np.int16), bits)
    valid = quant.pack_codes(np.zeros(shape, np.int16), bits)
    for malformed in (valid + b"\x00", valid[:-1]):
        with pytest.raises(ValueError, match="length"):
            quant.unpack_codes(malformed, shape, bits)


@pytest.mark.parametrize("bits", [4, 12])
def test_noncanonical_padding_rejected(bits):
    shape = (1, 1, 1)
    payload = bytearray(quant.pack_codes(np.ones(shape, np.int16), bits))
    payload[-1] |= 0x80
    with pytest.raises(ValueError, match="padding"):
        quant.unpack_codes(bytes(payload), shape, bits)


@pytest.mark.parametrize("shape", [(1, 0, 1), (129, 1, 1), (1, 130, 1),
                                  (1, 1, 65), (True, 1, 1), [1, 1, 1]])
def test_shape_is_bounded_before_payload_decode(shape):
    with pytest.raises(ValueError, match="shape"):
        quant.unpack_codes(b"", shape, 4)


def test_quantizer_training_ranges_and_saturation_are_explicit():
    train = np.array([[[1.0, 0], [-2, 0]], [[0.5, 0], [1.5, 0]]])
    q = quant.CoefficientQuantizer.fit(train, 8)
    np.testing.assert_array_equal(q.ranges, [2.0, quant.MIN_RANGE])
    assert not q.ranges.flags.writeable
    heldout = np.array([[[4.0, 0], [-3.0, 0.01]]])
    restored = q.decode(q.encode(heldout), heldout.shape)
    np.testing.assert_allclose(restored, [[[2, 0], [-2, quant.MIN_RANGE]]])
    result = q.diagnose(heldout, restored)
    assert result["saturated_fraction"] == 0.75
    assert result["saturated_fraction_by_coordinate"] == [1, 0.5]
    np.testing.assert_array_equal(q.ranges, [2.0, quant.MIN_RANGE])


@pytest.mark.parametrize("bits", quant.BIT_WIDTHS)
def test_error_bound_without_saturation(bits):
    train = np.array([[[1.0, 2.0]], [[-1.0, -2.0]]])
    q = quant.CoefficientQuantizer.fit(train, bits)
    values = np.random.default_rng(18).uniform(-1, 1, (3, 17, 2)) * q.ranges
    decoded = q.decode(q.encode(values), values.shape)
    assert np.all(np.abs(decoded - values) <= q.ranges / (2 * q.bound) + 1e-12)


@pytest.mark.parametrize("values", [np.zeros((1, 2, 2)), np.zeros((2, 2, 2), np.int16),
                                    np.full((2, 2, 2), np.nan)])
def test_invalid_quantizer_fit(values):
    with pytest.raises(ValueError):
        quant.CoefficientQuantizer.fit(values, 8)


def candidate(name, rate, correct, represented, error, noise_accepted=0):
    return {"id": name, "validation": {
        "payload_bits_per_second": rate, "reconstructed_correct_accepted": correct,
        "noise_controls": {"reconstructed_accepted": noise_accepted},
        "by_category": {str(i): {"reconstructed_correct_accepted": int(i < represented)} for i in range(6)},
        "numerical": {"relative_squared_error": error},
    }}


def test_validation_selection_uses_cap_then_correct_coverage_error():
    rows = [
        candidate("expensive", 16001, 12, 6, 0),
        candidate("best-error", 8000, 8, 4, 0.01),
        candidate("best-coverage", 16000, 8, 5, 0.1),
        candidate("less-accurate", 8000, 7, 6, 0.001),
    ]
    assert study.choose_candidate(rows, 16000, source_gate_passed=True) == "best-coverage"
    rows[2]["validation"]["noise_controls"]["reconstructed_accepted"] = 1
    assert study.choose_candidate(rows, 16000, source_gate_passed=True) == "best-error"
    rows[2]["validation"]["noise_controls"]["reconstructed_accepted"] = 0
    assert study.choose_candidate(rows, 16000, source_gate_passed=False) == "less-accurate"
    assert study.choose_candidate(rows, 7999, source_gate_passed=True) is None
    # Confirmation outcomes have no effect on selection.
    rows[2]["confirmation"] = {"reconstructed_correct_accepted": 0}
    assert study.choose_candidate(rows, 16000, source_gate_passed=True) == "best-coverage"


def test_actual_study_selects_only_validation_winners():
    report = study.run_study(methods=("dct_ranked", "mdct_ranked"), budgets=(8, 16),
                             widths=(4, 8), caps=(8000, 16000))
    assert report["ranges_and_transform_fit_on_training_only"]
    assert not report["confirmation_used_for_selection"]
    assert not report["exact_split_overlap"]
    assert report["payload_rate_excludes_shared_scales_basis_and_container"]
    assert not report["production_bitrate_established"]
    assert not report["audio_weights_or_payload_files_saved"]
    assert report["panel_fingerprints"]["fit"]["recordings"] == 36
    assert report["panel_fingerprints"]["validation"]["recordings"] == 12
    assert report["panel_fingerprints"]["confirmation"]["recordings"] == 24
    assert len({x["sha256"] for x in report["panel_fingerprints"].values()}) == 3
    assert len(report["validation_candidates"]) == 8
    assert report["noise_panel_fingerprints"]["validation"]["recordings"] == 21
    assert report["noise_panel_fingerprints"]["confirmation"]["recordings"] == 28
    assert report["selection_uses_noise_scores"] == report["selection_uses_source_scores"]
    assert report["selection_uses_source_scores"] == (
        report["reference_gate_passed"] and report["validation_original_noise_accepted"] == 0
    )
    for row in report["validation_candidates"]:
        noise = row["validation"]["noise_controls"]
        assert noise["recordings"] == 21
        assert noise["original_accepted"] == report["validation_original_noise_accepted"]
    selected = {row["candidate_id"] for row in report["selection"]}
    assert None not in selected
    assert set(report["confirmation"]) == selected
    for row in report["selection"]:
        assert row["candidate_id"] == study.choose_candidate(
            report["validation_candidates"], row["payload_cap_bps"],
            source_gate_passed=report["selection_uses_source_scores"],
        )
    for row in report["confirmation"].values():
        assert row["quantized"]["recordings"] == 24
        assert row["same_transform_unquantized"]["recordings"] == 24
        for key in ("quantized", "same_transform_unquantized"):
            assert row[key]["noise_controls"]["recordings"] == 28
            assert row[key]["noise_controls"]["original_accepted"] == report["confirmation_original_noise_accepted"]
        assert row["confirmation_noise_screen_passed"] == (
            report["selection_uses_source_scores"] and row["quantized"]["noise_controls"]["zero_original_and_reconstruction_acceptances"]
        )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"methods": ("missing",)}, {"widths": (True,)},
                                    {"budgets": (8, 8)}, {"caps": (12345,)}])
def test_invalid_study_grid(kwargs):
    with pytest.raises(ValueError, match="grid"):
        study.run_study(**kwargs)
