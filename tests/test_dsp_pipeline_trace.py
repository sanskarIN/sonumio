"""Verify pipeline tracing is exact, complete and rejects forged stage reports."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.dsp_pipeline_trace import (
    PROTOCOL, STAGES, build_trace, compare_traces,
    digest_array, trace_one, validate_trace,
)
from research.latent_audio import FRAMES, render_seeded_waveform
from research.text_audio import SYNTHETIC_COMBINATIONS


@pytest.fixture(scope="module")
def trace():
    return build_trace()


def test_stage_trace_matches_original_unmodified_data_for_individual_and_combined():
    for sources, seed in (((0,), 0), ((1,), 101), ((3,), 1),
                          ((5,), 102), ((0, 2), 101), ((2, 5), 0)):
        report = trace_one(sources, seed)
        assert set(STAGES).issubset(report["stages"])
        assert report["seed"] == seed
        assert 400 <= report["offset"] <= 1600
        original = render_seeded_waveform(sources, seed)
        stage = report["stages"]["cropped_8k"]
        assert stage["frames"] == FRAMES
        assert stage["dtype"] == "float32"
        assert stage["sha256"] == hashlib.sha256(
            original.astype("<f4").tobytes()
        ).hexdigest()
        assert report["stages"]["normalized_24k"]["frames"] == 36000
        assert report["stages"]["fir_filtered_24k"]["dtype"] == "float64"
        assert report["stages"]["downsampled_8k"]["frames"] == 12000


def test_trace_complete_and_has_single_sound_source_aggregate(trace):
    assert trace["protocol"] == PROTOCOL
    assert trace["matches_original_renderer_exactly"]
    assert trace["source_recipe_unchanged"]
    assert not trace["audio_files_saved"] and not trace["model_weights_saved"]
    assert len(trace["records"]) == 84
    assert set(trace["single_source_aggregate"]) == {
        "rain", "thunder", "wind", "forest", "ocean", "ambient"
    }
    assert trace["stage_aggregate"]["cropped_8k"]["count"] == 84
    assert trace["stage_aggregate"]["time_grid"]["count"] == 84
    for source, stages in trace["single_source_aggregate"].items():
        assert stages["cropped_8k"]["count"] == 4, source
    assert all("samples" not in e and "waveform" not in e
               for e in trace["records"])
    validate_trace(trace)
    assert json.dumps(trace, allow_nan=False)


def test_identical_pipeline_report_has_zero_stage_differences(trace):
    compared = compare_traces(trace, copy.deepcopy(trace))
    assert compared["recordings_compared"] == 84
    assert compared["exact_stage_arrays_match"] is True
    assert compared["first_differing_stage_counts"] == {}
    assert compared["cause_unproven"] is True


def test_different_final_crop_does_not_alter_earlier_stage_identity(trace):
    tampered = copy.deepcopy(trace)
    record = tampered["records"][0]
    record["stages"]["cropped_8k"]["sha256"] = "a" * 64
    # Correct aggregate to simulate an otherwise internally consistent but
    # changed final waveform on an independently executed runner.
    from research.dsp_pipeline_trace import _aggregate
    tampered["stage_aggregate"] = _aggregate(tampered["records"])
    tampered["single_source_aggregate"]["rain"] = _aggregate(
        [e for e in tampered["records"] if e["sources"] == ["rain"]]
    )
    validate_trace(tampered)
    delta = compare_traces(trace, tampered)
    assert not delta["exact_stage_arrays_match"]
    assert delta["first_differing_stage_counts"] == {"cropped_8k": 1}


def test_trace_rejects_inconsistent_summary_duplicate_cases_and_audio_forgery(trace):
    forged = copy.deepcopy(trace)
    forged["stage_aggregate"]["cropped_8k"]["count"] = 83
    with pytest.raises(ValueError, match="aggregate"):
        validate_trace(forged)
    forged = copy.deepcopy(trace)
    forged["records"][1]["seed"] = forged["records"][0]["seed"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_trace(forged)
    forged = copy.deepcopy(trace)
    forged["audio_files_saved"] = True
    with pytest.raises(ValueError, match="unsafe"):
        validate_trace(forged)
    forged = copy.deepcopy(trace)
    forged["records"][0]["stages"]["normalized_24k"]["sha256"] = "z" * 64
    with pytest.raises(ValueError, match="stage digest"):
        validate_trace(forged)


@pytest.mark.parametrize("bad", [
    np.zeros(0, dtype=np.float32),
    np.zeros(18, dtype=np.int16),
    np.zeros((8, 4), dtype=np.float32),
    np.full(20, np.nan, dtype=np.float32),
])
def test_stage_hash_rejects_nonfinite_wrong_dimensions_and_dtypes(bad):
    with pytest.raises(ValueError):
        digest_array(bad)


def test_stage_signature_differentiates_native_float32_from_float64():
    source = np.linspace(-0.2, 0.2, 50, dtype=np.float32)
    a = digest_array(source)
    b = digest_array(source.astype(np.float64))
    assert a["dtype"] == "float32"
    assert b["dtype"] == "float64"
    assert a["sha256"] != b["sha256"]
    assert a["bucket_0.001_sha256"] == b["bucket_0.001_sha256"]
    with pytest.raises(ValueError):
        trace_one((0,), True)
    with pytest.raises(ValueError):
        trace_one((99,), 0)
