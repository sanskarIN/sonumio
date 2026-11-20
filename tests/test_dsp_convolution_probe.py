"""Probe original float32 NumPy convolutions without changing training audio."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.dsp_convolution_probe import (
    PROTOCOL, SOURCES, SEEDS, build_probe, compare_probes,
    isolate_one, validate_probe,
)


@pytest.fixture(scope="module")
def report():
    return build_probe()


def test_original_convolutions_are_reconstructed_exactly():
    for source in SOURCES:
        for seed in (0, 101):
            row = isolate_one(source, seed)
            assert row["source"] == source
            assert row["seed"] == seed
            assert row["kernel_frames"] == {
                "rain": 180, "thunder": 140, "wind": 60, "water": 12,
            }[source]
            assert row["mode"] == ("full" if source == "rain" else "same")
            assert row["stages"]["input"]["frames"] == 36000
            assert row["stages"]["convolution"]["frames"] == 36000
            assert row["stages"]["scalar_fsum_reference"]["frames"] == 8
            assert 0 <= row["max_selected_fsum_difference"] < 2e-5


def test_complete_probe_keeps_only_hashes_and_has_all_sources(report):
    assert report["protocol"] == PROTOCOL
    assert len(report["recordings"]) == 16
    assert report["original_source_routines_unchanged"]
    assert report["source_sample_rate"] == 24000
    assert not report["audio_files_saved"]
    assert not report["model_weights_saved"]
    for source in SOURCES:
        entries = [x for x in report["recordings"] if x["source"] == source]
        assert {x["seed"] for x in entries} == set(SEEDS)
        for name in ("input", "kernel", "convolution", "source_layer",
                     "scalar_fsum_reference"):
            assert report["source_stage_aggregate"][source][name]["count"] == 4
    validate_probe(report)
    same = compare_probes(report, copy.deepcopy(report))
    assert same["all_exact_stages_match"]
    assert all(not x for x in same["differing_stages_by_source"].values())


def test_probe_can_distinguish_input_kernel_and_convolution_only_drift(report):
    changed = copy.deepcopy(report)
    target = next(x for x in changed["recordings"]
                  if x["source"] == "wind" and x["seed"] == 0)
    target["stages"]["convolution"]["sha256"] = "f" * 64
    from research.dsp_convolution_probe import _aggregate
    changed["source_stage_aggregate"] = _aggregate(changed["recordings"])
    validate_probe(changed)
    result = compare_probes(report, changed)
    assert result["differing_stages_by_source"]["wind"] == ["convolution"]
    assert result["differing_stages_by_source"]["rain"] == []
    assert not result["all_exact_stages_match"]
    assert result["cause_is_proven"] is False


def test_rejects_invalid_source_seed_and_forged_stage_reports(report):
    for source, seed in (("invalid", 0), ("rain", True), ("rain", 10000)):
        with pytest.raises(ValueError):
            isolate_one(source, seed)
    altered = copy.deepcopy(report)
    altered["model_weights_saved"] = True
    with pytest.raises(ValueError, match="Invalid original"):
        validate_probe(altered)
    altered = copy.deepcopy(report)
    altered["recordings"][0]["kernel_frames"] = 20
    with pytest.raises(ValueError, match="kernel"):
        validate_probe(altered)
    altered = copy.deepcopy(report)
    altered["recordings"][0]["stages"]["input"]["sha256"] = "z" * 64
    with pytest.raises(ValueError, match="hash"):
        validate_probe(altered)
    altered = copy.deepcopy(report)
    altered["recordings"][0]["max_selected_fsum_difference"] = float("nan")
    with pytest.raises(ValueError, match="scalar"):
        validate_probe(altered)
