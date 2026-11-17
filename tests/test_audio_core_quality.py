"""Basic signal-health guards without confusing them with perceptual quality."""
from __future__ import annotations

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.quality import (
    QualityFinding, QualityLimits, QualityReport,
    measure_quality, compare_quality_reports, waveform_integrity_report,
)
from sonumio.audio_core.errors import AudioContractError


@pytest.fixture
def healthy():
    n = 2400
    times = np.arange(n, dtype=np.float64) / 24000
    envelope = np.sin(np.linspace(0, np.pi, n, dtype=np.float64)) ** 2
    values = (.3 * np.sin(2 * np.pi * 440 * times) * envelope).astype(np.float32)
    return make_buffer(values, 24000)


def test_healthy_sine_passes_basic_signal_checks(healthy):
    report = measure_quality(healthy)
    assert report.passed
    assert report.sample_rate == 24000
    assert report.frames == healthy.frames
    assert report.channels == 1
    assert report.peak_amplitude <= .3 + 1e-5
    assert report.rms_amplitude > 0
    assert report.clipping_fraction == 0
    assert report.boundary_jump < .01
    assert report.dc_offset_absolute < .05
    summary = report.as_dict()
    assert summary["passed"] is True
    assert summary["perceptual_audio_quality_measured"] is False
    assert summary["text_prompt_adherence_measured"] is False
    assert summary["listening_test_performed"] is False


def test_overrange_signal_flags_explicit_peak_and_clip_limits(healthy):
    extra = healthy.copy_with(healthy.samples * 6)
    report = measure_quality(extra)
    codes = {f.code for f in report.findings}
    assert "peak" in codes
    assert "clip_fraction" in codes
    assert report.passed is False
    assert all(f.severity == "warning" for f in report.findings)


def test_dc_offset_detects_biased_pcm(healthy):
    biased = healthy.copy_with(healthy.samples + .25)
    report = measure_quality(biased)
    assert "dc_offset" in {x.code for x in report.findings}
    assert report.dc_offset_absolute > .2
    assert report.as_dict()["passed"] is False


def test_boundary_jump_warns_on_nonfaded_tone(healthy):
    new = healthy.mutable_copy()
    new[0, 0] = .8
    new[0, -1] = -.8
    changed = healthy.copy_with(new)
    result = measure_quality(changed)
    assert result.boundary_jump == pytest.approx(.8)
    assert "boundary_jump" in {x.code for x in result.findings}


def test_channel_imbalance_is_disclosed_not_mistaken_for_voice_quality(healthy):
    values = np.vstack((healthy.samples, healthy.samples * .01))
    stereo = make_buffer(values.astype(np.float32), 24000)
    findings = measure_quality(stereo)
    assert findings.channel_imbalance_db > 35
    assert "channel_imbalance" in {x.code for x in findings.findings}


def test_silence_is_warning_and_reports_no_inferred_perceptual_quality():
    silent = make_buffer(np.zeros(4096, dtype=np.float32), 24000)
    result = measure_quality(silent)
    assert result.passed is False
    assert "activity" in {x.code for x in result.findings}
    assert result.clipping_fraction == 0
    assert result.peak_amplitude == 0
    assert result.rms_amplitude == 0
    assert result.channel_imbalance_db == 0


def test_quality_report_differences_are_causally_scoped(healthy):
    before = measure_quality(healthy)
    clipped = healthy.copy_with(healthy.samples * 7)
    after = measure_quality(clipped)
    comparison = compare_quality_reports(before, after)
    assert comparison["source_frames"] == healthy.frames
    assert "clip_fraction" in comparison["new_failures"]
    assert comparison["peak_delta"] > 0
    assert comparison["human_perceptual_improvement_inferred"] is False
    restored = compare_quality_reports(after, before)
    assert "clip_fraction" in restored["resolved_failures"]


def test_integrity_sha_report_is_deterministic_and_explicit(healthy):
    a = waveform_integrity_report(healthy)
    b = waveform_integrity_report(healthy)
    assert a == b
    assert len(a["pcm_sha256"]) == 64
    assert a["numerically_finite"] is True
    assert a["model_weights_loaded"] is False
    assert a["does_not_verify_rights_or_consent"] is True


def test_quality_limits_reject_invalid_nonfinite_and_wrong_ranges():
    with pytest.raises(AudioContractError):
        QualityLimits(max_peak=0)
    with pytest.raises(AudioContractError):
        QualityLimits(max_clipping_fraction=2)
    with pytest.raises(AudioContractError):
        QualityLimits(max_dc_absolute=float("nan"))
    with pytest.raises(AudioContractError):
        QualityLimits(activity_floor_dbfs=10)


def test_quality_comparison_requires_same_audio_geometry(healthy):
    first = measure_quality(healthy)
    longer = make_buffer(np.zeros(4096, dtype=np.float32), 24000)
    second = measure_quality(longer)
    with pytest.raises(AudioContractError):
        compare_quality_reports(first, second)
