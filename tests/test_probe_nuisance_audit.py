"""Validate paired nuisance controls and frozen probe reporting."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from research.latent_audio import FRAMES
from research.probe_nuisance_transforms import (
    CONTROLS, INVARIANCE, DISTORTION, TEMPORAL, Transform,
    apply_transform, waveform_diagnostics,
)
from research.probe_nuisance_audit import run_audit, _summarize


def sample():
    t = np.arange(FRAMES, dtype=np.float64) / 8000
    return (.3*np.sin(2*np.pi*300*t)+.4*np.sin(2*np.pi*3300*t)).astype(np.float32)


@pytest.mark.parametrize("control", CONTROLS, ids=lambda c:c.name)
def test_all_controls_are_deterministic_and_do_not_mutate_inputs(control):
    wave = sample()
    before = wave.copy()
    a = apply_transform(wave,control)
    b = apply_transform(wave,control)
    np.testing.assert_array_equal(wave,before)
    np.testing.assert_array_equal(a,b)
    assert a.shape == (FRAMES,)
    assert a.dtype == np.float32
    assert not np.shares_memory(wave,a)
    assert np.max(np.abs(a)) <= 1.0001
    json.dumps(waveform_diagnostics(wave,a),allow_nan=False)


def test_exact_gain_polarity_shift_and_clipping():
    wave = sample()
    get = {t.name:t for t in CONTROLS}
    np.testing.assert_array_equal(apply_transform(wave,get["gain_half"]),wave*.5)
    np.testing.assert_array_equal(apply_transform(wave,get["polarity_flip"]),-wave)
    np.testing.assert_array_equal(apply_transform(wave,get["shift_128"]),np.roll(wave,128))
    np.testing.assert_array_equal(apply_transform(wave,get["clip_half"]),np.clip(wave,-.5,.5))


@pytest.mark.parametrize("bits",[4,8,16])
def test_quantization_rounding_error_bound(bits):
    wave = sample()
    result = apply_transform(wave,Transform("grid","quantize",bits,DISTORTION))
    bound = (1<<(bits-1))-1
    assert np.max(np.abs(result-wave)) <= .5/bound+1e-7
    assert np.max(np.abs(result*bound-np.rint(result*bound))) <= .003


def test_lowpass_reduces_high_tone_without_peak_renormalization():
    wave = sample()
    result = apply_transform(wave,Transform("low","lowpass",1500,DISTORTION))
    t = np.arange(FRAMES,dtype=np.float64)/8000
    interior = slice(128,-128)
    def amplitude(x,f):
        return abs(2*np.mean(x[interior]*np.exp(-2j*np.pi*f*t[interior])))
    assert amplitude(result,3300) < .01*amplitude(wave,3300)
    assert amplitude(result,300) == pytest.approx(amplitude(wave,300),rel=.02)


@pytest.mark.parametrize("kind,value,family",[
    ("gain",0,INVARIANCE),("gain",2,INVARIANCE),("gain",.5,DISTORTION),
    ("polarity",1,INVARIANCE),("circular_shift",True,TEMPORAL),
    ("circular_shift",FRAMES,TEMPORAL),("quantize",3,DISTORTION),
    ("quantize",8.0,DISTORTION),("lowpass",4000,DISTORTION),
    ("clip",float("nan"),DISTORTION),("unknown",1,DISTORTION),
])
def test_invalid_transform(kind,value,family):
    with pytest.raises(ValueError):
        Transform("bad",kind,value,family)


@pytest.mark.parametrize("wave",[
    np.zeros(FRAMES-1,np.float32),np.zeros(FRAMES,np.float64),
    np.full(FRAMES,np.nan,np.float32),np.full(FRAMES,2,np.float32),
])
def test_invalid_wave(wave):
    with pytest.raises(ValueError):
        apply_transform(wave,CONTROLS[0])


def test_silence_metrics_are_finite_or_explicitly_undefined():
    z = np.zeros(FRAMES,np.float32)
    row = waveform_diagnostics(z,z)
    assert row["exactly_equal"] and row["transformed_silent"]
    assert row["relative_squared_error"] is None
    assert row["waveform_cosine"] is None
    json.dumps(row,allow_nan=False)


def test_transition_accounting_distinguishes_rejection_and_label_change():
    def row(before,after,label="rain"):
        return {"label":label,"original":{"closest":before,"accepted":before is not None,"prediction":before},
                "transformed":{"closest":after,"accepted":after is not None,"prediction":after},
                "waveform":{"sample_mae":0.0,"transformed_rms":.2,"transformed_silent":False}}
    rows = [row(None,"rain"),row("rain",None),row("wind","rain"),row("rain","rain"),row(None,None)]
    summary = _summarize(rows,source=True)
    assert summary["new_acceptances"] == summary["removed_acceptances"] == 1
    assert summary["changed_accepted_category"] == 1
    assert summary["correct_acceptances_lost"] == 1
    assert summary["correct_acceptances_gained"] == 2
    assert sum(x["count"] for x in summary["transitions"]) == 5
    assert _summarize(rows,source=False)["new_false_acceptances"] == 1


def test_real_fresh_audit_deterministic_and_not_refit():
    names = ("gain_half","polarity_flip","shift_128","quantize_4bit")
    report = run_audit(control_names=names)
    assert report == run_audit(control_names=names)
    assert not report["exact_reference_overlap"]
    assert not report["exact_training_validation_overlap"]
    assert not report["training_or_calibration_changed"]
    assert not report["audio_or_weights_saved"]
    assert report["fresh_source_recordings"] == 24
    assert report["fresh_noise_recordings"] == 28
    assert set(report["control_results"]) == set(names)
    for name,panel in report["control_results"].items():
        for domain,count in (("source",24),("noise",28)):
            s = panel[domain]["aggregate"]
            assert s["cases"] == count
            assert s["transformed_accepted"] == s["original_accepted"]+s["new_acceptances"]-s["removed_acceptances"]
            assert sum(x["count"] for x in s["transitions"]) == count
            assert sum(x["cases"] for x in panel[domain]["by_label"].values()) == count
        if panel["family"] == INVARIANCE:
            assert isinstance(panel["exact_decision_invariance"],bool)
        else:
            assert panel["exact_decision_invariance"] is None
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize("names",[(),("missing",),("gain_half","gain_half"),["gain_half"]])
def test_invalid_control_selection(names):
    with pytest.raises(ValueError,match="selection"):
        run_audit(control_names=names)
