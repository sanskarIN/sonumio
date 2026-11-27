"""Sample-peak safety, finite lookahead and linked gain invariants."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer, AudioLayout
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.lookahead_limiter import (
    LookaheadSettings, limiter_envelope, limit_audio,
)


@pytest.mark.parametrize("lookahead", [0,1,5,10])
@pytest.mark.parametrize("release", [0,1,50])
def test_envelope_never_exceeds_required_gain(lookahead,release):
    rng=np.random.default_rng(4)
    peaks=rng.uniform(0,3,3001)
    settings=LookaheadSettings(-3,lookahead,release)
    gain=limiter_envelope(peaks,8000,settings)
    assert np.all(gain<=1) and np.all(gain>0)
    assert np.all(gain*peaks<=settings.ceiling+1e-14)


def test_single_peak_anticipation_bounded_by_declared_lookahead():
    peaks=np.full(300,.1)
    peaks[150]=2
    cfg=LookaheadSettings(0,5,0)
    gain=limiter_envelope(peaks,8000,cfg)
    n=cfg.lookahead_frames(8000)
    np.testing.assert_allclose(gain[:151-n],np.ones(151-n),atol=1e-14,rtol=0)
    assert np.all(np.diff(gain[150-n:151])<=0)
    assert gain[150]==.5
    np.testing.assert_array_equal(gain[151:],np.ones(149))


def test_release_extends_attenuation_without_violating_sample_cap():
    peaks=np.full(1000,.1)
    peaks[400]=2
    cfg=LookaheadSettings(0,0,10)
    gain=limiter_envelope(peaks,8000,cfg)
    assert gain[400]==.5
    assert .5<gain[401]<1.
    assert np.all(np.diff(gain[400:])>=0)
    assert gain[-1]>.99


def test_channel_linked_no_clipping_and_geometry():
    source=np.vstack((np.linspace(-2,2,2000),np.linspace(-1,1,2000))).astype(np.float32)
    layout=AudioLayout(2,"stereo",("A","B"))
    audio=make_buffer(source,24000,layout=layout)
    cfg=LookaheadSettings(-1,5,50)
    out=limit_audio(audio,cfg)
    assert out.audio.spec==audio.spec
    np.testing.assert_array_equal(out.audio.samples[0],out.audio.samples[1]*2)
    assert out.report.output_peak<=cfg.ceiling
    assert out.report.maximum_reduction_db>0
    assert out.report.attenuated_frames>0
    assert not out.report.as_dict()["hard_clipping_applied"]
    np.testing.assert_array_equal(audio.samples,source)


@pytest.mark.parametrize("level",[0.,.1,.5])
def test_unlimited_input_is_exact_identity(level):
    audio=make_buffer(np.full(201,level,dtype=np.float32),8000)
    out=limit_audio(audio)
    np.testing.assert_array_equal(out.audio.samples,audio.samples)
    assert out.report.minimum_gain==1.
    assert out.report.attenuated_frames==0


@pytest.mark.parametrize("kwargs",[
    {"ceiling_dbfs":True},{"ceiling_dbfs":1},{"ceiling_dbfs":float("nan")},
    {"lookahead_ms":-1},{"lookahead_ms":101},{"release_ms":True},{"release_ms":2001},
])
def test_invalid_settings(kwargs):
    with pytest.raises(AudioContractError):
        LookaheadSettings(**kwargs)


@pytest.mark.parametrize("peaks",[
    np.array([],dtype=np.float64),np.array([-1.]),np.array([np.nan]),
    np.ones((2,3)),np.array([1],dtype=np.int64),
])
def test_invalid_peak_vectors(peaks):
    with pytest.raises(AudioContractError):
        limiter_envelope(peaks,8000)


def test_limiter_preflight(monkeypatch):
    import sonumio.audio_core.lookahead_limiter as module
    monkeypatch.setattr(module,"MAX_LIMITER_BYTES",100)
    audio=make_buffer(np.ones(100,dtype=np.float32),8000)
    with pytest.raises(AudioLimitError,match="working-memory"):
        limit_audio(audio)
