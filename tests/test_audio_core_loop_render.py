"""Exact circular overlap geometry, linked channels and bounded repetition."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer, AudioLayout
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.loop_analysis import LoopSearchPlan, search_loop
from sonumio.audio_core.loop_render import (
    LoopRenderPlan, render_loop, circular_diagnostics,
)


@pytest.mark.parametrize("law", ["linear", "cosine", "equal_power"])
@pytest.mark.parametrize("blend", [2, 5, 16, 32])
def test_exact_overlap_endpoints_and_circular_geometry(law, blend):
    source = make_buffer(np.linspace(-.4, .4, 100, dtype=np.float32), 8000)
    start, stop = 12, 76
    plan = LoopRenderPlan(start, stop, blend, law, repeats=3)
    result = render_loop(source, plan)
    n = plan.period_frames
    cycle = result.audio.samples[0, :n]
    assert result.audio.frames == 3*n
    assert cycle[0] == source.samples[0, start+blend]
    assert cycle[-1] == source.samples[0, start+blend-1]
    for i in (1, 2):
        np.testing.assert_array_equal(cycle, result.audio.samples[0, i*n:(i+1)*n])
    expected_step = abs(float(source.samples[0, start+blend])-float(source.samples[0, start+blend-1]))
    assert result.report.diagnostics.boundary_step_max == expected_step
    assert result.report.period_frames == stop-start-blend
    assert result.report.rotation_source_frames == blend


def test_no_blend_is_literal_source_slice_and_repeat():
    rng = np.random.default_rng(12)
    audio = make_buffer(rng.normal(size=(2, 1000)).astype(np.float32)*.1, 24000)
    output = render_loop(audio, LoopRenderPlan(100, 500, repeats=4))
    np.testing.assert_array_equal(output.audio.samples, np.tile(audio.samples[:, 100:500], (1, 4)))
    assert output.report.attenuation == 1


def test_equal_power_peak_policy_is_explicit():
    source = make_buffer(np.ones(64, dtype=np.float32)*.9, 8000)
    raw = render_loop(source, LoopRenderPlan(0, 64, 32, "equal_power"))
    assert raw.report.pre_limit_peak > 1
    limited = render_loop(source, LoopRenderPlan(0, 64, 32, "equal_power", peak_ceiling=.8))
    assert limited.report.attenuation < 1
    assert limited.report.diagnostics.sample_peak <= .8000001


def test_candidate_render_and_layout():
    t = np.arange(4096)
    wave = (.2*np.sin(2*np.pi*t/64)).astype(np.float32)
    layout = AudioLayout(2, "stereo", ("custom L", "custom R"))
    source = make_buffer(np.stack((wave,-wave)), 24000, layout=layout)
    plan = LoopSearchPlan(100, 1000, start_radius=4, stop_radius=16,
                          context_frames=64, blend_frames=16)
    candidate = search_loop(source, plan).best
    output = render_loop(source, LoopRenderPlan.from_candidate(candidate, repeats=2))
    assert output.audio.frames == candidate.period_frames*2
    assert output.audio.spec.layout == layout
    np.testing.assert_array_equal(output.audio.samples[0], -output.audio.samples[1])


def test_silent_circular_report_is_json_safe():
    import json
    x = np.zeros((2, 100), dtype=np.float32)
    report = circular_diagnostics(x)
    assert report.boundary_to_rms_ratio is None
    assert report.boundary_step_max == report.rms_internal_step == 0
    audio = make_buffer(x, 8000)
    json.dumps(render_loop(audio, LoopRenderPlan(0, 100, 20)).report.as_dict(), allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"start_frame": -1}, {"stop_frame": 10}, {"blend_frames": 1},
    {"blend_frames": 46}, {"law": "invalid"}, {"repeats": True},
    {"repeats": 0}, {"peak_ceiling": float("nan")},
])
def test_invalid_render_settings(kwargs):
    config = {"start_frame": 10, "stop_frame": 100}
    config.update(kwargs)
    with pytest.raises(AudioContractError):
        LoopRenderPlan(**config)


def test_render_preflight_bounds(monkeypatch):
    import sonumio.audio_core.loop_render as module
    source = make_buffer(np.zeros(100, dtype=np.float32), 8000)
    with pytest.raises(AudioContractError):
        render_loop(source, LoopRenderPlan(0, 101))
    monkeypatch.setattr(module, "MAX_LOOP_WORK_BYTES", 1)
    with pytest.raises(AudioLimitError, match="working-memory"):
        render_loop(source, LoopRenderPlan(0, 100, repeats=2))
