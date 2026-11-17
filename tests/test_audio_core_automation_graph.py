"""Data-only graph execution, frame-accurate automation and security boundaries."""
from __future__ import annotations

from fractions import Fraction
import json

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.automation import (
    AutomationCurve, Keyframe, Tempo, apply_gain_automation, envelope_adsr,
)
from sonumio.audio_core.graph import (
    GraphNode, SignalGraph, execute_graph, graph_safety_report,
)
from sonumio.audio_core.hashing import digest_audio
from sonumio.audio_core.errors import AudioContractError, AudioLimitError


@pytest.fixture
def source():
    return make_buffer(np.ones(256, dtype=np.float32) * .5, 24000)


def test_automation_keyframe_linear_step_cosine_and_smoothstep():
    for curve in ("linear", "step", "cosine", "smoothstep"):
        keys = (Keyframe(0, 0, curve), Keyframe(100, 1))
        series = AutomationCurve(keys).render(101)
        assert series.shape == (101,)
        assert series[0] == 0
        assert series[-1] == 1
        assert np.isfinite(series).all()
        if curve == "step":
            assert np.all(series[:-1] == 0)
        else:
            assert series[50] == pytest.approx(.5, abs=1e-5)


def test_exponential_keyframe_requires_compatible_nonzero_signs():
    stable = AutomationCurve((Keyframe(0, 1, "exponential"),
                              Keyframe(100, 4)))
    assert stable.value_at(50) == pytest.approx(2)
    np.testing.assert_allclose(stable.render(101)[50], 2, atol=1e-6)
    bad = AutomationCurve((Keyframe(0, -1, "exponential"), Keyframe(100, 4)))
    with pytest.raises(AudioContractError):
        bad.render(101)


def test_automation_boundaries_and_transforms_are_immutable():
    a = AutomationCurve((Keyframe(20, 3), Keyframe(40, 9)),
                        before="zero", after="zero")
    assert a.value_at(0) == 0
    assert a.value_at(50) == 0
    assert a.value_at(20) == 3
    assert a.value_at(40) == 9
    shifted = a.shifted(10)
    assert shifted.start == 30
    assert shifted.stop == 50
    assert a.start == 20
    scaled = a.scaled(.5, 1)
    assert scaled.keys[0].value == pytest.approx(2.5)
    clipped = scaled.clipped(3, 5)
    assert clipped.keys[0].value == 3
    with pytest.raises(AudioContractError):
        a.shifted(-30)


def test_audio_gain_automation_ramp_is_sample_exact(source):
    curve = AutomationCurve((Keyframe(0, 0), Keyframe(255, 1)))
    result = apply_gain_automation(source, curve)
    assert result.samples[0, 0] == 0
    assert result.samples[0, -1] == pytest.approx(.5)
    assert result.samples[0, 128] == pytest.approx(.5 * 128 / 255, abs=1e-6)
    assert np.all(source.samples == .5)


def test_tempo_exact_fraction_beat_grid_and_bar_positions():
    tempo = Tempo(24000, Fraction(120))
    assert tempo.beat_to_frame(1) == 12000
    assert tempo.beat_to_frame(Fraction(1, 2)) == 6000
    assert tempo.bar_to_frame(1) == 48000
    assert tempo.beat_grid(4) == (0, 12000, 24000, 36000)
    assert Tempo.from_bpm(24000, 120).beat_to_frame(2) == 24000
    with pytest.raises(AudioContractError):
        Tempo.from_bpm(24000, 0)


def test_adsr_phases_fit_sound_duration_and_end_at_zero():
    env = envelope_adsr(1000, attack=100, decay=100,
                        sustain=.3, release=200)
    assert env.shape == (1000,)
    assert env[0] == 0
    assert env[99] == pytest.approx(1)
    assert env[199] == pytest.approx(.3)
    assert env[300] == pytest.approx(.3)
    assert env[-1] == 0
    with pytest.raises(AudioContractError):
        envelope_adsr(100, attack=100, decay=50, sustain=.5, release=100)


def _graph():
    return SignalGraph((
        GraphNode("root", "source", parameters={"source_id": "test"}),
        GraphNode("fade", "fade", ("root",),
                  {"fade_in": 32, "fade_out": 32}),
        GraphNode("gain", "gain", ("fade",), {"db": -6.020599913}),
        GraphNode("limited", "peak_limit", ("gain",),
                  {"ceiling_dbfs": -1}),
    ), "limited")


def test_data_graph_has_canonical_hash_and_stable_report(source):
    graph = _graph()
    original = graph.as_dict()
    loaded = SignalGraph.from_dict(json.loads(json.dumps(original)))
    assert graph.sha256 == loaded.sha256
    executed = execute_graph(graph, {"test": source})
    repeat = execute_graph(graph, {"test": source})
    assert executed.graph_sha256 == graph.sha256
    assert executed.audio.frames == 256
    assert executed.audio.sample_rate == 24000
    assert digest_audio(executed.audio) == digest_audio(repeat.audio)
    assert executed.audio.samples[0, 0] == 0
    assert executed.audio.samples[0, -1] == 0
    assert np.max(executed.audio.samples) <= .25 + 1e-4
    assert executed.source_modification is False
    assert executed.neural_model_inference is False
    result = executed.report()
    assert len(result["nodes"]) == 4
    assert result["nodes"][-1]["operation"] == "peak_limit"


def test_graph_explicit_mixing_of_sibling_nodes(source):
    graph = SignalGraph((
        GraphNode("root", "source", (), {"source_id": "test"}),
        GraphNode("quiet", "gain", ("root",), {"db": -6.020599913}),
        GraphNode("mixed", "mix", ("root", "quiet"),
                  {"normalize": "none"}),
    ), "mixed")
    output = execute_graph(graph, {"test": source}).audio
    assert output.frames == 256
    np.testing.assert_allclose(output.samples, .75, atol=1e-6)
    safety = graph_safety_report(graph, {"test": source})
    assert safety["unresolved_sources"] == []
    assert safety["executable_code_allowed"] is False
    assert safety["file_or_network_access_allowed"] is False
    assert safety["model_inference_allowed"] is False


def test_graph_rejects_cycles_missing_names_duplicate_ids_and_arbitrary_operators():
    with pytest.raises(AudioContractError):
        SignalGraph((GraphNode("a", "gain", ("b",), {"db": 1}),
                     GraphNode("b", "source", (), {"source_id": "test"})), "a")
    with pytest.raises(AudioContractError):
        SignalGraph((GraphNode("a", "source", (), {"source_id": "test"}),
                     GraphNode("a", "source", (), {"source_id": "test"})), "a")
    with pytest.raises(AudioContractError):
        GraphNode("a", "__import__", (), {})
    with pytest.raises(AudioContractError):
        GraphNode("a", "source", (), {"source_id": "test",
                                      "arbitrary_python": "os.system"})
    with pytest.raises(AudioContractError):
        GraphNode("a", "gain", ("root",), {"db": float("nan")})


def test_graph_refuses_missing_source_and_too_many_materialized_frames(source):
    graph = _graph()
    with pytest.raises(AudioContractError):
        execute_graph(graph, {})
    with pytest.raises(AudioLimitError):
        execute_graph(graph, {"test": source}, max_total_frames=500)


def test_graph_parser_only_accepts_exact_allowlisted_schema():
    graph = _graph().as_dict()
    graph["unexpected"] = "script"
    with pytest.raises(AudioContractError):
        SignalGraph.from_dict(graph)
    graph = _graph().as_dict()
    graph["nodes"][0]["parameters"]["path"] = "/tmp/private"
    with pytest.raises(AudioContractError):
        SignalGraph.from_dict(graph)


def test_graph_output_is_buffer_not_mutable_global_cache(source):
    graph = SignalGraph((
        GraphNode("root", "source", (), {"source_id": "test"}),
        GraphNode("reversed", "reverse", ("root",), {}),
    ), "reversed")
    result = execute_graph(graph, {"test": source})
    assert result.audio.frames == source.frames
    with pytest.raises(ValueError):
        result.audio.samples[0, 0] = 1
    assert source.samples[0, 0] == pytest.approx(.5)
