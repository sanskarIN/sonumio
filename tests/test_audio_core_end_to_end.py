"""End-to-end first-party local PCM pipeline and scientific provenance gates.

These tests cross real package boundaries while avoiding neural checkpoint
downloads, external sounds, secrets, user voice capture or network requests.
They ensure local rendered audio can be audited after DSP, transforms and
WAV quantization, without changing original Sonumio procedural generation.
"""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.synthesis import SynthNote, render_synth_score
from sonumio.audio_core.filters import FilterSpec, biquad_filter
from sonumio.audio_core.dynamics import peak_limit
from sonumio.audio_core.stft import StftConfig, stft, istft
from sonumio.audio_core.waveio import encode_wav_bytes, decode_wav_bytes
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.resample import ResampleSettings, resample_audio
from sonumio.audio_core.timeline import Clip, Track, Timeline, render_timeline
from sonumio.audio_core.streaming import (
    FramerConfig, FramePacket, StreamingFramer, split_packets,
    PacketCollector, offline_window_reference,
)
from sonumio.audio_core.quality import (
    QualityLimits, measure_quality, waveform_integrity_report,
)
from sonumio.audio_core.graph import GraphNode, SignalGraph, execute_graph
from sonumio.audio_core.hashing import digest_audio
from research.audio_eval.reference import (
    RightsDeclaration, AudioReference, EvaluationPair, EvaluationSet,
)
from research.audio_eval.protocol import EvaluationPlan, MetricRule
from research.audio_eval.runner import run_evaluation
from research.audio_eval.report_io import (
    NumericReport, serialize_numeric_report, deserialize_numeric_report,
)
from sonumio.audio_core.errors import AudioContractError


@pytest.fixture
def rendered():
    notes = (
        SynthNote(60, 0, 6000, attack_frames=50, decay_frames=100,
                  release_frames=400, pan=-.4),
        SynthNote(64, 2000, 6000, attack_frames=50, decay_frames=100,
                  release_frames=400, pan=.3),
        SynthNote(67, 4000, 6000, attack_frames=50, decay_frames=100,
                  release_frames=400, pan=.1),
    )
    return render_synth_score(
        notes, sample_rate=24000, frames=12000,
        channels=2, master_gain=.25,
    )


def test_from_original_note_score_to_dsp_and_roundtrip_wav(rendered):
    unchanged = digest_audio(rendered)
    filtered = biquad_filter(rendered, FilterSpec(24000, 3500, "lowpass"))
    limited = peak_limit(filtered, ceiling_dbfs=-1)
    wav = encode_wav_bytes(limited, settings=QuantizationSettings(bits=24))
    assert wav[:4] == b"RIFF"
    decoded = decode_wav_bytes(wav)
    assert decoded.frames == rendered.frames
    assert decoded.channels == rendered.channels
    assert decoded.sample_rate == rendered.sample_rate
    np.testing.assert_allclose(decoded.samples, limited.samples, atol=2e-7)
    assert digest_audio(rendered) == unchanged
    assert digest_audio(decoded) != unchanged


def test_the_same_procedural_score_produces_same_wave_bytes(rendered):
    a = encode_wav_bytes(rendered, settings=QuantizationSettings(bits=16))
    b = encode_wav_bytes(rendered, settings=QuantizationSettings(bits=16))
    assert a == b
    assert decode_wav_bytes(a).frames == rendered.frames
    assert rendered.samples.flags.writeable is False
    with pytest.raises(ValueError):
        rendered.samples.setflags(write=True)


def test_pcm_stft_and_sinc_stage_preserve_clocks_without_changing_inputs(rendered):
    before = digest_audio(rendered)
    spectrum = stft(rendered, StftConfig(n_fft=512, hop_length=128))
    reconstructed = istft(spectrum)
    np.testing.assert_allclose(reconstructed.samples, rendered.samples, atol=1e-5)
    low_rate = resample_audio(reconstructed, ResampleSettings(8000, lobes=32))
    assert low_rate.frames == 4000
    restored = resample_audio(low_rate, ResampleSettings(24000, lobes=32))
    assert restored.frames == rendered.frames
    assert np.isfinite(restored.samples).all()
    assert digest_audio(rendered) == before


def test_real_multitrack_mix_can_be_processed_by_data_only_graph(rendered):
    one = rendered.sliced(0, 6000)
    two = rendered.sliced(6000, 12000)
    timeline = Timeline(24000, AudioLayout.for_channels(2), 12000, (
        Track("main", (
            Clip("a", one, start_frame=0, gain_db=-3),
            Clip("b", two, start_frame=6000, gain_db=-3),
        )),
    ))
    mixed = render_timeline(timeline, normalization="peak")
    graph = SignalGraph((
        GraphNode("audio", "source", parameters={"source_id": "mix"}),
        GraphNode("dc", "dc_block", ("audio",), {"pole": .98}),
        GraphNode("limited", "peak_limit", ("dc",),
                  {"ceiling_dbfs": -1}),
    ), "limited")
    result = execute_graph(graph, {"mix": mixed})
    assert result.audio.frames == rendered.frames
    assert result.audio.channels == 2
    assert result.audio.sample_rate == 24000
    assert len(result.trace) == 3
    assert result.source_modification is False
    assert result.neural_model_inference is False
    assert digest_audio(result.audio) == digest_audio(
        execute_graph(graph, {"mix": mixed}).audio
    )


def test_stream_packet_sizes_cannot_change_window_contents(rendered):
    cfg = FramerConfig(window_frames=256, hop_frames=128, pad_last=True)
    oracle = offline_window_reference(rendered, cfg)
    for packet_size in (17, 113, 256, 777):
        packets = split_packets(rendered, packet_frames=packet_size)
        framing = StreamingFramer(sample_rate=24000, channels=2, config=cfg)
        produced = []
        collector = PacketCollector(
            sample_rate=24000, layout=AudioLayout.for_channels(2),
            maximum_frames=12000,
        )
        for packet in packets:
            collector.push(packet)
            produced.extend(framing.push(packet))
        produced.extend(framing.finish())
        assert len(produced) == len(oracle)
        for a, b in zip(produced, oracle):
            np.testing.assert_array_equal(a.samples, b.samples)
            assert a.start_frame == b.start_frame
        np.testing.assert_array_equal(collector.finish().samples, rendered.samples)


def test_numeric_quality_report_discloses_nonperceptual_scope(rendered):
    result = measure_quality(rendered, limits=QualityLimits())
    health = waveform_integrity_report(rendered)
    assert health["pcm_sha256"] == digest_audio(rendered)
    assert health["numerically_finite"] is True
    assert health["model_weights_loaded"] is False
    assert health["does_not_verify_rights_or_consent"] is True
    assert result.as_dict()["perceptual_audio_quality_measured"] is False


def _rightful():
    return RightsDeclaration(
        "first-party-synthetic", "first-party-test-fixture",
        rights_reviewed=True, voice_present=False, creator_permission=True,
    )


def test_two_independent_first_party_groups_can_be_evaluated_without_audio_upload(
    rendered
):
    left = rendered.sliced(0, 6000)
    right = rendered.sliced(6000, 12000)
    sources = (
        AudioReference("a", left, "g1", "test", "nonverbal",
                       "First original note score excerpt", _rightful()),
        AudioReference("b", right, "g2", "test", "nonverbal",
                       "Second original note score excerpt", _rightful()),
    )
    pairs = (
        EvaluationPair("case_a", sources[0], left.copy_with(left.samples * .95),
                       "reference-procedural"),
        EvaluationPair("case_b", sources[1], right.copy_with(right.samples * .95),
                       "reference-procedural"),
    )
    evaluation = EvaluationSet(pairs)
    plan = EvaluationPlan(
        "local_prereg_v1", "test", MetricRule("sample_rmse"),
        "reference-procedural", fft_sizes=(256,),
        bootstrap_iterations=100, minimum_groups=2, seed=123,
    )
    result = run_evaluation(evaluation, plan)
    assert result.independent_groups == 2
    assert result.mean_metric > 0
    assert result.as_dict()["audio_files_saved"] is False
    assert result.as_dict()["perceptual_audio_quality_assessed"] is False
    archived = NumericReport(
        "synthetic_study", "numerical-acoustic-evaluation",
        result.as_dict(),
    )
    serialized = serialize_numeric_report(archived)
    assert deserialize_numeric_report(serialized).sha256 == archived.sha256
    assert len(serialized) < 500_000


def test_numeric_report_rejects_any_embedded_pcm_or_model_checkpoint(rendered):
    with pytest.raises(AudioContractError):
        NumericReport("invalid", "numerical-acoustic-evaluation", {
            "audio_bytes": encode_wav_bytes(rendered)
        })
    with pytest.raises(AudioContractError):
        NumericReport("invalid", "numerical-acoustic-evaluation", {
            "raw_waveform": rendered.samples
        })


def test_current_model_registry_still_distinguishes_dsp_from_neural_generation():
    from sonumio.registry import list_models
    entries = list_models()
    first_party = next(x for x in entries if x["id"] == "procedural-sfx-v0")
    assert first_party["original_sonumio_model"] is False
    assert "NOT AI" in first_party["model_kind"]
    assert all(not x["original_sonumio_model"] for x in entries)
