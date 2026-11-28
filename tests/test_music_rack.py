"""Independent signal and routing oracles for the bounded music rack."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan, TRACKS, compose_arrangement
from sonumio.audio import SAMPLE_RATE
from sonumio.errors import GenerationError
from sonumio.music_automation import AutomationPlan, MixKey, MixLane, mix_automated_stems
from sonumio.music_mix import MixPlan, render_stems
from sonumio.music_rack import (
    AuxBus, RackPlan, Send, TrackRack, decode_rack, encode_rack,
    describe_rack, generate_rack_music, mix_rack_stems, MAX_RACK_BYTES,
)
from sonumio.music_rack_effects import Insert, apply_insert, apply_chain, stage_digest
from sonumio.music_rack_cli import main
from sonumio.music_score_io import encode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4, bpm=160, seed=71, swing=.4))


@pytest.fixture(scope="module")
def stems(score):
    return render_stems(score)


def impulse(frames=256):
    data = np.zeros((2, frames), dtype=np.float32)
    data[0, 0] = .5
    return data


def synthetic(score, track="lead"):
    frames = score.plan.frame(score.plan.ticks)
    data = {name: np.zeros((2, frames), dtype=np.float32) for name in TRACKS}
    data[track][:, 2000] = [.2, .1]
    return data


def unity(track="lead"):
    return AutomationPlan((), MixPlan(
        gains=tuple(1. if name == track else 0. for name in TRACKS),
        room=0., peak_ceiling=.99))


@pytest.mark.parametrize("kind", ["gain", "width", "eq", "compressor", "gate",
                                  "delay", "tremolo", "saturation"])
def test_bypass_and_zero_wet_are_exact(kind):
    data = impulse()
    assert apply_insert(data, Insert.create(kind, bypass=True)) is data
    assert apply_insert(data, Insert.create(kind, wet=0.)) is data
    output, report = apply_chain(data, (Insert.create(kind, bypass=True),))
    assert output is data
    assert report[0]["input"] == report[0]["output"]
    assert report[0]["enabled"] is False


def test_gain_and_partial_wet():
    data = impulse()
    insert = Insert.create("gain", db=20*np.log10(2.).item(), wet=.25)
    np.testing.assert_allclose(apply_insert(data, insert), data * 1.25, atol=1e-8)
    assert data[0, 0] == .5


def test_width_preserves_mid_and_mono_fold():
    data = np.array([[.25, -.1], [-.25, .3]], dtype=np.float64)
    mono = apply_insert(data, Insert.create("width", width=0))
    np.testing.assert_allclose(mono[0], mono[1], atol=0)
    np.testing.assert_allclose(mono[0], data.mean(axis=0))
    wide = apply_insert(data, Insert.create("width", width=2))
    np.testing.assert_allclose(wide.mean(axis=0), data.mean(axis=0))
    np.testing.assert_allclose(wide[0]-wide[1], 2*(data[0]-data[1]))


def test_delay_has_explicit_wet_branch_and_crossfeed():
    data = impulse()
    delayed = apply_insert(data, Insert.create("delay", delay_seconds=.001,
                          feedback=.5, taps=2, ping_pong=True))
    assert delayed[1, 24] == .5
    assert delayed[0, 48] == .25
    assert np.count_nonzero(delayed) == 2
    silent = apply_insert(data, Insert.create("delay", delay_seconds=2))
    assert not np.any(silent)


def test_eq_discriminates_low_and_high_frequency():
    t = np.arange(4800) / SAMPLE_RATE
    low = np.sin(2*np.pi*100*t).astype(np.float32)
    high = np.sin(2*np.pi*6000*t).astype(np.float32)
    data = np.stack((low, high))*.2
    out = apply_insert(data, Insert.create("eq", kind="lowpass", cutoff_hz=1000))
    assert np.sqrt(np.mean(out[0, 1000:]**2)) > .12
    assert np.sqrt(np.mean(out[1, 1000:]**2)) < .01


def test_compressor_is_linked_and_reduces_level():
    data = np.array([[.8]*512, [.4]*512], dtype=np.float32)
    output = apply_insert(data, Insert.create("compressor", threshold_dbfs=-18,
                          ratio=4, attack_seconds=0, knee_db=0))
    assert np.max(output) < .5
    np.testing.assert_allclose(output[0], output[1]*2)


def test_gate_suppresses_quiet_signal():
    data = np.full((2, 100), .00001, dtype=np.float32)
    output = apply_insert(data, Insert.create("gate", floor_db=-60))
    assert np.max(np.abs(output)) < 2e-8


def test_effect_order_is_audible_signal_difference():
    data = impulse()
    gain = Insert.create("gain", db=12)
    clip = Insert.create("saturation", threshold=.3)
    forward, trace = apply_chain(data, (gain, clip))
    reverse, _ = apply_chain(data, (clip, gain))
    assert not np.allclose(forward, reverse)
    assert trace[0]["output"] == trace[1]["input"]


@pytest.mark.parametrize("kwargs", [
    {"db": True}, {"db": float("nan")}, {"db": 100},
    {"file": "anything"}, {"width": 1},
])
def test_strict_gain_parameters(kwargs):
    with pytest.raises(GenerationError):
        Insert.create("gain", **kwargs)


def test_reject_duplicate_params_and_invalid_route():
    with pytest.raises(GenerationError):
        Insert("gain", (("db", 0), ("db", 1)))
    with pytest.raises(GenerationError):
        RackPlan(tracks=(TrackRack("lead", ()), TrackRack("lead", ())))
    with pytest.raises(GenerationError):
        AuxBus("aux1", (Send("lead", .2), Send("lead", .3)), (Insert("gain"),))
    with pytest.raises(GenerationError):
        AuxBus("aux3", (Send("lead", .2),), (Insert("gain"),))
    with pytest.raises(GenerationError):
        Insert.create("delay", taps=2.5)
    with pytest.raises(GenerationError):
        Insert.create("eq", kind="lowpass", gain_db=1)
    with pytest.raises(GenerationError):
        Insert.create("eq", kind="lowshelf", q=2)


def test_canonical_roundtrip():
    plan = RackPlan(tracks=(TrackRack("lead", (Insert.create("gain", db=2),)),),
                    aux=(AuxBus("aux1", (Send("lead", .2),), (Insert("delay"),)),),
                    master=(Insert.create("width", width=1.2),))
    data = encode_rack(plan)
    assert decode_rack(data) == plan
    assert decode_rack(data).sha256 == hashlib.sha256(data).hexdigest()
    assert Insert.create("gain", db=0) == Insert("gain")


@pytest.mark.parametrize("data", [b"", b"[]", b"\xff", b"{}",
                                  b"x"*(MAX_RACK_BYTES+1),
                                  b'{"format":1,"format":2}'])
def test_malformed_documents(data):
    with pytest.raises(GenerationError):
        decode_rack(data)


def test_nested_duplicate_and_unknown_json_keys():
    data = encode_rack(RackPlan(master=(Insert("gain"),)))
    with pytest.raises(GenerationError):
        decode_rack(data.replace(b'"db":0.0', b'"db":0.0,"db":1.0'))
    obj = json.loads(data)
    obj["master"][0]["parameters"]["plugin"] = "external"
    with pytest.raises(GenerationError):
        decode_rack(json.dumps(obj).encode())


def test_noop_rack_exact_automation_parity(score, stems):
    auto = AutomationPlan((MixLane("lead", "gain", (MixKey(0, .4), MixKey(32, 1.))),),
                          MixPlan(room=.2))
    expected, _ = mix_automated_stems(score, stems, auto)
    actual, receipt = mix_rack_stems(score, stems, RackPlan(auto))
    np.testing.assert_array_equal(actual, expected)
    assert receipt["rendered"]
    assert receipt["true_peak_measured"] is False
    bypass = RackPlan(auto, master=(Insert.create("eq", bypass=True),))
    actual, _ = mix_rack_stems(score, stems, bypass)
    np.testing.assert_array_equal(actual, expected)


def test_post_fader_send_uses_automated_level(score):
    data = synthetic(score)
    auto = AutomationPlan((MixLane("lead", "gain", (MixKey(0, .5),)),), unity().mix)
    bus = AuxBus("aux1", (Send("lead", .5),),
                 (Insert.create("delay", delay_seconds=.001, taps=1),))
    output, receipt = mix_rack_stems(score, data, RackPlan(auto, aux=(bus,)))
    np.testing.assert_allclose(output[:, 2000], [.1, .05], atol=1e-7)
    np.testing.assert_allclose(output[:, 2024], [.05, .025], atol=1e-7)
    assert np.count_nonzero(output) == 4
    assert receipt["tracks"]["lead"]["sends"] == {"aux1": .5}


def test_mute_prevents_all_sends(score):
    data = synthetic(score)
    auto = AutomationPlan((), replace(unity().mix, mute=("lead",)))
    bus = AuxBus("aux1", (Send("lead", 1.),), (Insert("delay"),))
    output, receipt = mix_rack_stems(score, data, RackPlan(auto, aux=(bus,)))
    assert not np.any(output)
    assert receipt["aux"]["aux1"]["input"]["peak"] == 0
    assert receipt["tracks"]["lead"]["post_fader"] is None


def test_master_peak_ceiling_and_no_source_mutation(score):
    data = synthetic(score)
    old = {key: stage_digest(value) for key, value in data.items()}
    plan = RackPlan(unity(), master=(Insert.create("gain", db=18),))
    output, receipt = mix_rack_stems(score, data, plan)
    assert np.max(np.abs(output)) <= .99 + 1e-7
    assert receipt["master_attenuation"] < 1.
    assert {key: stage_digest(value) for key, value in data.items()} == old
    second, second_receipt = mix_rack_stems(score, data, plan)
    np.testing.assert_array_equal(output, second)
    assert receipt == second_receipt


def test_oversized_stage_fails_closed():
    data = np.full((2, 4), 20., dtype=np.float32)
    with pytest.raises(GenerationError):
        apply_insert(data, Insert.create("gain", db=18))


def test_processing_budget_checked_before_render(score, monkeypatch):
    plan = RackPlan(master=(Insert("gate"),)*6,
                    tracks=tuple(TrackRack(name, (Insert("gate"),)*6) for name in TRACKS))
    import sonumio.music_rack as rack
    def forbidden(_):
        pytest.fail("render started before budget validation")
    monkeypatch.setattr(rack, "render_stems", forbidden)
    with pytest.raises(GenerationError, match="budget"):
        generate_rack_music(score, plan)


def test_cli_pins_and_atomic_output(tmp_path, score, capsys):
    score_path = tmp_path / "score.json"
    rack_path = tmp_path / "rack.json"
    target = tmp_path / "out.wav"
    score_path.write_bytes(encode_score(score))
    plan = RackPlan()
    rack_path.write_bytes(encode_rack(plan))
    assert main(["inspect", str(score_path), str(rack_path)]) == 0
    assert json.loads(capsys.readouterr().out)["rendered"] is False
    args = ["render", str(score_path), str(rack_path), "--output", str(target)]
    assert main(args + ["--expected-rack-sha256", "0"*64]) == 2
    assert not target.exists()
    assert main(args + ["--expected-rack-sha256", plan.sha256]) == 0
    report = json.loads(capsys.readouterr().out)
    original = target.read_bytes()
    assert report["output_sha256"] == hashlib.sha256(original).hexdigest()
    assert main(args) == 2
    assert target.read_bytes() == original
    assert main(["render", str(score_path), str(rack_path),
                 "--output", str(score_path), "--overwrite"]) == 2
