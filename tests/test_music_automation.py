"""Regression coverage for the original score-clock automation mixer."""
from __future__ import annotations

import hashlib
import io
import json
import wave

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan, TRACKS, compose_arrangement, arrangement_digest
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.music_automation import (
    AutomationPlan, FrameLane, MixKey, MixLane, BLOCK_FRAMES, MAX_PLAN_BYTES,
    compile_lanes, decode_automation, encode_automation, generate_automated_music,
    mix_automated_stems,
)
from sonumio.music_mix import MixPlan, generate_arrangement_audio, render_stems
from sonumio.music_score_io import encode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4, bpm=160, seed=71, swing=.4))


@pytest.fixture(scope="module")
def stems(score):
    return render_stems(score)


def lane(track="lead", parameter="gain", keys=None):
    return MixLane(track, parameter, (MixKey(0, 0.), MixKey(32, 1.))
                   if keys is None else keys)


def plan_for(*lanes, mix=None):
    return AutomationPlan(tuple(lanes), MixPlan() if mix is None else mix)


@pytest.mark.parametrize("law,expected", [
    ("linear", [0., .25, .5, .75, 1.]),
    ("smoothstep", [0., .15625, .5, .84375, 1.]),
    ("step", [0., 0., 0., 0., 1.]),
])
def test_frame_interpolation_exact_edges(law, expected):
    compiled = FrameLane((2, 6), (0., 1.), (law, "linear"))
    np.testing.assert_array_equal(compiled.render(0, 9),
                                  [0., 0., *expected, 1., 1.])


def test_endpoint_hold_and_step_discontinuity():
    compiled = FrameLane((2, 5, 7), (.2, .7, .1), ("step", "step", "linear"))
    np.testing.assert_allclose(compiled.render(0, 10),
                               [.2, .2, .2, .2, .2, .7, .7, .1, .1, .1])
    constant = FrameLane((10,), (.25,), ("linear",))
    np.testing.assert_array_equal(constant.render(0, 20), np.full(20, .25))


def test_exact_partition_invariance():
    compiled = FrameLane((0, 13, 8191, 12000, 20000), (0., .2, .5, 1., .3),
                         ("linear", "smoothstep", "step", "linear", "linear"))
    whole = compiled.render(0, BLOCK_FRAMES)
    for split in (1, 13, 257, 8191, BLOCK_FRAMES - 1):
        partial = np.concatenate((compiled.render(0, split),
                                  compiled.render(split, BLOCK_FRAMES - split)))
        np.testing.assert_array_equal(whole, partial)


@pytest.mark.parametrize("start,count", [(-1, 1), (0, 0), (0, BLOCK_FRAMES + 1),
                                           (True, 1), (0, True), (0, 2.0)])
def test_reject_invalid_frame_span(start, count):
    with pytest.raises(GenerationError):
        FrameLane((0,), (1.,), ("linear",)).render(start, count)


@pytest.mark.parametrize("tick", [-1, True, 1.5, 1025])
def test_invalid_tick(tick):
    with pytest.raises(GenerationError):
        MixKey(tick, 1.)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -2., 3., "1"])
def test_invalid_value(value):
    with pytest.raises(GenerationError):
        MixKey(0, value)


@pytest.mark.parametrize("keys", [
    (), (MixKey(2, .5), MixKey(1, .5)),
    (MixKey(2, .5), MixKey(2, 1.)),
    tuple(MixKey(i, .5) for i in range(65)),
])
def test_invalid_key_order_or_budget(keys):
    with pytest.raises(GenerationError):
        lane(keys=keys)


def test_invalid_track_parameter_and_parameter_range():
    with pytest.raises(GenerationError):
        lane(track="unknown")
    with pytest.raises(GenerationError):
        lane(parameter="pan")
    with pytest.raises(GenerationError):
        lane(keys=(MixKey(0, -.1),))
    with pytest.raises(GenerationError):
        lane(parameter="balance", keys=(MixKey(0, 1.1),))
    with pytest.raises(GenerationError):
        MixKey(0, 1., "cubic")


def test_duplicate_and_noncanonical_lanes():
    a = lane(track="keys")
    b = lane(track="lead")
    with pytest.raises(GenerationError, match="Duplicate"):
        plan_for(a, a)
    with pytest.raises(GenerationError, match="canonical"):
        plan_for(b, a)


def test_score_clock_swing_and_end_bound(score):
    keys = (MixKey(0, 0.), MixKey(4, 1.), MixKey(score.plan.ticks, .3))
    compiled = compile_lanes(score, plan_for(lane(keys=keys)))[("lead", "gain")]
    assert compiled.positions == tuple(score.plan.frame(k.tick) for k in keys)
    assert compiled.positions[1] > score.plan.frame(8) / 2
    with pytest.raises(GenerationError, match="beyond"):
        compile_lanes(score, plan_for(lane(keys=(MixKey(score.plan.ticks + 1, 1.),))))


def test_json_defaults_and_canonical_hash():
    raw = b'{"format":"sonumio-mix-automation-v1","lanes":[{"track":"lead","parameter":"gain","keys":[{"tick":0,"value":1}]}]}'
    a = decode_automation(raw)
    b = decode_automation(encode_automation(a))
    assert a == b
    assert a.sha256 == b.sha256 == hashlib.sha256(encode_automation(a)).hexdigest()
    raw_float = raw.replace(b'"value":1', b'"value":1.0')
    assert decode_automation(raw_float).sha256 == a.sha256


def test_json_lane_and_mix_order_normalization():
    a = lane(track="keys")
    b = lane(track="lead")
    raw = json.loads(encode_automation(plan_for(a, b)))
    raw["lanes"].reverse()
    raw["mix"]["mute"] = ["hat", "pad"]
    left = decode_automation(json.dumps(raw).encode())
    raw["mix"]["mute"].reverse()
    right = decode_automation(json.dumps(raw).encode())
    assert left.sha256 == right.sha256
    assert tuple(x.track for x in left.lanes) == ("keys", "lead")


@pytest.mark.parametrize("data", [
    b"", bytes(MAX_PLAN_BYTES + 1), b"\xff", b"[]", b"{}",
    b'{"format":"sonumio-mix-automation-v1","format":"sonumio-mix-automation-v1","lanes":[]}',
    b'{"format":"sonumio-mix-automation-v1","lanes":[],"unexpected":0}',
    b'{"format":"wrong","lanes":[]}',
    b'{"format":"sonumio-mix-automation-v1","lanes":{}}',
    b'{"format":"sonumio-mix-automation-v1","lanes":[],"mix":{"gains":1}}',
    b'{"format":"sonumio-mix-automation-v1","lanes":[],"mix":{"mute":["lead"],"solo":["lead"]}}',
    b'{"format":"sonumio-mix-automation-v1","lanes":[{"track":"lead","parameter":"gain","keys":[{"tick":0,"value":NaN}]}]}',
    b'{"format":"sonumio-mix-automation-v1","lanes":[{"track":"lead","parameter":"gain","keys":[{"tick":0,"value":1,"extra":true}]}]}',
])
def test_malformed_documents(data):
    with pytest.raises(GenerationError):
        decode_automation(data)


def test_empty_plan_matches_existing_static_wav(score):
    mix = MixPlan(room=.15, peak_ceiling=.7)
    expected, _ = generate_arrangement_audio(score, mix)
    actual, report = generate_automated_music(score, plan_for(mix=mix))
    # The existing and automated mixers use the same multiplication,
    # summation order, finite room response, fades and quantization.
    assert expected == actual
    assert report["automation_lanes"] == 0


def test_gain_zero_and_balance_extreme(score, stems):
    mix = MixPlan(solo=("lead",), room=0.)
    silent, report = mix_automated_stems(score, stems, plan_for(
        lane(keys=(MixKey(0, 0.),)), mix=mix))
    assert np.count_nonzero(silent) == 0
    assert report["tracks"]["lead"]["automated_rms"] == 0.
    right, report = mix_automated_stems(score, stems, plan_for(
        lane(parameter="balance", keys=(MixKey(0, 1.),)), mix=mix))
    assert np.count_nonzero(right[0]) == 0
    assert np.max(np.abs(right[1])) > 0.
    assert report["tracks"]["lead"]["balance_automated"]


def test_stems_not_mutated_and_replay(score, stems):
    original = {name: hashlib.sha256(stem.tobytes()).hexdigest()
                for name, stem in stems.items()}
    plan = plan_for(lane(), lane(parameter="balance", keys=(
        MixKey(0, -1.), MixKey(32, 1., "smoothstep"), MixKey(64, 0.))))
    x, report = mix_automated_stems(score, stems, plan)
    y, second = mix_automated_stems(score, stems, plan)
    np.testing.assert_array_equal(x, y)
    assert report == second
    assert report["score_sha256"] == arrangement_digest(score)
    assert report["automation_sha256"] == plan.sha256
    assert report["learned_model"] is False
    assert report["true_peak_measured"] is False
    assert all(hashlib.sha256(stem.tobytes()).hexdigest() == original[name]
               for name, stem in stems.items())


def test_attenuation_ceiling_and_inactive_stats(score, stems):
    mix = MixPlan(gains=(2.,) * len(TRACKS), mute=("lead",), room=1., peak_ceiling=.05)
    audio, report = mix_automated_stems(score, stems, plan_for(lane(), mix=mix))
    assert np.max(np.abs(audio)) <= .05000001
    assert 0 < report["master_attenuation"] <= 1
    assert report["tracks"]["lead"]["active"] is False
    assert report["tracks"]["lead"]["automated_peak"] == 0


def test_invalid_stem_data_fails(score, stems):
    bad = dict(stems)
    bad["lead"] = bad["lead"][:, :-1]
    with pytest.raises(GenerationError):
        mix_automated_stems(score, bad, plan_for())
    bad["lead"] = stems["lead"].astype(np.float64)
    with pytest.raises(GenerationError):
        mix_automated_stems(score, bad, plan_for())
    bad["lead"] = stems["lead"].copy()
    bad["lead"][0, 0] = np.nan
    with pytest.raises(GenerationError):
        mix_automated_stems(score, bad, plan_for())


def test_invalid_plan_fails_before_synthesis(score, monkeypatch):
    def forbidden(*args):
        pytest.fail("Invalid automation must not render any stems")

    monkeypatch.setattr("sonumio.music_automation.render_stems", forbidden)
    invalid = plan_for(lane(keys=(MixKey(score.plan.ticks + 1, 1.),)))
    with pytest.raises(GenerationError):
        generate_automated_music(score, invalid)


def test_cli_digest_and_no_clobber(score, tmp_path, capsys):
    score_path = tmp_path / "score.json"
    plan_path = tmp_path / "mix.json"
    output = tmp_path / "mix.wav"
    score_path.write_bytes(encode_score(score))
    plan = plan_for(lane(), mix=MixPlan(room=0.))
    plan_path.write_bytes(encode_automation(plan))
    args = ["music-automate", str(score_path), str(plan_path),
            "--expected-score-sha256", arrangement_digest(score),
            "--expected-automation-sha256", plan.sha256,
            "--output", str(output)]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    original = output.read_bytes()
    assert report["output_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["bytes_written"] == len(original)
    with wave.open(io.BytesIO(original)) as wav:
        assert wav.getnchannels() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() == score.plan.frame(score.plan.ticks)
    assert main(args) == 2
    assert main(args + ["--overwrite", "--expected-output-sha256", "0" * 64]) == 2
    assert main(args + ["--overwrite", "--expected-automation-sha256", "0" * 64]) == 2
    assert output.read_bytes() == original
    assert main(args + ["--overwrite", "--expected-output-sha256", report["output_sha256"]]) == 0
    assert output.read_bytes() == original


def test_cli_refuses_input_identity(score, tmp_path):
    score_path = tmp_path / "score.wav"
    mix_path = tmp_path / "mix.json"
    original = encode_score(score)
    score_path.write_bytes(original)
    mix_path.write_bytes(encode_automation(plan_for()))
    assert main(["music-automate", str(score_path), str(mix_path),
                 "--output", str(score_path), "--overwrite"]) == 2
    assert score_path.read_bytes() == original
