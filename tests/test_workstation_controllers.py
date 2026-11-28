"""MIDI CC and pitch-bend lane extraction, interpolation and RPN boundaries."""
from fractions import Fraction
import json

import pytest

from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.controllers import (
    ControlPoint, ControllerLane, PitchBendPoint, PitchBendLane,
    extract_controllers, extract_pitch_bends, extract_bend_sensitivity,
    inspect_controllers,
)
from sonumio.midi_workstation.model import MidiEvent, MidiFile, MidiTrack, TimeDivision


def channel(tick, kind, c, *data):
    return MidiEvent.channel(tick, kind, c, *data)


def midi(*events):
    return MidiFile(0, TimeDivision(480), (
        MidiTrack.from_events(tuple(sorted(events, key=lambda e: e.tick)), end=960),
    ))


def volume():
    return ControllerLane(0, 7, (
        ControlPoint(0, 0), ControlPoint(100, 100), ControlPoint(200, 50),
    ))


def test_controller_hold_uses_last_message_value():
    lane = volume()
    assert lane.value_at(0) == 0
    assert lane.value_at(50) == 0
    assert lane.value_at(100) == 100
    assert lane.value_at(199) == 100
    assert lane.value_at(201) == 50


def test_controller_linear_interpolation_is_exact_rational():
    lane = volume()
    assert lane.value_at(50, mode="linear") == 50
    assert lane.value_at(150, mode="linear") == 75
    assert lane.value_at(175, mode="linear") == Fraction(125, 2)
    assert isinstance(lane.value_at(151, mode="linear"), Fraction)


def test_controller_insert_and_remove_do_not_mutate_original():
    original = volume()
    added = original.insert(ControlPoint(150, 90))
    assert len(added.points) == 4
    assert len(original.points) == 3
    assert added.remove(150) == original
    with pytest.raises(MidiWorkstationError):
        original.insert(ControlPoint(100, 1))
    replaced = original.insert(ControlPoint(100, 1), replace=True)
    assert replaced.value_at(100) == 1


def test_controller_linear_resampling_uses_consistent_grid():
    lane = volume()
    sampled = lane.resample(start=0, stop=200, step=20)
    assert [p.tick for p in sampled.points] == list(range(0, 200, 20)) + [200]
    assert sampled.value_at(40) == 40
    assert sampled.value_at(100) == 100
    assert sampled.value_at(200) == 50


def test_controller_simplification_preserves_exact_piecewise_line():
    source = ControllerLane(0, 7, tuple(ControlPoint(t, t // 2)
                                        for t in range(0, 201, 2)))
    reduced = source.simplify(maximum_error=0)
    assert len(reduced.points) == 2
    assert reduced.points[0] == source.points[0]
    assert reduced.points[-1] == source.points[-1]
    for tick in range(0, 201):
        assert reduced.value_at(tick, mode="linear") == source.value_at(tick, mode="linear")


def test_controller_simplification_limits_error_below_tolerance():
    source = ControllerLane(0, 11, (
        ControlPoint(0, 0), ControlPoint(10, 10),
        ControlPoint(20, 16), ControlPoint(30, 30), ControlPoint(40, 40),
    ))
    reduced = source.simplify(maximum_error=4)
    for item in source.points:
        error = abs(reduced.value_at(item.tick, mode="linear") - item.value)
        assert error <= 4


def test_controller_to_midi_preserves_channel_controller_value():
    lane = volume()
    events = lane.to_midi()
    assert len(events) == 3
    assert events[1].status == 0xB0
    assert events[1].data == bytes((7, 100))


def test_controller_extraction_last_value_wins_same_tick():
    source = midi(
        channel(0, 0xB, 2, 7, 50),
        channel(0, 0xB, 2, 7, 80),
        channel(30, 0xB, 2, 10, 64),
        channel(100, 0xB, 2, 7, 127),
        channel(100, 0xB, 3, 7, 30),
    )
    lanes = extract_controllers(source, track=0, channel=2)
    assert len(lanes) == 2
    volumes = next(v for v in lanes if v.controller == 7)
    assert len(volumes.points) == 2
    assert volumes.points[0].value == 80
    assert volumes.points[-1].value == 127


def test_pitch_bend_positive_negative_centers():
    lane = PitchBendLane(0, (
        PitchBendPoint(0, 0),
        PitchBendPoint(100, 8192),
        PitchBendPoint(200, 16383),
    ))
    assert lane.semitones_at(0) == Fraction(-2)
    assert lane.semitones_at(100) == Fraction(0)
    assert lane.semitones_at(200) == Fraction(2)
    assert lane.value_at(250) == 16383


def test_pitch_bend_to_midi_is_fourteen_bit_little_endian():
    lane = PitchBendLane(3, (PitchBendPoint(0, 0x2345),))
    events = lane.to_midi()
    assert events[0].status == 0xE3
    assert events[0].data == bytes((0x45, 0x46))
    source = midi(*events)
    recovered = extract_pitch_bends(source, track=0, channel=3)
    assert recovered == lane


def test_pitch_bend_uses_last_same_tick_message():
    source = midi(
        channel(100, 0xE, 0, 0, 0),
        channel(100, 0xE, 0, 0, 64),
        channel(110, 0xE, 0, 127, 127),
    )
    lane = extract_pitch_bends(source, track=0, channel=0)
    assert [p.value for p in lane.points] == [8192, 16383]


def test_pitch_bend_sensitivity_only_tracks_rpn_zero_zero():
    source = midi(
        channel(0, 0xB, 0, 101, 0),
        channel(0, 0xB, 0, 100, 0),
        channel(10, 0xB, 0, 6, 12),
        channel(10, 0xB, 0, 38, 50),
        channel(50, 0xB, 0, 101, 127),
        channel(60, 0xB, 0, 6, 90),
    )
    curve = extract_bend_sensitivity(source, track=0, channel=0)
    assert curve[0].semitones == 2
    assert curve[1].semitones == 12
    assert curve[1].cents == 50
    assert len(curve) == 2


def test_performance_control_inspection_is_json_and_non_neural():
    source = midi(
        channel(0, 0xB, 0, 7, 90),
        channel(10, 0xE, 0, 0, 64),
    )
    report = inspect_controllers(source, track=0, channel=0)
    assert report["model_inference"] is False
    assert report["channel_controls_applied_to_audio"] is False
    assert report["controller_lanes"][0]["controller"] == 7
    assert report["pitch_bend"]["points"] == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad", [
    lambda: ControlPoint(-1, 0), lambda: ControlPoint(0, 128),
    lambda: ControllerLane(16, 7, ()), lambda: ControllerLane(0, 128, ()),
    lambda: ControllerLane(0, 7, (ControlPoint(100, 1), ControlPoint(100, 2))),
    lambda: ControllerLane(0, 7, (ControlPoint(200, 1), ControlPoint(100, 2))),
    lambda: PitchBendPoint(0, 16384),
    lambda: PitchBendLane(16, ()),
    lambda: PitchBendLane(0, (PitchBendPoint(30, 0), PitchBendPoint(20, 0))),
])
def test_invalid_controller_models_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        bad()


def test_invalid_controller_sample_geometry_rejected():
    with pytest.raises(MidiWorkstationError):
        volume().resample(start=0, stop=0, step=1)
    with pytest.raises(MidiLimitError):
        volume().resample(start=0, stop=1000000, step=1)
    with pytest.raises(MidiWorkstationError):
        volume().value_at(0, mode="cubic")


def test_source_midi_unchanged_after_control_analysis():
    source = midi(channel(0, 0xB, 0, 7, 64), channel(20, 0xE, 0, 0, 64))
    before = source
    extract_controllers(source, track=0, channel=0)
    extract_pitch_bends(source, track=0, channel=0)
    extract_bend_sensitivity(source, track=0, channel=0)
    assert source == before
