"""Exact rational MIDI tempo maps, SMPTE timing, meter and frame conversions."""
from fractions import Fraction
import random

import pytest

from sonumio.midi_workstation.contracts import MidiFormatError, MidiWorkstationError
from sonumio.midi_workstation.model import MidiEvent, MidiFile, MidiTrack, TimeDivision
from sonumio.midi_workstation.tempo import (
    MeterMap, MeterPoint, MusicalPosition, TempoMap, TempoPoint,
    tempo_summary,
)


def tempo_file(*changes, ppqn=480):
    events = tuple(MidiEvent.meta(tick, 0x51, value.to_bytes(3, "big"))
                   for tick, value in sorted(changes))
    return MidiFile(0, TimeDivision(ppqn),
                    (MidiTrack.from_events(events, end=19200),))


@pytest.mark.parametrize("ppqn", [1, 24, 96, 480, 960, 8192, 32767])
def test_default_quarter_note_at_120_bpm(ppqn):
    tempo = TempoMap.constant(ppqn)
    assert tempo.tick_to_microseconds(ppqn) == Fraction(500000)
    assert tempo.tick_to_microseconds(2 * ppqn) == Fraction(1000000)
    assert tempo.tick_to_seconds(ppqn) == Fraction(1, 2)


def test_tempo_change_accumulates_exact_fraction_without_drift():
    tempo = TempoMap(TimeDivision(480), (
        TempoPoint(0, 500000),
        TempoPoint(480, 400000),
        TempoPoint(1440, 600000),
    ))
    assert tempo.tick_to_microseconds(480) == 500000
    assert tempo.tick_to_microseconds(1440) == 1300000
    assert tempo.tick_to_microseconds(1920) == 1900000
    assert tempo.tempo_at(0).microseconds_per_quarter == 500000
    assert tempo.tempo_at(800).microseconds_per_quarter == 400000
    assert tempo.tempo_at(1920).microseconds_per_quarter == 600000
    assert tempo.duration(480, 1920) == Fraction(1400000)


@pytest.mark.parametrize("rounding", ["nearest", "floor", "ceil"])
def test_ticks_roundtrip_over_tempo_changes(rounding):
    tempo = TempoMap(TimeDivision(480), (
        TempoPoint(0, 500000), TempoPoint(96, 300000),
        TempoPoint(480, 510123), TempoPoint(950, 900000),
    ))
    for tick in [0, 1, 2, 95, 96, 97, 470, 480, 949, 950, 3000]:
        micros = tempo.tick_to_microseconds(tick)
        assert tempo.microseconds_to_tick(micros, rounding=rounding) == tick


def test_inverse_time_rounding_is_monotonic():
    tempo = TempoMap(TimeDivision(960), (
        TempoPoint(0, 333333), TempoPoint(901, 777777),
    ))
    inputs = [Fraction(value, 10) for value in range(0, 10000000, 14991)]
    for rounding in ("nearest", "ceil", "floor"):
        ticks = [tempo.microseconds_to_tick(us, rounding=rounding) for us in inputs]
        assert ticks == sorted(ticks)
        if rounding == "floor":
            assert all(tempo.tick_to_microseconds(t) <= us for t, us in zip(ticks, inputs))
        if rounding == "ceil":
            assert all(tempo.tick_to_microseconds(t) >= us for t, us in zip(ticks, inputs))


def test_tick_audio_frame_conversion_is_sample_clock_consistent():
    tempo = TempoMap.constant(480, 500000)
    assert tempo.ticks_to_frame(480, 48000) == 24000
    assert tempo.ticks_to_frame(960, 44100) == 44100
    assert tempo.frame_to_tick(24000, 48000) == 480
    assert tempo.frame_to_tick(44100, 44100) == 960


def test_frame_round_trip_with_rational_microseconds():
    tempo = TempoMap(TimeDivision(960), (
        TempoPoint(0, 451234), TempoPoint(490, 500123),
    ))
    for sample_rate in (8000, 11025, 44100, 48000, 96000):
        for tick in (0, 1, 10, 100, 480, 490, 960, 1500):
            frame = tempo.ticks_to_frame(tick, sample_rate)
            recovered = tempo.frame_to_tick(frame, sample_rate)
            assert abs(recovered - tick) <= 1


@pytest.mark.parametrize("code,ticks", [(-24, 80), (-25, 40), (-29, 80), (-30, 100)])
def test_smpte_clock_is_independent_of_tempo_events(code, ticks):
    division = TimeDivision(None, code, ticks)
    source = MidiFile(0, division, (MidiTrack.from_events((
        MidiEvent.meta(0, 0x51, (1000000).to_bytes(3, "big")),
        MidiEvent.meta(100, 0x51, (250000).to_bytes(3, "big")),
    ), end=2000),))
    tempo = TempoMap.from_midi(source)
    assert tempo.points == (TempoPoint(0, 500000),)
    assert tempo.tick_to_microseconds(ticks * 10) == Fraction(10000000, 1) / division.frame_rate
    assert tempo.microseconds_to_tick(tempo.tick_to_microseconds(1000)) == 1000


def test_midi_tempo_conflict_is_rejected():
    one = MidiTrack.from_events((MidiEvent.meta(0, 0x51, (400000).to_bytes(3, "big")),), end=1000)
    two = MidiTrack.from_events((MidiEvent.meta(0, 0x51, (500000).to_bytes(3, "big")),), end=1000)
    source = MidiFile(1, TimeDivision(), (one, two))
    with pytest.raises(MidiFormatError):
        TempoMap.from_midi(source)


def test_same_tempo_duplicate_events_are_not_conflicting():
    events = (MidiEvent.meta(0, 0x51, (500000).to_bytes(3, "big")),)
    one = MidiTrack.from_events(events, end=1000)
    two = MidiTrack.from_events(events, end=1000)
    source = MidiFile(1, TimeDivision(), (one, two))
    assert TempoMap.from_midi(source) == TempoMap.constant()


def test_format_two_requires_track_specific_tempo_map():
    source = MidiFile(2, TimeDivision(480), (
        MidiTrack.from_events((MidiEvent.meta(0, 0x51, (500000).to_bytes(3, "big")),), end=480),
        MidiTrack.from_events((MidiEvent.meta(0, 0x51, (300000).to_bytes(3, "big")),), end=480),
    ))
    with pytest.raises(MidiWorkstationError):
        TempoMap.from_midi(source)
    assert TempoMap.from_midi(source, track=1).tick_to_microseconds(480) == 300000
    assert TempoMap.from_midi(source, track=0).tick_to_microseconds(480) == 500000


def test_tempo_insert_replace_delete():
    clock = TempoMap.constant()
    faster = clock.insert(240, 250000)
    assert len(faster.points) == 2
    assert faster.tempo_at(480).microseconds_per_quarter == 250000
    with pytest.raises(MidiWorkstationError):
        faster.insert(240, 200000)
    updated = faster.insert(240, 200000, replace=True)
    assert updated.tempo_at(240).microseconds_per_quarter == 200000
    assert updated.remove(240) == clock
    with pytest.raises(MidiWorkstationError):
        updated.remove(0)


@pytest.mark.parametrize("bad", [
    (0, 0), (-1, 500000), (0, 0x1000000), (MAX := 0x0FFFFFFF + 1, 500000),
])
def test_bad_tempo_points_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        TempoPoint(*bad)


def test_nonmonotonic_tempo_points_rejected():
    with pytest.raises(MidiFormatError):
        TempoMap(TimeDivision(), (TempoPoint(0, 500000), TempoPoint(0, 400000)))
    with pytest.raises(MidiFormatError):
        TempoMap(TimeDivision(), (TempoPoint(1, 500000),))


@pytest.mark.parametrize("numerator,denominator", [(4, 4), (3, 4), (6, 8), (5, 16), (7, 8)])
def test_meter_binary_meta_roundtrip(numerator, denominator):
    point = MeterPoint(240, numerator, denominator)
    encoded = point.to_midi()
    assert encoded.data == bytes((numerator, denominator.bit_length() - 1, 24, 8))
    assert encoded.tick == 240


def test_meter_change_resets_bar_position():
    timeline = MeterMap(TimeDivision(480), (
        MeterPoint(0, 4, 4), MeterPoint(1920, 3, 4),
    ))
    assert timeline.position(0).bar == 1
    assert timeline.position(480).beat == 1
    assert timeline.position(1919).bar == 1
    assert timeline.position(1920).bar == 2
    assert timeline.position(3360).bar == 3
    assert timeline.position(3360).beat == 0


def test_midmeasure_meter_change_starts_fresh_bar():
    timeline = MeterMap(TimeDivision(480), (
        MeterPoint(0, 4, 4), MeterPoint(960, 6, 8),
    ))
    assert timeline.position(959).bar == 1
    assert timeline.position(960).bar == 2
    assert timeline.position(960).beat == 0


def test_meter_from_midi_extracts_explicit_changes():
    source = MidiFile(0, TimeDivision(480), (
        MidiTrack.from_events((
            MeterPoint(0, 4, 4).to_midi(),
            MeterPoint(1920, 3, 8).to_midi(),
        ), end=5000),
    ))
    timeline = MeterMap.from_midi(source)
    assert len(timeline.points) == 2
    assert timeline.position(1920).meter.denominator == 8


def test_tempo_report_is_json_safe_and_untrained():
    data = tempo_summary(tempo_file((0, 500000), (480, 300000)))
    assert data["protocol"] == "sonumio-midi-tempo-v1"
    assert data["points"][1]["tick"] == 480
    assert data["duration_microseconds"] > 0
    assert data["clock_drift_compensation"] is False
