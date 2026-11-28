"""Optional bounded Standard MIDI File and symbolic music workstation.

Sonumio's original strict arrangement/MIDI APIs remain unchanged. This
subpackage provides a separate, explicit expressive-performance workflow.
No external MIDI libraries, online services or trained models are used.
"""
from .contracts import (
    MidiWorkstationError, MidiFormatError, MidiLimitError, MidiIntegrityError,
)
from .model import MidiFile, MidiTrack, MidiEvent, TimeDivision
from .smf import parse_smf, encode_smf, read_smf, validate_roundtrip
from .tempo import TempoMap, TempoPoint, MeterMap, MeterPoint
from .performance import PerformanceSettings, Performance, NoteSpan, reconstruct_performance
from .score import Score, ScoreTrack, ScoreNote, from_midi, to_midi

__all__ = [
    "MidiWorkstationError", "MidiFormatError", "MidiLimitError", "MidiIntegrityError",
    "MidiFile", "MidiTrack", "MidiEvent", "TimeDivision",
    "parse_smf", "encode_smf", "read_smf", "validate_roundtrip",
    "TempoMap", "TempoPoint", "MeterMap", "MeterPoint",
    "PerformanceSettings", "Performance", "NoteSpan", "reconstruct_performance",
    "Score", "ScoreTrack", "ScoreNote", "from_midi", "to_midi",
]
