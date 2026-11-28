"""Interpretable tonal harmony tools for original symbolic compositions.

All routines use explicit pitch-class interval sets. Chord hypotheses are not
trained classification outputs and cannot prove musical key or intended
harmony of a real recording. Generation uses mathematical voice-leading costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

from .contracts import MidiLimitError, MidiWorkstationError, integer, finite
from .score import Score, ScoreNote, ScoreTrack, note_sort_key

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural-minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic-minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic-minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "pentatonic-major": (0, 2, 4, 7, 9),
    "pentatonic-minor": (0, 3, 5, 7, 10),
    "chromatic": tuple(range(12)),
}
CHORD_INTERVALS = {
    "major": (0, 4, 7), "minor": (0, 3, 7),
    "diminished": (0, 3, 6), "augmented": (0, 4, 8),
    "suspended2": (0, 2, 7), "suspended4": (0, 5, 7),
    "major7": (0, 4, 7, 11), "minor7": (0, 3, 7, 10),
    "dominant7": (0, 4, 7, 10), "half-diminished7": (0, 3, 6, 10),
    "diminished7": (0, 3, 6, 9), "minor-major7": (0, 3, 7, 11),
    "major6": (0, 4, 7, 9), "minor6": (0, 3, 7, 9),
    "power": (0, 7),
}
MAX_VOICING_CANDIDATES = 4096
MAX_HARMONY_EVENTS = 10_000


def pitch_class_name(number: int) -> str:
    return PITCH_CLASSES[integer(number, "pitch class", 0, 127) % 12]


def parse_pitch_class(label: str) -> int:
    if type(label) is not str:
        raise MidiWorkstationError("Pitch class must be string")
    aliases = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10,
               "Cb": 11, "B#": 0, "Fb": 4, "E#": 5}
    if label in aliases:
        return aliases[label]
    if label not in PITCH_CLASSES:
        raise MidiWorkstationError("Unknown pitch class")
    return PITCH_CLASSES.index(label)


@dataclass(frozen=True, slots=True)
class Scale:
    tonic: int
    mode: str = "major"

    def __post_init__(self):
        integer(self.tonic, "scale tonic", 0, 11)
        if type(self.mode) is not str or self.mode not in SCALE_INTERVALS:
            raise MidiWorkstationError("Unsupported explicitly named scale")

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple((self.tonic + interval) % 12 for interval in SCALE_INTERVALS[self.mode])

    def contains(self, midi_note: int) -> bool:
        return integer(midi_note, "MIDI pitch", 0, 127) % 12 in self.classes

    def degree(self, midi_note: int) -> int | None:
        pitch = integer(midi_note, "degree MIDI pitch", 0, 127) % 12
        return self.classes.index(pitch) + 1 if pitch in self.classes else None

    def closest(self, midi_note: int, *, prefer: str = "nearest",
                minimum: int = 0, maximum: int = 127) -> int:
        integer(midi_note, "source pitch", 0, 127)
        integer(minimum, "minimum pitch", 0, 127)
        integer(maximum, "maximum pitch", minimum, 127)
        if prefer not in ("nearest", "up", "down"):
            raise MidiWorkstationError("Unsupported diatonic pitch correction")
        options = [pitch for pitch in range(minimum, maximum + 1) if self.contains(pitch)]
        if not options:
            raise MidiWorkstationError("Requested register has no scale pitches")
        if prefer == "up":
            upward = [p for p in options if p >= midi_note]
            if not upward:
                raise MidiWorkstationError("No upward scale pitch in register")
            return upward[0]
        if prefer == "down":
            downward = [p for p in options if p <= midi_note]
            if not downward:
                raise MidiWorkstationError("No downward scale pitch in register")
            return downward[-1]
        return min(options, key=lambda p: (abs(p - midi_note), p))


@dataclass(frozen=True, slots=True)
class Chord:
    root: int
    quality: str = "major"
    inversion: int = 0

    def __post_init__(self):
        integer(self.root, "chord root", 0, 11)
        if type(self.quality) is not str or self.quality not in CHORD_INTERVALS:
            raise MidiWorkstationError("Unsupported triad/seventh chord quality")
        integer(self.inversion, "chord inversion", 0, len(CHORD_INTERVALS[self.quality]) - 1)

    @property
    def intervals(self) -> tuple[int, ...]:
        return CHORD_INTERVALS[self.quality]

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple((self.root + value) % 12 for value in self.intervals)

    @property
    def symbol(self) -> str:
        suffix = {
            "major": "", "minor": "m", "diminished": "dim", "augmented": "+",
            "suspended2": "sus2", "suspended4": "sus4", "major7": "maj7",
            "minor7": "m7", "dominant7": "7", "half-diminished7": "m7b5",
            "diminished7": "dim7", "minor-major7": "mMaj7",
            "major6": "6", "minor6": "m6", "power": "5",
        }[self.quality]
        return PITCH_CLASSES[self.root] + suffix

    def contains(self, pitch: int) -> bool:
        return integer(pitch, "candidate chord pitch", 0, 127) % 12 in self.classes

    def root_midi(self, octave: int) -> int:
        integer(octave, "MIDI root octave", -1, 9)
        return integer((octave + 1) * 12 + self.root, "root pitch", 0, 127)

    def pitches(self, octave: int) -> tuple[int, ...]:
        root = self.root_midi(octave)
        degrees = list(root + interval for interval in self.intervals)
        for index in range(self.inversion):
            degrees[index] += 12
        if any(not 0 <= note <= 127 for note in degrees):
            raise MidiWorkstationError("Chord inversion exceeds MIDI pitch range")
        return tuple(sorted(degrees))


@dataclass(frozen=True, slots=True)
class ChordEvidence:
    chord: Chord
    score: float
    coverage: float
    root_present: bool
    out_of_chord_weight: float

    def __post_init__(self):
        if not isinstance(self.chord, Chord):
            raise MidiWorkstationError("Chord evidence requires Chord")
        for field in ("score", "coverage", "out_of_chord_weight"):
            finite(getattr(self, field), field, 0., 1.)
        if type(self.root_present) is not bool:
            raise MidiWorkstationError("root_present must be bool")


def estimate_chords(pitches: tuple[tuple[int, float], ...], *,
                    qualities: tuple[str, ...] = ("major", "minor", "diminished", "dominant7"),
                    candidates: int = 5,
                    require_three_distinct: bool = True) -> tuple[ChordEvidence, ...]:
    """Rank transparent template overlap, not calibrated recognition."""
    if type(pitches) is not tuple or len(pitches) > MAX_HARMONY_EVENTS:
        raise MidiWorkstationError("Expected bounded weighted pitch tuple")
    if type(qualities) is not tuple or not qualities or (
        len(set(qualities)) != len(qualities) or any(q not in CHORD_INTERVALS for q in qualities)
    ):
        raise MidiWorkstationError("Invalid chord-quality selection")
    integer(candidates, "number of chord candidates", 1, 48)
    if type(require_three_distinct) is not bool:
        raise MidiWorkstationError("Triad degree policy must be bool")
    energy = [0.] * 12
    for pitch, weight in pitches:
        integer(pitch, "weighted MIDI pitch", 0, 127)
        energy[pitch % 12] += finite(weight, "pitch-class evidence", 0., 1000.)
    total = sum(energy)
    if total <= 0:
        return ()
    energy = [entry / total for entry in energy]
    result = []
    for root in range(12):
        for quality in qualities:
            chord = Chord(root, quality)
            evidence = [energy[index] for index in chord.classes]
            distinct = sum(value > 1e-6 for value in evidence)
            if require_three_distinct and distinct < min(3, len(evidence)):
                continue
            matched = sum(evidence)
            root_score = energy[root]
            # Favor a balanced triad over a single-note root, but preserve
            # a separate clarity signal for listeners to inspect.
            balance = min(evidence) * len(evidence)
            score = max(0., min(1., .8 * matched + .2 * balance))
            result.append(ChordEvidence(chord, score, matched,
                                        root_score > 1e-6, 1 - matched))
    result.sort(key=lambda v: (-v.score, -v.coverage, v.chord.root, v.chord.quality))
    return tuple(result[:candidates])


@dataclass(frozen=True, slots=True)
class VoicingSettings:
    minimum_pitch: int = 48
    maximum_pitch: int = 84
    voices: int = 3
    maximum_spread: int = 36
    allow_doubling: bool = False
    no_crossing: bool = True
    prefer_common_tones: bool = True
    beam_width: int = 128

    def __post_init__(self):
        integer(self.minimum_pitch, "voicing low", 0, 127)
        integer(self.maximum_pitch, "voicing high", self.minimum_pitch, 127)
        integer(self.voices, "voicing voices", 2, 6)
        integer(self.maximum_spread, "maximum voicing spread", 5, 72)
        integer(self.beam_width, "voicing beam", 1, 1024)
        for name in ("allow_doubling", "no_crossing", "prefer_common_tones"):
            if type(getattr(self, name)) is not bool:
                raise MidiWorkstationError(f"{name} must be bool")


def candidate_voicings(chord: Chord, options: VoicingSettings = VoicingSettings()) -> tuple[tuple[int, ...], ...]:
    if not isinstance(chord, Chord) or not isinstance(options, VoicingSettings):
        raise MidiWorkstationError("Expected chord and voicing options")
    allowed = [pitch for pitch in range(options.minimum_pitch,
                                       options.maximum_pitch + 1)
               if pitch % 12 in chord.classes]
    if len(allowed) < options.voices:
        return ()
    result = []
    for pitches in combinations(allowed, options.voices):
        if pitches[-1] - pitches[0] > options.maximum_spread:
            continue
        classes = {p % 12 for p in pitches}
        if not options.allow_doubling and len(classes) < min(options.voices, len(chord.classes)):
            continue
        if any(cl not in classes for cl in chord.classes) and (
            options.voices >= len(chord.classes)
        ):
            continue
        result.append(pitches)
        if len(result) >= MAX_VOICING_CANDIDATES:
            break
    return tuple(result)


def voice_leading_cost(a: tuple[int, ...], b: tuple[int, ...],
                       options: VoicingSettings = VoicingSettings()) -> int:
    if (type(a) is not tuple or type(b) is not tuple or len(a) != len(b)
            or len(a) != options.voices
            or any(type(p) is not int or not 0 <= p <= 127 for p in (*a, *b))):
        raise MidiWorkstationError("Voice-leading pitches must match declared voices")
    if options.no_crossing and (list(a) != sorted(a) or list(b) != sorted(b)):
        raise MidiWorkstationError("Crossing voices are disallowed")
    motion = sum(abs(x - y) for x, y in zip(a, b))
    leap = sum(max(0, abs(x - y) - 7) ** 2 for x, y in zip(a, b))
    # Shared exact notes can be held; reward them by lowering cost.
    common = sum(x == y for x, y in zip(a, b))
    bonus = common * 4 if options.prefer_common_tones else 0
    return motion * 3 + leap - bonus


@dataclass(frozen=True, slots=True)
class VoicedChord:
    chord: Chord
    pitches: tuple[int, ...]
    transition_cost: int

    def __post_init__(self):
        if not isinstance(self.chord, Chord):
            raise MidiWorkstationError("VoicedChord requires Chord")
        if type(self.pitches) is not tuple or len(self.pitches) < 2:
            raise MidiWorkstationError("Voiced chord needs at least two voices")
        integer(self.transition_cost, "transition cost", -100, 10**9)


def arrange_voice_leading(chords: tuple[Chord, ...],
                          options: VoicingSettings = VoicingSettings()) -> tuple[VoicedChord, ...]:
    """Bounded dynamic programming across original progression chord labels."""
    if (type(chords) is not tuple or not 1 <= len(chords) <= 128
            or any(not isinstance(chord, Chord) for chord in chords)
            or not isinstance(options, VoicingSettings)):
        raise MidiWorkstationError("Expected bounded chord progression")
    all_candidates = []
    for chord in chords:
        values = candidate_voicings(chord, options)
        if not values:
            raise MidiWorkstationError("Chord has no voicing in requested register")
        # Limit before transitions to keep DP O(chords * beam^2).
        values = tuple(sorted(values, key=lambda v: (
            abs((v[0] + v[-1]) / 2 - (options.minimum_pitch + options.maximum_pitch) / 2), v,
        ))[:options.beam_width])
        all_candidates.append(values)
    previous = {candidate: (0, ()) for candidate in all_candidates[0]}
    for i in range(1, len(chords)):
        updated = {}
        for target in all_candidates[i]:
            choices = []
            for source, (cost, route) in previous.items():
                movement = voice_leading_cost(source, target, options)
                choices.append((cost + movement, route + (source,), source))
            total, route, _ = min(choices, key=lambda x: (x[0], x[1], x[2]))
            updated[target] = (total, route)
        previous = updated
    total, history, last = min(
        ((cost, path, target) for target, (cost, path) in previous.items()),
        key=lambda item: (item[0], item[1], item[2]),
    )
    timeline = history + (last,)
    return tuple(VoicedChord(chords[i], timeline[i], 0 if i == 0 else
                             voice_leading_cost(timeline[i - 1], timeline[i], options))
                 for i in range(len(chords)))


def transpose_to_scale(score: Score, scale: Scale, *, track_ids: tuple[str, ...] | None = None,
                       preserve_percussion: bool = True) -> tuple[Score, dict]:
    if not isinstance(score, Score) or not isinstance(scale, Scale):
        raise MidiWorkstationError("Expected Score and Scale")
    if track_ids is not None and (type(track_ids) is not tuple or
                                  not set(track_ids) <= {t.id for t in score.tracks}):
        raise MidiWorkstationError("Unknown scale correction tracks")
    if type(preserve_percussion) is not bool:
        raise MidiWorkstationError("Percussion policy must be bool")
    selected = {t.id for t in score.tracks} if track_ids is None else set(track_ids)
    modified, changed = [], 0
    for track in score.tracks:
        if track.id not in selected or (preserve_percussion and not track.pitched):
            modified.append(track)
            continue
        notes = []
        for note in track.notes:
            corrected = scale.closest(note.pitch)
            changed += int(corrected != note.pitch)
            notes.append(ScoreNote(note.id, note.start, note.duration, corrected,
                                   note.velocity, note.release_velocity, note.pan))
        modified.append(track.replace_notes(tuple(notes)))
    candidate = score.with_tracks(tuple(modified))
    return candidate, {
        "protocol": "sonumio-scale-correction-v1",
        "source_sha256": score.sha256, "result_sha256": candidate.sha256,
        "changed_pitches": changed, "scale": scale.mode, "tonic": scale.tonic,
        "percussion_preserved": preserve_percussion,
        "source_mutated": False, "model_inference": False,
    }
