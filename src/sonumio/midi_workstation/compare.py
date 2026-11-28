"""Reproducible timing/voice comparisons of two independent symbolic Scores.

A bounded dynamic program aligns like-pitch notes in temporal order within
explicit track/channel groups. The resulting edit distance is an objective
symbolic diagnostic, not a perceptual or musical quality score. No note
is created, deleted, or changed by comparison.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from .contracts import MidiLimitError, MidiWorkstationError, integer
from .score import Score, ScoreNote

MAX_COMPARE_STATES = 2_000_000
MAX_REPORT_MATCHES = 4096


@dataclass(frozen=True, slots=True)
class CompareSettings:
    onset_tolerance_ticks: int = 120
    release_tolerance_ticks: int = 120
    pitch_mode: str = "exact"
    route_mode: str = "track"
    include_muted: bool = False
    maximum_matches: int = MAX_REPORT_MATCHES

    def __post_init__(self):
        integer(self.onset_tolerance_ticks, "onset tolerance", 0, 8192)
        integer(self.release_tolerance_ticks, "release tolerance", 0, 8192)
        integer(self.maximum_matches, "match report cap", 1, MAX_REPORT_MATCHES)
        if self.pitch_mode not in ("exact", "class"):
            raise MidiWorkstationError("Unknown comparison pitch matching mode")
        if self.route_mode not in ("track", "channel"):
            raise MidiWorkstationError("Unknown comparison route policy")
        if type(self.include_muted) is not bool:
            raise MidiWorkstationError("Comparison mute policy must be boolean")


@dataclass(frozen=True, slots=True)
class NoteMatch:
    source_track: str
    target_track: str
    source_id: str
    target_id: str
    source_pitch: int
    target_pitch: int
    onset_difference: int
    release_difference: int
    velocity_difference: int

    @property
    def matching_pitch_class(self):
        return self.source_pitch % 12 == self.target_pitch % 12

    @property
    def same_midi_note(self):
        return self.source_pitch == self.target_pitch


@dataclass(frozen=True, slots=True)
class CompareReport:
    source_sha256: str
    target_sha256: str
    source_notes: int
    target_notes: int
    matched_notes: int
    missing_notes: int
    additional_notes: int
    mean_onset_error_ticks: float
    maximum_onset_error_ticks: int
    mean_release_error_ticks: float
    maximum_release_error_ticks: int
    mean_velocity_error: float
    exact_pitch_matches: int
    pitch_class_matches: int
    comparison_groups: int
    states_evaluated: int
    matches: tuple[NoteMatch, ...]
    protocol: str = "sonumio-workstation-symbolic-comparison-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "match_algorithm": "bounded_monotone_dynamic_programming",
            "quality_calibrated": False,
            "source_mutated": False,
            "model_inference": False,
        }


@dataclass(frozen=True, slots=True)
class _TaggedNote:
    track_id: str
    note: ScoreNote


def _groups(score: Score, options: CompareSettings):
    result = defaultdict(list)
    active = {track.id for track in score.tracks} if options.include_muted else set(
        score.active_track_ids
    )
    for track in score.tracks:
        if track.id not in active:
            continue
        route = track.id if options.route_mode == "track" else track.channel
        for note in track.notes:
            pitch = note.pitch if options.pitch_mode == "exact" else note.pitch % 12
            result[(route, pitch)].append(_TaggedNote(track.id, note))
    for key in result:
        result[key].sort(key=lambda item: (
            item.note.start, item.note.stop, item.note.id, item.track_id,
        ))
    return result


def _align(left: list[_TaggedNote], right: list[_TaggedNote],
           options: CompareSettings):
    """Needleman-Wunsch with match-only edges inside explicit time windows."""
    n, m = len(left), len(right)
    count = (n + 1) * (m + 1)
    if count > MAX_COMPARE_STATES:
        raise MidiLimitError("Comparison dynamic-programming work cap exceeded")
    skip_cost = options.onset_tolerance_ticks + options.release_tolerance_ticks + 128
    sentinel = 2**60
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * skip_cost
        trace[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j * skip_cost
        trace[0][j] = 2
    for i in range(1, n + 1):
        a = left[i - 1].note
        for j in range(1, m + 1):
            b = right[j - 1].note
            onset = abs(b.start - a.start)
            release = abs(b.stop - a.stop)
            substitution = sentinel
            if (onset <= options.onset_tolerance_ticks
                    and release <= options.release_tolerance_ticks):
                substitution = (dp[i - 1][j - 1] + onset + release
                                + abs(a.velocity - b.velocity))
            # Ties prefer a legitimate note pairing, then an input-side
            # deletion, then a target-side insertion. This is deterministic.
            best = min(
                (substitution, 0),
                (dp[i - 1][j] + skip_cost, 1),
                (dp[i][j - 1] + skip_cost, 2),
            )
            dp[i][j], trace[i][j] = best
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        decision = trace[i][j]
        if decision == 0:
            a, b = left[i - 1], right[j - 1]
            pairs.append(NoteMatch(
                a.track_id, b.track_id, a.note.id, b.note.id,
                a.note.pitch, b.note.pitch,
                b.note.start - a.note.start,
                b.note.stop - a.note.stop,
                b.note.velocity - a.note.velocity,
            ))
            i -= 1
            j -= 1
        elif decision == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs, count


def compare_scores(source: Score, target: Score,
                   options: CompareSettings = CompareSettings()) -> CompareReport:
    if (not isinstance(source, Score) or not isinstance(target, Score)
            or not isinstance(options, CompareSettings)):
        raise MidiWorkstationError("Expected typed Scores and CompareSettings")
    if source.ppqn != target.ppqn:
        raise MidiWorkstationError("Symbolic comparison requires identical PPQN")
    if options.route_mode == "channel" and any(
        track.channel == 9 for score in (source, target) for track in score.tracks
    ) and options.pitch_mode == "class":
        raise MidiWorkstationError("Percussion should not use pitch-class matching")
    a = _groups(source, options)
    b = _groups(target, options)
    all_matches = []
    states = 0
    for key in sorted(set(a) | set(b), key=lambda item: (str(item[0]), item[1])):
        matches, operations = _align(a.get(key, []), b.get(key, []), options)
        states += operations
        if states > MAX_COMPARE_STATES:
            raise MidiLimitError("Cumulative comparison work cap exceeded")
        if len(all_matches) + len(matches) > options.maximum_matches:
            raise MidiLimitError("Comparison exceeds bounded reported matches")
        all_matches.extend(matches)
    all_matches.sort(key=lambda row: (
        row.source_track, row.source_id, row.target_track, row.target_id,
    ))
    source_count = sum(len(value) for value in a.values())
    target_count = sum(len(value) for value in b.values())
    matched = len(all_matches)
    def mean_abs(attribute):
        return (sum(abs(getattr(note, attribute)) for note in all_matches) / matched
                if matched else 0.)
    def max_abs(attribute):
        return max((abs(getattr(note, attribute)) for note in all_matches), default=0)
    return CompareReport(
        source.sha256, target.sha256, source_count, target_count,
        matched, source_count - matched, target_count - matched,
        mean_abs("onset_difference"), max_abs("onset_difference"),
        mean_abs("release_difference"), max_abs("release_difference"),
        mean_abs("velocity_difference"),
        sum(note.same_midi_note for note in all_matches),
        sum(note.matching_pitch_class for note in all_matches),
        len(set(a) | set(b)), states, tuple(all_matches),
    )
