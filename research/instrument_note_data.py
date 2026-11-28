"""First-party synthetic note corpus for learned harmonic instrument research.

The targets are existing Sonumio procedural instrument waveforms, not
recorded musical performances. Splits are explicit pitch/velocity/duration
axes; every evaluation panel holds out at least one complete control axis.
No audio files, external datasets or trained weights are written.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from sonumio.arrangement import Event, PITCHED
from sonumio.music_instruments import render_event

PROTOCOL = "sonumio-instrument-note-grid-v1"
INSTRUMENTS = tuple(PITCHED)
MIN_PITCH = 36
MAX_PITCH = 88
MIN_VELOCITY = 100
MAX_VELOCITY = 1000
MIN_FRAMES = 2048
MAX_FRAMES = 8192
MAX_ROWS = 1024
MAX_SOURCE_SAMPLES = 10_000_000
MAX_BATCH = 16
MAX_UPDATES = 512
PANELS = ("train", "pitch", "velocity", "duration", "joint")


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid instrument " + label)
    return value


def _axis(values, low, high, label, max_count=16):
    if (type(values) is not tuple or not 1 <= len(values) <= max_count
            or any(type(v) is not int or not low <= v <= high for v in values)
            or tuple(sorted(set(values))) != values):
        raise ValueError("Invalid canonical instrument " + label + " axis")
    return values


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


@dataclass(frozen=True, slots=True)
class NoteSpec:
    instrument: str
    pitch: int
    velocity_milli: int
    frames: int

    def __post_init__(self):
        if type(self.instrument) is not str or self.instrument not in INSTRUMENTS:
            raise ValueError("Unknown pitched instrument")
        _integer(self.pitch, MIN_PITCH, MAX_PITCH, "pitch")
        _integer(self.velocity_milli, MIN_VELOCITY, MAX_VELOCITY, "velocity")
        _integer(self.frames, MIN_FRAMES, MAX_FRAMES, "note frame count")

    @property
    def instrument_id(self):
        return INSTRUMENTS.index(self.instrument)

    @property
    def identity(self):
        return _hash(PROTOCOL.encode("ascii") + b"\0" + _canonical(
            [self.instrument, self.pitch, self.velocity_milli, self.frames]
        ))

    @property
    def velocity(self):
        return self.velocity_milli / 1000.


@dataclass(frozen=True, slots=True)
class NoteExample:
    spec: NoteSpec
    pcm_le_f32: bytes

    def __post_init__(self):
        if not isinstance(self.spec, NoteSpec) or type(self.pcm_le_f32) is not bytes:
            raise ValueError("Invalid immutable note example")
        if len(self.pcm_le_f32) != 4 * self.spec.frames:
            raise ValueError("Instrument target byte count differs from note")
        values = self.samples
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.000001:
            raise ValueError("Invalid instrument waveform values")
        if not np.any(values):
            raise ValueError("Unexpected silent pitched training target")

    @property
    def samples(self):
        return np.frombuffer(self.pcm_le_f32, dtype="<f4")

    @property
    def waveform_sha256(self):
        return _hash(self.pcm_le_f32)


@dataclass(frozen=True, slots=True)
class GridPlan:
    training_pitches: tuple[int, ...] = (48, 60, 72, 84)
    evaluation_pitches: tuple[int, ...] = (54, 66, 78)
    training_velocities: tuple[int, ...] = (300, 700, 950)
    evaluation_velocities: tuple[int, ...] = (500, 850)
    training_frames: tuple[int, ...] = (4096, 8192)
    evaluation_frames: tuple[int, ...] = (6144,)

    def __post_init__(self):
        for left, right, low, high, name in (
            (self.training_pitches, self.evaluation_pitches, MIN_PITCH, MAX_PITCH, "pitch"),
            (self.training_velocities, self.evaluation_velocities,
             MIN_VELOCITY, MAX_VELOCITY, "velocity"),
            (self.training_frames, self.evaluation_frames, MIN_FRAMES, MAX_FRAMES, "duration"),
        ):
            _axis(left, low, high, name)
            _axis(right, low, high, name)
            if set(left) & set(right):
                raise ValueError("Training and evaluation " + name + " axes overlap")
        rows = 0
        samples = 0
        for panel in PANELS:
            pitches, velocities, durations = self.axes(panel)
            multiplier = len(INSTRUMENTS) * len(pitches) * len(velocities)
            rows += multiplier * len(durations)
            samples += multiplier * sum(durations)
        if rows > MAX_ROWS or samples > MAX_SOURCE_SAMPLES:
            raise ValueError("Instrument source grid exceeds row or sample budget")

    def axes(self, panel):
        if type(panel) is not str or panel not in PANELS:
            raise ValueError("Unknown instrument evaluation panel")
        return (
            self.evaluation_pitches if panel in ("pitch", "joint") else self.training_pitches,
            self.evaluation_velocities if panel in ("velocity", "joint") else self.training_velocities,
            self.evaluation_frames if panel in ("duration", "joint") else self.training_frames,
        )

    def specifications(self, panel):
        pitches, velocities, durations = self.axes(panel)
        return tuple(
            NoteSpec(instrument, pitch, velocity, frames)
            for instrument in INSTRUMENTS
            for pitch in pitches
            for velocity in velocities
            for frames in durations
        )


def render_example(spec):
    if not isinstance(spec, NoteSpec):
        raise ValueError("Expected NoteSpec")
    # Pitched source renderers are deterministic and do not use the seed.
    event = Event(spec.instrument, spec.pitch, 0, 1, spec.velocity, 0.)
    waveform = render_event(event, spec.frames, 0)
    return NoteExample(spec, np.ascontiguousarray(waveform, dtype="<f4").tobytes())


def panel_digest(examples):
    if (type(examples) is not tuple or not 1 <= len(examples) <= MAX_ROWS
            or any(not isinstance(x, NoteExample) for x in examples)):
        raise ValueError("Expected bounded note panel")
    trace = hashlib.sha256(PROTOCOL.encode("ascii") + b"\0panel\0")
    for example in examples:
        trace.update(bytes.fromhex(example.spec.identity))
        trace.update(bytes.fromhex(example.waveform_sha256))
    return trace.hexdigest()


@dataclass(frozen=True, slots=True)
class NoteCorpus:
    plan: GridPlan
    panels: tuple[tuple[NoteExample, ...], ...]

    def __post_init__(self):
        if not isinstance(self.plan, GridPlan):
            raise ValueError("Expected GridPlan")
        if type(self.panels) is not tuple or len(self.panels) != len(PANELS):
            raise ValueError("Expected all instrument evaluation panels")
        all_ids = set()
        train_waveforms = set()
        for name, rows in zip(PANELS, self.panels, strict=True):
            if (type(rows) is not tuple or any(not isinstance(row, NoteExample) for row in rows)
                    or tuple(row.spec for row in rows) != self.plan.specifications(name)):
                raise ValueError("Instrument panel differs from prescribed grid")
            ids = {row.spec.identity for row in rows}
            if len(ids) != len(rows) or ids & all_ids:
                raise ValueError("Duplicate instrument note specification")
            all_ids.update(ids)
            hashes = {row.waveform_sha256 for row in rows}
            if name == "train":
                train_waveforms = hashes
            elif train_waveforms & hashes:
                raise ValueError("Instrument waveform crosses train/evaluation split")

    def panel(self, name):
        if type(name) is not str or name not in PANELS:
            raise ValueError("Unknown instrument panel")
        return self.panels[PANELS.index(name)]

    def report(self):
        reports = {}
        for name in PANELS:
            examples = self.panel(name)
            pitches, velocities, durations = self.plan.axes(name)
            reports[name] = {
                "rows": len(examples), "samples": sum(x.spec.frames for x in examples),
                "sha256": panel_digest(examples),
                "unique_waveforms": len({x.waveform_sha256 for x in examples}),
                "pitches": list(pitches), "velocities_milli": list(velocities),
                "frames": list(durations),
                "instruments": list(INSTRUMENTS),
            }
        return {
            "protocol": PROTOCOL, "panels": reports,
            "cross_split_exact_waveform_overlap": False,
            "cross_split_note_identity_overlap": False,
            "external_recordings_used": False,
            "source": "Sonumio original procedural pitched instrument engine",
        }


def build_corpus(plan=None):
    plan = GridPlan() if plan is None else plan
    if not isinstance(plan, GridPlan):
        raise ValueError("Expected GridPlan")
    panels = tuple(tuple(render_example(spec) for spec in plan.specifications(name))
                   for name in PANELS)
    return NoteCorpus(plan, panels)


@dataclass(frozen=True, slots=True)
class NoteBatch:
    instruments: torch.Tensor
    pitches: torch.Tensor
    velocities: torch.Tensor
    frames: int
    targets: torch.Tensor
    identities: tuple[str, ...]


def collate(examples):
    if (type(examples) not in (tuple, list) or not 1 <= len(examples) <= MAX_BATCH
            or any(not isinstance(x, NoteExample) for x in examples)):
        raise ValueError("Invalid instrument note batch")
    frames = examples[0].spec.frames
    if any(row.spec.frames != frames for row in examples):
        raise ValueError("Instrument batch must share exact note frame count")
    return NoteBatch(
        torch.tensor([row.spec.instrument_id for row in examples], dtype=torch.long),
        torch.tensor([row.spec.pitch for row in examples], dtype=torch.long),
        torch.tensor([row.spec.velocity_milli for row in examples], dtype=torch.long),
        frames,
        torch.from_numpy(np.stack([row.samples for row in examples]).astype(np.float32)),
        tuple(row.spec.identity for row in examples),
    )


def _groups(examples):
    if (type(examples) is not tuple or not 1 <= len(examples) <= MAX_ROWS
            or any(not isinstance(x, NoteExample) for x in examples)):
        raise ValueError("Expected bounded note rows")
    groups = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.spec.frames, []).append(index)
    return {length: tuple(groups[length]) for length in sorted(groups)}


def training_schedule(examples, *, updates, batch_size, seed):
    """Cycle note lengths; shuffle each pool without replacement per epoch."""
    _integer(updates, 1, MAX_UPDATES, "training update count")
    _integer(batch_size, 1, MAX_BATCH, "batch size")
    _integer(seed, 0, 2**32 - 1, "sampler seed")
    groups = _groups(examples)
    lengths = tuple(groups)
    rng = {
        length: np.random.default_rng(np.random.SeedSequence([seed, length, 0x4E4F5445]))
        for length in lengths
    }
    pending = {length: [] for length in lengths}
    schedule = []
    for step in range(updates):
        length = lengths[step % len(lengths)]
        while len(pending[length]) < batch_size:
            pending[length].extend(int(i) for i in rng[length].permutation(groups[length]))
        schedule.append(tuple(pending[length][:batch_size]))
        del pending[length][:batch_size]
    trace = hashlib.sha256(PROTOCOL.encode("ascii") + b"\0schedule\0")
    for batch in schedule:
        trace.update(len(batch).to_bytes(2, "little"))
        for i in batch:
            trace.update(bytes.fromhex(examples[i].spec.identity))
    return tuple(schedule), trace.hexdigest()


def evaluation_batches(examples, batch_size):
    _integer(batch_size, 1, MAX_BATCH, "evaluation batch size")
    for indices in _groups(examples).values():
        for start in range(0, len(indices), batch_size):
            yield tuple(indices[start:start + batch_size])
