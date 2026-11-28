"""Deterministic sustained-note source panels and stratified window schedules."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json

import numpy as np
import torch

from research.instrument_note_data import INSTRUMENTS, MIN_PITCH, MAX_PITCH, MIN_VELOCITY, MAX_VELOCITY
from research.instrument_long_model import MAX_NOTE_FRAMES, MAX_WINDOW
from sonumio.arrangement import Event
from sonumio.music_instruments import render_event

PROTOCOL = "sonumio-sustained-note-data-v1"
MIN_LENGTH = 12000
WINDOW = 2048
REGIONS = ("attack", "sustain", "release")
PANELS = ("train", "pitch", "velocity", "duration", "joint")
MAX_ROWS = 512
MAX_SOURCE_SAMPLES = 12_000_000
MAX_UPDATES = 384
MAX_BATCH = 8


def _int(value, lo, hi, label):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError("Invalid sustained note " + label)
    return value


def _axis(values, lo, hi):
    if (type(values) is not tuple or not 1 <= len(values) <= 8
            or any(type(v) is not int or not lo <= v <= hi for v in values)
            or tuple(sorted(set(values))) != values):
        raise ValueError("Invalid canonical sustained note grid axis")


@dataclass(frozen=True, slots=True)
class LongNote:
    instrument: str
    pitch: int
    velocity_milli: int
    frames: int

    def __post_init__(self):
        if self.instrument not in INSTRUMENTS:
            raise ValueError("Unknown sustained instrument")
        _int(self.pitch, MIN_PITCH, MAX_PITCH, "pitch")
        _int(self.velocity_milli, MIN_VELOCITY, MAX_VELOCITY, "velocity")
        _int(self.frames, MIN_LENGTH, MAX_NOTE_FRAMES, "duration")

    @property
    def identity(self):
        payload = json.dumps([self.instrument, self.pitch, self.velocity_milli, self.frames],
                             separators=(",", ":")).encode("ascii")
        return hashlib.sha256(PROTOCOL.encode("ascii") + b"\0" + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Example:
    note: LongNote
    raw: bytes

    def __post_init__(self):
        if not isinstance(self.note, LongNote) or type(self.raw) is not bytes or len(self.raw) != self.note.frames * 4:
            raise ValueError("Invalid sustained note source")
        if not np.isfinite(self.samples).all() or np.max(np.abs(self.samples)) > 1.000001 or not np.any(self.samples):
            raise ValueError("Invalid sustained note PCM")

    @property
    def samples(self):
        return np.frombuffer(self.raw, dtype="<f4")

    @property
    def sha256(self):
        return hashlib.sha256(self.raw).hexdigest()


@dataclass(frozen=True, slots=True)
class LongGrid:
    training_pitches: tuple = (48, 60, 72)
    evaluation_pitches: tuple = (54, 66)
    training_velocities: tuple = (350, 800)
    evaluation_velocities: tuple = (600,)
    training_frames: tuple = (12000, 36000)
    evaluation_frames: tuple = (24000, 48000)

    def __post_init__(self):
        for a, b, lo, hi in (
            (self.training_pitches, self.evaluation_pitches, MIN_PITCH, MAX_PITCH),
            (self.training_velocities, self.evaluation_velocities, MIN_VELOCITY, MAX_VELOCITY),
            (self.training_frames, self.evaluation_frames, MIN_LENGTH, MAX_NOTE_FRAMES),
        ):
            _axis(a, lo, hi)
            _axis(b, lo, hi)
            if set(a) & set(b):
                raise ValueError("Training and evaluation axes overlap")
        rows = [note for panel in PANELS for note in self.specs(panel)]
        if len(rows) > MAX_ROWS or sum(note.frames for note in rows) > MAX_SOURCE_SAMPLES:
            raise ValueError("Sustained note source budget exceeded")

    def specs(self, panel):
        if panel not in PANELS:
            raise ValueError("Unknown sustained note panel")
        pitches = self.evaluation_pitches if panel in ("pitch", "joint") else self.training_pitches
        velocities = self.evaluation_velocities if panel in ("velocity", "joint") else self.training_velocities
        lengths = self.evaluation_frames if panel in ("duration", "joint") else self.training_frames
        return tuple(LongNote(i, p, v, n) for i in INSTRUMENTS for p in pitches for v in velocities for n in lengths)


def render_example(note):
    if not isinstance(note, LongNote):
        raise ValueError("Expected LongNote")
    event = Event(note.instrument, note.pitch, 0, 1, note.velocity_milli / 1000., 0.)
    wave = render_event(event, note.frames, 0)
    return Example(note, np.ascontiguousarray(wave, dtype="<f4").tobytes())


def panel_digest(rows):
    trace = hashlib.sha256(PROTOCOL.encode("ascii") + b"\0panel\0")
    for row in rows:
        trace.update(bytes.fromhex(row.note.identity))
        trace.update(bytes.fromhex(row.sha256))
    return trace.hexdigest()


@dataclass(frozen=True, slots=True)
class Corpus:
    grid: LongGrid
    panels: tuple

    def __post_init__(self):
        if not isinstance(self.grid, LongGrid) or type(self.panels) is not tuple or len(self.panels) != len(PANELS):
            raise ValueError("Invalid sustained source corpus")
        identities = set()
        train_pcm = set()
        for name, rows in zip(PANELS, self.panels, strict=True):
            if (type(rows) is not tuple or any(not isinstance(r, Example) for r in rows)
                    or tuple(r.note for r in rows) != self.grid.specs(name)):
                raise ValueError("Sustained panel differs from prescribed grid")
            ids = {r.note.identity for r in rows}
            pcm = {r.sha256 for r in rows}
            if len(ids) != len(rows) or ids & identities:
                raise ValueError("Sustained source identity overlap")
            identities.update(ids)
            if name == "train":
                train_pcm = pcm
            elif pcm & train_pcm:
                raise ValueError("Sustained source PCM overlaps training")

    def get(self, panel):
        return self.panels[PANELS.index(panel)]

    def receipt(self):
        return {
            name: {"notes": len(rows), "samples": sum(r.note.frames for r in rows),
                   "sha256": panel_digest(rows)}
            for name, rows in zip(PANELS, self.panels, strict=True)
        }


def build_corpus(grid=None):
    grid = LongGrid() if grid is None else grid
    if not isinstance(grid, LongGrid):
        raise ValueError("Expected LongGrid")
    return Corpus(grid, tuple(tuple(render_example(note) for note in grid.specs(name)) for name in PANELS))


def region_span(frames, region):
    _int(frames, MIN_LENGTH, MAX_NOTE_FRAMES, "region length")
    if region not in REGIONS:
        raise ValueError("Unknown sustained note region")
    edge = frames // 3
    return {"attack": (0, edge), "sustain": (edge, 2 * edge), "release": (2 * edge, frames)}[region]


@dataclass(frozen=True, slots=True)
class WindowBatch:
    indices: tuple[int, ...]
    frames: int
    start: int
    count: int
    region: str

    def __post_init__(self):
        if (type(self.indices) is not tuple or not 1 <= len(self.indices) <= MAX_BATCH
                or any(type(i) is not int or not 0 <= i < MAX_ROWS for i in self.indices)):
            raise ValueError("Invalid sustained window batch indices")
        _int(self.count, 1024, MAX_WINDOW, "window count")
        lo, hi = region_span(self.frames, self.region)
        if type(self.start) is not int or not lo <= self.start <= hi - self.count:
            raise ValueError("Window crosses its prescribed region")


def training_schedule(rows, *, updates, batch_size, seed):
    _int(updates, 1, MAX_UPDATES, "updates")
    _int(batch_size, 1, MAX_BATCH, "batch size")
    _int(seed, 0, 2**32 - 1, "sampler seed")
    if type(rows) is not tuple or not rows or any(not isinstance(r, Example) for r in rows):
        raise ValueError("Expected sustained training examples")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.note.frames].append(i)
    lengths = tuple(sorted(groups))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4C4F4E47]))
    pending = {length: [] for length in lengths}
    schedule = []
    trace = hashlib.sha256(PROTOCOL.encode("ascii") + b"\0schedule\0")
    for step in range(updates):
        length = lengths[(step // len(REGIONS)) % len(lengths)]
        region = REGIONS[step % len(REGIONS)]
        lo, hi = region_span(length, region)
        while len(pending[length]) < batch_size:
            pending[length].extend(int(i) for i in rng.permutation(groups[length]))
        indices = tuple(pending[length][:batch_size])
        del pending[length][:batch_size]
        # Always observe the actual note edges in attack/release regions.
        start = lo if region == "attack" else hi-WINDOW if region == "release" else int(rng.integers(lo, hi-WINDOW+1))
        batch = WindowBatch(indices, length, start, WINDOW, region)
        schedule.append(batch)
        trace.update(json.dumps([length, start, WINDOW, region], separators=(",", ":")).encode("ascii"))
        for index in indices:
            trace.update(bytes.fromhex(rows[index].note.identity))
    return tuple(schedule), trace.hexdigest()


def collate(rows, batch):
    if not isinstance(batch, WindowBatch):
        raise ValueError("Expected WindowBatch")
    if any(i >= len(rows) for i in batch.indices):
        raise ValueError("Unknown training note index")
    selected = tuple(rows[i] for i in batch.indices)
    if any(r.note.frames != batch.frames for r in selected):
        raise ValueError("Sustained note batch length mismatch")
    labels = torch.tensor([INSTRUMENTS.index(r.note.instrument) for r in selected], dtype=torch.long)
    pitches = torch.tensor([r.note.pitch for r in selected], dtype=torch.long)
    velocities = torch.tensor([r.note.velocity_milli for r in selected], dtype=torch.long)
    targets = torch.from_numpy(np.stack([r.samples[batch.start:batch.start+batch.count] for r in selected]).astype(np.float32))
    return labels, pitches, velocities, targets
