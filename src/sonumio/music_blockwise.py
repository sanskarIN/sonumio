"""Bounded-block rendering of original symbolic music.

The note clock, float32 per-track accumulation, track summation order,
finite room taps, edge ramps and PCM16 quantization match the existing
whole-array renderer. Only the final bounded WAV byte buffer is retained.
Two deterministic passes establish the global peak and encode the result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
import wave

import numpy as np

from .arrangement import Arrangement, TRACKS, MAX_FRAMES, arrangement_digest
from .audio import SAMPLE_RATE
from .errors import GenerationError
from .music_instruments import MAX_NOTE_FRAMES, render_event
from .music_mix import MixPlan

PROTOCOL = "sonumio-blockwise-music-v1"
MIN_BLOCK = 256
MAX_BLOCK = 16384
DEFAULT_BLOCK = 8192
MAX_ACTIVE_NOTES = 14
ROOM_TAPS = tuple((round(ms * SAMPLE_RATE / 1000), gain, cross)
                  for ms, gain, cross in ((29,.23,False),(43,.19,True),(71,.14,True),
                                         (113,.09,False),(173,.06,True),(251,.03,False)))
ROOM_HISTORY = max(tap[0] for tap in ROOM_TAPS)


def _validate(score, block_frames):
    if not isinstance(score, Arrangement):
        raise GenerationError("Expected Arrangement for blockwise rendering")
    if type(block_frames) is not int or not MIN_BLOCK <= block_frames <= MAX_BLOCK:
        raise GenerationError("Invalid music block size")
    frames = score.plan.frame(score.plan.ticks)
    if not 1 <= frames <= MAX_FRAMES:
        raise GenerationError("Blockwise music exceeds duration budget")
    return frames


@dataclass(frozen=True, slots=True)
class _Voice:
    track: str
    start: int
    stop: int
    pcm: np.ndarray

    def add(self, stem, start, stop):
        left = max(start, self.start)
        right = min(stop, self.stop)
        if left < right:
            stem[:, left-start:right-start] += self.pcm[:, left-self.start:right-self.start]


def _voice(score, event, start, stop):
    frames = stop - start
    if not 2 <= frames <= MAX_NOTE_FRAMES:
        raise GenerationError("Blockwise note exceeds synthesis budget")
    mono = render_event(event, frames, score.plan.seed)
    if (not isinstance(mono, np.ndarray) or mono.dtype != np.float64
            or mono.shape != (frames,) or not np.isfinite(mono).all()
            or np.max(np.abs(mono)) > 1.000001):
        raise GenerationError("Invalid blockwise note waveform")
    pan = (event.pan + 1) * math.pi / 4
    pcm = np.empty((2, frames), dtype=np.float32)
    pcm[0] = (mono * math.cos(pan)).astype(np.float32)
    pcm[1] = (mono * math.sin(pan)).astype(np.float32)
    return _Voice(event.track, start, stop, pcm)


def iter_stem_blocks(score, block_frames=DEFAULT_BLOCK):
    """Yield (absolute start frame, track blocks) in exact event order.

    Voices that cross block boundaries are retained at most once. Voices
    ending within the current block are released immediately. The validated
    arrangement limits simultaneous polyphony to fourteen notes.
    """
    total = _validate(score, block_frames)
    cursor = 0
    active = []
    events = score.events
    for start in range(0, total, block_frames):
        stop = min(total, start + block_frames)
        stems = {track: np.zeros((2, stop-start), dtype=np.float32) for track in TRACKS}
        retained = []
        for voice in active:
            voice.add(stems[voice.track], start, stop)
            if voice.stop > stop:
                retained.append(voice)
        voice = None
        active = []
        while cursor < len(events):
            event = events[cursor]
            onset = score.plan.frame(event.tick)
            if onset >= stop:
                break
            if onset < start:
                raise GenerationError("Blockwise score traversal skipped an onset")
            end = score.plan.frame(event.tick + event.length)
            voice = _voice(score, event, onset, end)
            voice.add(stems[event.track], start, stop)
            if end > stop:
                retained.append(voice)
            voice = None
            cursor += 1
        if len(retained) > MAX_ACTIVE_NOTES:
            raise GenerationError("Blockwise active voice budget exceeded")
        active = retained
        for stem in stems.values():
            if not np.isfinite(stem).all():
                raise GenerationError("Nonfinite blockwise music stem")
            stem.setflags(write=False)
        yield start, stems
    if cursor != len(events) or active:
        raise GenerationError("Incomplete blockwise score traversal")


class _Room:
    """Apply the original finite cross-channel tap sequence with bounded history."""

    def __init__(self, amount, total):
        self.amount = amount
        self.total = total
        self.history = np.empty((2, 0), dtype=np.float64)
        self.cursor = 0

    def apply(self, start, dry):
        if start != self.cursor or dry.ndim != 2 or dry.shape[0] != 2:
            raise GenerationError("Noncontiguous blockwise room input")
        count = dry.shape[1]
        stop = start + count
        if count <= 0 or stop > self.total:
            raise GenerationError("Invalid blockwise room geometry")
        prefix = self.history.shape[1]
        joined = np.concatenate((self.history, dry), axis=1)
        origin = start - prefix
        result = dry.copy()
        for delay, gain, cross in ROOM_TAPS:
            if delay >= self.total:
                continue
            left = max(start, delay)
            if left >= stop:
                continue
            source_start = left - delay - origin
            source_stop = stop - delay - origin
            if not 0 <= source_start < source_stop <= joined.shape[1]:
                raise GenerationError("Insufficient blockwise room history")
            source = joined[:, source_start:source_stop]
            if cross:
                source = source[::-1]
            result[:, left-start:] += source * (gain * self.amount)
        self.history = joined[:, -ROOM_HISTORY:].copy()
        self.cursor = stop
        return result


class _Edges:
    """Generate the original global fade-in/out without a full-song array."""

    def __init__(self, frames):
        a = min(240, frames // 2)
        r = min(1200, frames - a)
        self.frames = frames
        self.attack = np.sin(np.linspace(0, math.pi/2, a, endpoint=False))**2
        if r == 1:
            self.release = np.zeros(1, dtype=np.float64)
        else:
            self.release = np.cos(np.linspace(0, math.pi/2, r))**2

    def apply(self, start, wet):
        stop = start + wet.shape[1]
        count = wet.shape[1]
        ramp = np.ones(count, dtype=np.float64)
        attack_stop = min(stop, len(self.attack))
        if start < attack_stop:
            ramp[:attack_stop-start] = self.attack[start:attack_stop]
        release_start = self.frames - len(self.release)
        left = max(start, release_start)
        if left < stop:
            ramp[left-start:] = self.release[left-release_start:stop-release_start]
        wet *= ramp


class _Stats:
    """Float64 block accumulation; no full waveform copy for diagnostics."""

    def __init__(self):
        self.samples = 0
        self.energy = 0.
        self.peak = 0.
        self.clipped = 0

    def add(self, data):
        x = data.astype(np.float64, copy=False)
        self.samples += x.size
        self.energy += float(np.sum(x*x, dtype=np.float64))
        self.peak = max(self.peak, float(np.max(np.abs(x))))
        self.clipped += int(np.count_nonzero(np.abs(x) >= 1.))

    @property
    def rms(self):
        if not self.samples:
            raise GenerationError("Empty blockwise audio statistics")
        return math.sqrt(self.energy / self.samples)

    def metrics(self):
        return {"peak_dbfs": round(20 * math.log10(max(self.peak, 1e-12)), 3),
                "rms_dbfs": round(20 * math.log10(max(self.rms, 1e-12)), 3),
                "clipping_fraction": self.clipped / self.samples}


def iter_wet_blocks(score, mix, block_frames=DEFAULT_BLOCK, *, track_stats=None):
    total = _validate(score, block_frames)
    if not isinstance(mix, MixPlan):
        raise GenerationError("Expected MixPlan for blockwise music")
    room = _Room(mix.room, total)
    edges = _Edges(total)
    for start, stems in iter_stem_blocks(score, block_frames):
        count = stems[TRACKS[0]].shape[1]
        dry = np.zeros((2, count), dtype=np.float64)
        for track, gain in zip(TRACKS, mix.gains, strict=True):
            stem = stems[track]
            if track_stats is not None:
                track_stats[track].add(stem)
            if mix.active(track):
                dry += stem.astype(np.float64) * gain
        wet = room.apply(start, dry)
        edges.apply(start, wet)
        if not np.isfinite(wet).all():
            raise GenerationError("Nonfinite blockwise wet music")
        yield start, wet
    if room.cursor != total:
        raise GenerationError("Incomplete blockwise room traversal")


def _trace_update(trace, start, data):
    trace.update(start.to_bytes(8, "little"))
    trace.update(data.shape[1].to_bytes(4, "little"))
    trace.update(np.ascontiguousarray(data, dtype="<f8").tobytes())


def generate_blockwise_music(score, mix=None, *, block_frames=DEFAULT_BLOCK):
    """Return canonical stereo WAV bytes and a verified two-pass receipt.

    The first pass computes global peak and stem statistics. The second
    recomputes the identical unscaled stream, checks its SHA256 and encodes
    globally attenuated PCM16 into the bounded WAV buffer. No disk temporary
    PCM, full-song float stems, or trained model is required.
    """
    frames = _validate(score, block_frames)
    mix = MixPlan() if mix is None else mix
    if not isinstance(mix, MixPlan):
        raise GenerationError("Expected MixPlan")
    tracks = {track: _Stats() for track in TRACKS}
    trace = hashlib.sha256()
    pre_peak = 0.
    first_count = 0
    for start, wet in iter_wet_blocks(score, mix, block_frames, track_stats=tracks):
        _trace_update(trace, start, wet)
        pre_peak = max(pre_peak, float(np.max(np.abs(wet))))
        first_count += wet.shape[1]
    if first_count != frames:
        raise GenerationError("Incomplete first music render pass")
    scale = min(1., mix.peak_ceiling / max(pre_peak, 1e-15))
    first_digest = trace.hexdigest()
    second = hashlib.sha256()
    stats = _Stats()
    output = io.BytesIO()
    encoded_frames = 0
    with wave.open(output, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.setnframes(frames)
        for start, wet in iter_wet_blocks(score, mix, block_frames):
            _trace_update(second, start, wet)
            if start != encoded_frames:
                raise GenerationError("Noncontiguous blockwise PCM encoding")
            audio = (wet * scale).astype(np.float32)
            stats.add(audio)
            pcm = (np.clip(audio, -1., 1.) * 32767.).round().astype("<i2")
            writer.writeframesraw(pcm.T.tobytes(order="C"))
            encoded_frames += audio.shape[1]
    if encoded_frames != frames or second.hexdigest() != first_digest:
        raise GenerationError("Blockwise music replay differs from peak pass")
    wav = output.getvalue()
    if len(wav) != 44 + frames * 4:
        raise GenerationError("Blockwise WAV byte geometry mismatch")
    report = {
        "protocol": PROTOCOL, "sample_rate": SAMPLE_RATE, "channels": 2, "frames": frames,
        "score_sha256": arrangement_digest(score), "wav_sha256": hashlib.sha256(wav).hexdigest(),
        "pre_master_peak": pre_peak, "master_attenuation": scale,
        "room_amount": float(mix.room), "peak_ceiling": float(mix.peak_ceiling),
        "metrics": stats.metrics(),
        "tracks": {track: {"active": mix.active(track), "gain": float(gain),
                           "dry_peak": tracks[track].peak, "dry_rms": tracks[track].rms}
                   for track, gain in zip(TRACKS, mix.gains, strict=True)},
        "block_frames": block_frames, "render_passes": 2,
        "unscaled_stream_sha256": first_digest, "replay_verified": True,
        "retains_full_float_stems": False, "retains_final_wav_bytes": True,
        "max_active_note_limit": MAX_ACTIVE_NOTES, "room_history_frames": ROOM_HISTORY,
        "true_peak_measured": False, "lufs_measured": False,
        "learned_model": False, "external_recordings_used": False,
    }
    return wav, report
