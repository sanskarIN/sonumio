"""Score-clock gain and stereo-balance automation for original music.

This offline mixer reuses the existing dry stems and room model. Automation
is applied before room reflections and the final sample-peak attenuation.
It neither changes note generation nor trains a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .arrangement import Arrangement, MAX_BARS, MAX_FRAMES, TRACKS, arrangement_digest
from .audio import SAMPLE_RATE, encode_wav_channels, metrics_channels
from .errors import GenerationError
from .music_instruments import smooth_edges
from .music_mix import MixPlan, MAX_STEM_BYTES, render_stems, room_response

PROTOCOL = "sonumio-mix-automation-v1"
MAX_KEYS = 64
MAX_TICKS = MAX_BARS * 32
MAX_PLAN_BYTES = 128 * 1024
BLOCK_FRAMES = 16384
LAWS = ("linear", "smoothstep", "step")
PARAMETERS = ("gain", "balance")


def _real(value, minimum, maximum, label):
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise GenerationError("Invalid automation " + label)
    return float(value)


@dataclass(frozen=True, slots=True)
class MixKey:
    """A score tick and value; interpolation belongs to the outgoing segment."""

    tick: int
    value: float
    law: str = "linear"

    def __post_init__(self):
        if type(self.tick) is not int or not 0 <= self.tick <= MAX_TICKS:
            raise GenerationError("Invalid automation tick")
        _real(self.value, -1, 2, "value")
        if type(self.law) is not str or self.law not in LAWS:
            raise GenerationError("Unknown automation interpolation")


@dataclass(frozen=True, slots=True)
class MixLane:
    track: str
    parameter: str
    keys: tuple[MixKey, ...]

    def __post_init__(self):
        if type(self.track) is not str or self.track not in TRACKS:
            raise GenerationError("Unknown automation track")
        if type(self.parameter) is not str or self.parameter not in PARAMETERS:
            raise GenerationError("Unknown automation parameter")
        if (type(self.keys) is not tuple or not 1 <= len(self.keys) <= MAX_KEYS
                or any(not isinstance(key, MixKey) for key in self.keys)):
            raise GenerationError("Expected bounded automation key tuple")
        ticks = tuple(key.tick for key in self.keys)
        if ticks != tuple(sorted(set(ticks))):
            raise GenerationError("Automation ticks must strictly increase")
        low, high = (0, 2) if self.parameter == "gain" else (-1, 1)
        for key in self.keys:
            _real(key.value, low, high, self.parameter)

    @property
    def identity(self):
        return self.track, self.parameter


def _lane_order(lane):
    return TRACKS.index(lane.track), PARAMETERS.index(lane.parameter)


@dataclass(frozen=True, slots=True)
class AutomationPlan:
    lanes: tuple[MixLane, ...]
    mix: MixPlan = MixPlan()

    def __post_init__(self):
        if not isinstance(self.mix, MixPlan):
            raise GenerationError("Expected static MixPlan")
        if (type(self.lanes) is not tuple or len(self.lanes) > len(TRACKS) * len(PARAMETERS)
                or any(not isinstance(lane, MixLane) for lane in self.lanes)):
            raise GenerationError("Invalid bounded automation lane tuple")
        identities = tuple(lane.identity for lane in self.lanes)
        if len(set(identities)) != len(identities):
            raise GenerationError("Duplicate automation lane")
        if tuple(sorted(self.lanes, key=_lane_order)) != self.lanes:
            raise GenerationError("Automation lanes must have canonical order")

    def validate_score(self, score):
        if not isinstance(score, Arrangement):
            raise GenerationError("Expected Arrangement for automation")
        if any(lane.keys[-1].tick > score.plan.ticks for lane in self.lanes):
            raise GenerationError("Automation extends beyond score clock")

    @property
    def sha256(self):
        return hashlib.sha256(encode_automation(self)).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GenerationError("Duplicate automation JSON field")
        result[key] = value
    return result


def _constant(value):
    raise GenerationError("Nonfinite automation JSON number")


def _object(value, required, optional=()):
    if (type(value) is not dict or not set(required) <= set(value)
            or not set(value) <= set(required) | set(optional)):
        raise GenerationError("Unexpected automation JSON fields")
    return value


def encode_automation(plan):
    if not isinstance(plan, AutomationPlan):
        raise GenerationError("Expected AutomationPlan")
    # Encode normalized numeric fields so int and float spellings share identity.
    lanes = [
        {"track": lane.track, "parameter": lane.parameter,
         "keys": [{"tick": key.tick, "value": float(key.value), "law": key.law}
                  for key in lane.keys]}
        for lane in plan.lanes
    ]
    mix = asdict(plan.mix)
    mix["gains"] = [float(value) for value in plan.mix.gains]
    mix["room"] = float(plan.mix.room)
    mix["peak_ceiling"] = float(plan.mix.peak_ceiling)
    mix["mute"] = [name for name in TRACKS if name in plan.mix.mute]
    mix["solo"] = [name for name in TRACKS if name in plan.mix.solo]
    return json.dumps({"format": PROTOCOL, "lanes": lanes, "mix": mix},
                      sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def decode_automation(data):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_PLAN_BYTES:
        raise GenerationError("Invalid automation document byte count")
    try:
        obj = _object(json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                                 parse_constant=_constant),
                      ("format", "lanes"), ("mix",))
        if obj["format"] != PROTOCOL:
            raise GenerationError("Unknown automation document format")
        rows = obj["lanes"]
        if type(rows) is not list or len(rows) > len(TRACKS) * len(PARAMETERS):
            raise GenerationError("Invalid automation lane array")
        lanes = []
        for row in rows:
            row = _object(row, ("track", "parameter", "keys"))
            keys = row["keys"]
            if type(keys) is not list or not 1 <= len(keys) <= MAX_KEYS:
                raise GenerationError("Invalid automation key array")
            lanes.append(MixLane(row["track"], row["parameter"], tuple(
                MixKey(**_object(key, ("tick", "value"), ("law",))) for key in keys
            )))
        raw_mix = dict(_object(obj.get("mix", {}), (),
                               ("gains", "mute", "solo", "room", "peak_ceiling")))
        for name in ("gains", "mute", "solo"):
            if name in raw_mix:
                if type(raw_mix[name]) is not list or len(raw_mix[name]) > len(TRACKS):
                    raise GenerationError("Invalid automation mix array")
                raw_mix[name] = tuple(raw_mix[name])
        return AutomationPlan(tuple(sorted(lanes, key=_lane_order)), MixPlan(**raw_mix))
    except (ValueError, UnicodeError, TypeError, OverflowError, RecursionError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed automation document") from exc


def read_automation(path):
    with open(path, "rb") as stream:
        return decode_automation(stream.read(MAX_PLAN_BYTES + 1))


@dataclass(frozen=True, slots=True)
class FrameLane:
    """Compiled sample-clock lane. Values outside the keys hold the endpoints."""

    positions: tuple[int, ...]
    values: tuple[float, ...]
    laws: tuple[str, ...]

    def __post_init__(self):
        if (type(self.positions) is not tuple or type(self.values) is not tuple
                or type(self.laws) is not tuple
                or not 1 <= len(self.positions) <= MAX_KEYS
                or len(self.values) != len(self.positions) or len(self.laws) != len(self.positions)
                or any(type(p) is not int or not 0 <= p <= MAX_FRAMES for p in self.positions)
                or self.positions != tuple(sorted(set(self.positions)))):
            raise GenerationError("Invalid compiled automation lane")
        for value in self.values:
            _real(value, -1, 2, "compiled value")
        if any(type(law) is not str or law not in LAWS for law in self.laws):
            raise GenerationError("Invalid compiled interpolation")

    def render(self, start, count):
        if (type(start) is not int or type(count) is not int
                or not 0 <= start < MAX_FRAMES or not 1 <= count <= BLOCK_FRAMES
                or start + count > MAX_FRAMES):
            raise GenerationError("Invalid bounded automation render interval")
        frames = np.arange(start, start + count, dtype=np.int64)
        positions = np.asarray(self.positions, dtype=np.int64)
        values = np.asarray(self.values, dtype=np.float64)
        if len(positions) == 1:
            return np.full(count, values[0], dtype=np.float64)
        # Search once per output sample, not once per key per sample.
        index = np.searchsorted(positions, frames, side="right") - 1
        left = np.clip(index, 0, len(positions) - 2)
        phase = np.clip((frames - positions[left]) /
                        (positions[left + 1] - positions[left]), 0., 1.)
        for i in np.unique(left):
            mask = left == i
            law = self.laws[int(i)]
            if law == "step":
                phase[mask] = 0.
            elif law == "smoothstep":
                t = phase[mask]
                phase[mask] = t * t * (3. - 2. * t)
        output = values[left] + (values[left + 1] - values[left]) * phase
        output[frames < positions[0]] = values[0]
        output[frames >= positions[-1]] = values[-1]
        return output


def compile_lanes(score, plan):
    if not isinstance(plan, AutomationPlan):
        raise GenerationError("Expected AutomationPlan")
    plan.validate_score(score)
    return {
        lane.identity: FrameLane(tuple(score.plan.frame(key.tick) for key in lane.keys),
                                 tuple(float(key.value) for key in lane.keys),
                                 tuple(key.law for key in lane.keys))
        for lane in plan.lanes
    }


def _accumulate_track(dry, stem, level, gain_lane, balance_lane):
    """Apply bounded gain and attenuation-only stereo balance, in place on dry."""
    frames = dry.shape[1]
    energy = 0.
    peak = 0.
    for start in range(0, frames, BLOCK_FRAMES):
        stop = min(start + BLOCK_FRAMES, frames)
        count = stop - start
        block = stem[:, start:stop].astype(np.float64)
        block *= level
        if gain_lane is not None:
            block *= gain_lane.render(start, count)[None, :]
        if balance_lane is not None:
            balance = balance_lane.render(start, count)
            # Unlike panning a mono signal, do not synthesize crossfeed or
            # boost either existing stereo channel.
            block[0] *= 1. - np.maximum(balance, 0.)
            block[1] *= 1. + np.minimum(balance, 0.)
        dry[:, start:stop] += block
        energy += float(np.sum(block * block, dtype=np.float64))
        peak = max(peak, float(np.max(np.abs(block))))
    return {"automated_peak": peak, "automated_rms": math.sqrt(energy / (2 * frames))}


def mix_automated_stems(score, stems, plan):
    """Render a score-pinned automated mix without mutating any dry stem."""
    lanes = compile_lanes(score, plan)
    frames = score.plan.frame(score.plan.ticks)
    if (not 1 <= frames <= MAX_FRAMES or len(TRACKS) * 2 * frames * 4 > MAX_STEM_BYTES
            or type(stems) is not dict or set(stems) != set(TRACKS)):
        raise GenerationError("Invalid automated stem collection")
    for stem in stems.values():
        if (not isinstance(stem, np.ndarray) or stem.dtype != np.float32
                or stem.shape != (2, frames) or not np.isfinite(stem).all()):
            raise GenerationError("Invalid automated stereo stem")
    dry = np.zeros((2, frames), dtype=np.float64)
    tracks = {}
    for track, level in zip(TRACKS, plan.mix.gains, strict=True):
        active = plan.mix.active(track)
        gain_lane = lanes.get((track, "gain"))
        balance_lane = lanes.get((track, "balance"))
        stats = (_accumulate_track(dry, stems[track], level, gain_lane, balance_lane)
                 if active else {"automated_peak": 0., "automated_rms": 0.})
        tracks[track] = {**stats, "active": active, "static_gain": float(level),
                         "gain_automated": gain_lane is not None,
                         "balance_automated": balance_lane is not None}
    wet = room_response(dry, plan.mix.room)
    wet *= smooth_edges(frames, 240, 1200)
    peak = float(np.max(np.abs(wet)))
    if not math.isfinite(peak):
        raise GenerationError("Nonfinite automated music mix")
    scale = min(1., plan.mix.peak_ceiling / max(peak, 1e-15))
    audio = (wet * scale).astype(np.float32)
    if not np.isfinite(audio).all():
        raise GenerationError("Nonfinite automated output")
    return audio, {
        "protocol": PROTOCOL, "automation_sha256": plan.sha256,
        "score_sha256": arrangement_digest(score), "tracks": tracks,
        "automation_lanes": len(lanes), "pre_master_peak": peak,
        "master_attenuation": scale, "metrics": metrics_channels(audio),
        "sample_rate": SAMPLE_RATE, "frames": frames, "channels": 2,
        "true_peak_measured": False, "lufs_measured": False,
        "learned_model": False, "external_recordings_used": False,
    }


def generate_automated_music(score, plan):
    if not isinstance(plan, AutomationPlan):
        raise GenerationError("Expected AutomationPlan")
    plan.validate_score(score)  # Fail before expensive synthesis.
    samples, report = mix_automated_stems(score, render_stems(score), plan)
    wav = encode_wav_channels(samples, SAMPLE_RATE)
    report["wav_sha256"] = hashlib.sha256(wav).hexdigest()
    return wav, report
