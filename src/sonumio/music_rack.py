"""Score-pinned track inserts, parallel auxiliary sends, and master inserts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .arrangement import Arrangement, TRACKS, arrangement_digest
from .audio import SAMPLE_RATE, encode_wav_channels, metrics_channels
from .errors import GenerationError
from .music_automation import (
    AutomationPlan, BLOCK_FRAMES, compile_lanes, decode_automation,
    encode_automation, _unique, _constant, _object,
)
from .music_instruments import smooth_edges
from .music_mix import render_stems, room_response, _validate_stem
from .music_rack_effects import (
    Insert, validate_chain, apply_chain, stage_stats, _checked,
    MAX_EFFECT_FRAMES, _real,
)

PROTOCOL = "sonumio-music-processing-rack-v1"
MAX_RACK_BYTES = 256 * 1024
MAX_AUX = 2
MAX_COST = 80_000_000
AUX_NAMES = ("aux1", "aux2")


@dataclass(frozen=True, slots=True)
class TrackRack:
    track: str
    inserts: tuple[Insert, ...]

    def __post_init__(self):
        if type(self.track) is not str or self.track not in TRACKS:
            raise GenerationError("Unknown rack track")
        validate_chain(self.inserts)


@dataclass(frozen=True, slots=True)
class Send:
    track: str
    level: float

    def __post_init__(self):
        if type(self.track) is not str or self.track not in TRACKS:
            raise GenerationError("Unknown auxiliary send track")
        object.__setattr__(self, "level", _real(self.level, 0., 1., "send"))


@dataclass(frozen=True, slots=True)
class AuxBus:
    name: str
    sends: tuple[Send, ...]
    inserts: tuple[Insert, ...]
    return_gain: float = 1.

    def __post_init__(self):
        if type(self.name) is not str or self.name not in AUX_NAMES:
            raise GenerationError("Unknown auxiliary bus")
        if (type(self.sends) is not tuple or not 1 <= len(self.sends) <= len(TRACKS)
                or any(not isinstance(send, Send) for send in self.sends)):
            raise GenerationError("Invalid auxiliary sends")
        tracks = tuple(send.track for send in self.sends)
        if tracks != tuple(track for track in TRACKS if track in tracks):
            raise GenerationError("Auxiliary sends must be unique and canonical")
        validate_chain(self.inserts)
        if not self.inserts:
            raise GenerationError("Auxiliary bus requires a processing chain")
        object.__setattr__(self, "return_gain",
                           _real(self.return_gain, 0., 2., "return gain"))


@dataclass(frozen=True, slots=True)
class RackPlan:
    automation: AutomationPlan = AutomationPlan(())
    tracks: tuple[TrackRack, ...] = ()
    aux: tuple[AuxBus, ...] = ()
    master: tuple[Insert, ...] = ()

    def __post_init__(self):
        if not isinstance(self.automation, AutomationPlan):
            raise GenerationError("Expected rack automation plan")
        if (type(self.tracks) is not tuple or len(self.tracks) > len(TRACKS)
                or any(not isinstance(row, TrackRack) for row in self.tracks)):
            raise GenerationError("Invalid rack tracks")
        names = tuple(row.track for row in self.tracks)
        if names != tuple(track for track in TRACKS if track in names):
            raise GenerationError("Rack tracks must be unique and canonical")
        if (type(self.aux) is not tuple or len(self.aux) > MAX_AUX
                or any(not isinstance(row, AuxBus) for row in self.aux)):
            raise GenerationError("Invalid auxiliary buses")
        names = tuple(row.name for row in self.aux)
        if names != tuple(name for name in AUX_NAMES if name in names):
            raise GenerationError("Auxiliary buses must be unique and canonical")
        validate_chain(self.master)

    @property
    def sha256(self):
        return hashlib.sha256(encode_rack(self)).hexdigest()

    def validate_score(self, score):
        self.automation.validate_score(score)
        frames = score.plan.frame(score.plan.ticks)
        if not 1 <= frames <= MAX_EFFECT_FRAMES:
            raise GenerationError("Music rack exceeds 60-second frame budget")
        chains = [row.inserts for row in self.tracks]
        chains.extend(row.inserts for row in self.aux)
        chains.append(self.master)
        cost = frames * sum(item.cost for chain in chains for item in chain)
        if cost > MAX_COST:
            raise GenerationError("Music rack exceeds processing budget")
        return frames, cost


def encode_rack(plan):
    if not isinstance(plan, RackPlan):
        raise GenerationError("Expected RackPlan")
    obj = {
        "format": PROTOCOL,
        "automation": json.loads(encode_automation(plan.automation)),
        "tracks": [{"track": row.track, "inserts": [i.as_dict() for i in row.inserts]}
                   for row in plan.tracks],
        "aux": [{"name": row.name,
                 "sends": [{"track": s.track, "level": s.level} for s in row.sends],
                 "inserts": [i.as_dict() for i in row.inserts],
                 "return_gain": row.return_gain} for row in plan.aux],
        "master": [i.as_dict() for i in plan.master],
    }
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")
    if len(data) > MAX_RACK_BYTES:
        raise GenerationError("Music rack document exceeds byte budget")
    return data


def _array(value, limit, label):
    if type(value) is not list or len(value) > limit:
        raise GenerationError("Invalid rack " + label)
    return value


def _chain(raw):
    from .music_rack_effects import MAX_CHAIN
    result = []
    for value in _array(raw, MAX_CHAIN, "inserts"):
        row = _object(value, ("kind",), ("parameters", "wet", "bypass"))
        params = row.get("parameters", {})
        if type(params) is not dict:
            raise GenerationError("Expected rack parameter object")
        # No ** expansion: parameters named wet/bypass cannot override controls.
        result.append(Insert(row["kind"], tuple(params.items()),
                             row.get("wet", 1.), row.get("bypass", False)))
    return tuple(result)


def decode_rack(data):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_RACK_BYTES:
        raise GenerationError("Invalid music rack byte count")
    try:
        obj = _object(json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                                 parse_constant=_constant),
                      ("format", "automation"), ("tracks", "aux", "master"))
        if obj["format"] != PROTOCOL:
            raise GenerationError("Unknown rack document format")
        automation = decode_automation(json.dumps(obj["automation"], allow_nan=False,
                                                  separators=(",", ":")).encode("utf-8"))
        tracks = []
        for raw in _array(obj.get("tracks", []), len(TRACKS), "tracks"):
            row = _object(raw, ("track", "inserts"))
            tracks.append(TrackRack(row["track"], _chain(row["inserts"])))
        aux = []
        for raw in _array(obj.get("aux", []), MAX_AUX, "buses"):
            row = _object(raw, ("name", "sends", "inserts"), ("return_gain",))
            sends = [Send(**_object(s, ("track", "level")))
                     for s in _array(row["sends"], len(TRACKS), "sends")]
            aux.append(AuxBus(row["name"], tuple(sorted(sends, key=lambda s: TRACKS.index(s.track))),
                              _chain(row["inserts"]), row.get("return_gain", 1.)))
        return RackPlan(automation, tuple(sorted(tracks, key=lambda x: TRACKS.index(x.track))),
                        tuple(sorted(aux, key=lambda x: AUX_NAMES.index(x.name))),
                        _chain(obj.get("master", [])))
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed music rack document") from exc


def read_rack(path):
    with open(path, "rb") as stream:
        return decode_rack(stream.read(MAX_RACK_BYTES + 1))


def _fader(stem, level, gain_lane, balance_lane):
    out = np.empty(stem.shape, dtype=np.float64)
    frames = stem.shape[1]
    for start in range(0, frames, BLOCK_FRAMES):
        stop = min(start + BLOCK_FRAMES, frames)
        block = stem[:, start:stop].astype(np.float64)
        block *= level
        if gain_lane is not None:
            block *= gain_lane.render(start, stop - start)[None, :]
        if balance_lane is not None:
            balance = balance_lane.render(start, stop - start)
            block[0] *= 1. - np.maximum(balance, 0.)
            block[1] *= 1. + np.minimum(balance, 0.)
        out[:, start:stop] = block
    return _checked(out)


def describe_rack(score, plan):
    if not isinstance(score, Arrangement) or not isinstance(plan, RackPlan):
        raise GenerationError("Expected score and rack")
    frames, cost = plan.validate_score(score)
    return {"protocol": PROTOCOL, "score_sha256": arrangement_digest(score),
            "rack_sha256": plan.sha256, "automation_sha256": plan.automation.sha256,
            "frames": frames, "sample_rate": SAMPLE_RATE, "channels": 2,
            "cost_units": cost, "maximum_cost_units": MAX_COST,
            "track_chains": len(plan.tracks), "aux_buses": len(plan.aux),
            "master_inserts": len(plan.master),
            "rendered": False, "learned_model": False}


def mix_rack_stems(score, stems, plan):
    receipt = describe_rack(score, plan)
    frames = receipt["frames"]
    if type(stems) is not dict or set(stems) != set(TRACKS):
        raise GenerationError("Invalid rack stem collection")
    for stem in stems.values():
        _validate_stem(stem, frames)
        _checked(stem)
    lanes = compile_lanes(score, plan.automation)
    mix = plan.automation.mix
    chains = {row.track: row.inserts for row in plan.tracks}
    aux_inputs = {row.name: np.zeros((2, frames), dtype=np.float64) for row in plan.aux}
    send_levels = {row.name: {s.track: s.level for s in row.sends} for row in plan.aux}
    dry = np.zeros((2, frames), dtype=np.float64)
    track_reports = {}
    for track, level in zip(TRACKS, mix.gains, strict=True):
        source = stems[track]
        active = mix.active(track)
        info = {"active": active, "input": stage_stats(source), "inserts": [],
                "static_gain": float(level), "sends": {}}
        if active:
            processed, stages = apply_chain(source, chains.get(track, ()))
            post = _fader(processed, level, lanes.get((track, "gain")),
                          lanes.get((track, "balance")))
            dry += post
            _checked(dry)
            info["inserts"] = stages
            info["post_fader"] = stage_stats(post)
            for name, bus in aux_inputs.items():
                send = send_levels[name].get(track, 0.)
                if send:
                    bus += post * send
                    _checked(bus)
                    info["sends"][name] = send
        else:
            info["post_fader"] = None
        track_reports[track] = info

    buses = {}
    for row in plan.aux:
        source = aux_inputs.pop(row.name)
        processed, stages = apply_chain(source, row.inserts)
        returned = _checked(processed.astype(np.float64) * row.return_gain)
        dry += returned
        _checked(dry)
        buses[row.name] = {"input": stage_stats(source), "inserts": stages,
                           "return_gain": row.return_gain, "returned": stage_stats(returned)}
    room = _checked(room_response(dry, mix.room))
    master, stages = apply_chain(room, plan.master)
    wet = master.astype(np.float64, copy=True)
    wet *= smooth_edges(frames, 240, 1200)
    peak = float(np.max(np.abs(wet)))
    scale = min(1., mix.peak_ceiling / max(peak, 1e-15))
    samples = (wet * scale).astype(np.float32)
    receipt.update({
        "rendered": True, "tracks": track_reports, "aux": buses,
        "room_output": stage_stats(room), "master": stages,
        "pre_master_peak": peak, "master_attenuation": scale,
        "peak_ceiling": float(mix.peak_ceiling), "metrics": metrics_channels(samples),
        "output_pcm": stage_stats(samples),
        "true_peak_measured": False, "lufs_measured": False,
        "external_recordings_used": False, "effect_tail_extended": False,
    })
    return samples, receipt


def generate_rack_music(score, plan):
    describe_rack(score, plan)  # Reject invalid geometry before note synthesis.
    samples, report = mix_rack_stems(score, render_stems(score), plan)
    wav = encode_wav_channels(samples, SAMPLE_RATE)
    report["wav_sha256"] = hashlib.sha256(wav).hexdigest()
    return wav, report
