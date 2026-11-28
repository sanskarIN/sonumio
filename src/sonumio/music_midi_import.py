"""Constrained MIDI-to-arrangement import with explicit semantic losses.

MIDI cannot encode Sonumio's full arrangement metadata or exact panning
and velocity. An existing score supplies clock, harmony and sections.
Only explicitly mapped physical key-on/key-off notes are imported.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import bisect
import json

from .arrangement import Arrangement, Event, TRACKS, event_key, arrangement_digest
from .music_midi import midi_tick, PPQ, CHANNELS, PROGRAMS
from .music_midi_read import MidiDocument, _tempo_map
from .errors import GenerationError

PROTOCOL = "sonumio-midi-import-v1"
MAX_MAPPING_BYTES = 16384
MAX_MAPPINGS = 128
MAX_TICK_ERROR = 16
DEFAULT_PAN = {"keys":-.3,"pad":0.,"bass":0.,"lead":.18,"pluck":.35,
               "kick":0.,"snare":0.,"hat":.35}


@dataclass(frozen=True,slots=True)
class Mapping:
    track: int
    channel: int
    target: str
    pan: float = 0.
    expected_program: int | None = None

    def __post_init__(self):
        if type(self.track) is not int or not 0<=self.track<32:
            raise GenerationError("Invalid MIDI mapping track")
        if type(self.channel) is not int or not 0<=self.channel<16:
            raise GenerationError("Invalid MIDI mapping channel")
        if type(self.target) is not str or self.target not in TRACKS:
            raise GenerationError("Unknown MIDI destination instrument")
        if type(self.pan) not in (int,float) or not -1<=self.pan<=1:
            raise GenerationError("Invalid MIDI import pan")
        if self.expected_program is not None and (type(self.expected_program) is not int
                or not 0<=self.expected_program<=127):
            raise GenerationError("Invalid MIDI expected program")
        if (self.target in ("kick","snare","hat")) != (self.channel == 9):
            raise GenerationError("MIDI percussion channel must map to a percussion track")

    @property
    def source(self):
        return self.track,self.channel


@dataclass(frozen=True,slots=True)
class ImportPlan:
    mappings: tuple[Mapping,...]
    max_tick_error: int = 0

    def __post_init__(self):
        if (type(self.mappings) is not tuple or not 1<=len(self.mappings)<=MAX_MAPPINGS
                or any(not isinstance(m,Mapping) for m in self.mappings)):
            raise GenerationError("Expected bounded MIDI mapping tuple")
        keys=tuple(m.source for m in self.mappings)
        if keys != tuple(sorted(set(keys))):
            raise GenerationError("MIDI mapping sources must be unique and canonical")
        if type(self.max_tick_error) is not int or not 0<=self.max_tick_error<=MAX_TICK_ERROR:
            raise GenerationError("Invalid MIDI grid error tolerance")


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise GenerationError("Duplicate MIDI mapping JSON field")
        result[key]=value
    return result


def _constant(value):
    raise GenerationError("Nonfinite MIDI mapping number")


def decode_mapping(data):
    if type(data) is not bytes or not 1<=len(data)<=MAX_MAPPING_BYTES:
        raise GenerationError("Invalid MIDI mapping byte count")
    try:
        obj=json.loads(data.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_constant)
        if type(obj) is not dict or set(obj)!={"format","mappings","max_tick_error"} or obj["format"]!=PROTOCOL:
            raise GenerationError("Unknown MIDI mapping schema")
        rows=obj["mappings"]
        if type(rows) is not list or not 1<=len(rows)<=MAX_MAPPINGS:
            raise GenerationError("Invalid MIDI mapping row count")
        mapped=[]
        for row in rows:
            if (type(row) is not dict or not {"track","channel","target"}<=set(row)
                    or not set(row)<={"track","channel","target","pan","expected_program"}):
                raise GenerationError("Invalid MIDI mapping row fields")
            mapped.append(Mapping(**row))
        return ImportPlan(tuple(mapped),obj["max_tick_error"])
    except (UnicodeError,RecursionError,TypeError,OverflowError,ValueError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed MIDI mapping document") from exc


def infer_sonumio_mapping(document):
    """Infer only the track naming/channel/program convention of our own exporter."""
    if not isinstance(document,MidiDocument) or document.format!=1:
        raise GenerationError("Automatic mapping requires Sonumio-style MIDI type 1")
    mappings=[]
    names=set()
    for track in document.tracks:
        if not track.notes:
            continue
        name=track.name
        if name not in TRACKS or name in names:
            raise GenerationError("MIDI track naming is not uniquely Sonumio-compatible")
        names.add(name)
        channel=CHANNELS.get(name,9)
        program=PROGRAMS.get(name,0)
        if any(n.channel!=channel or n.program!=program for n in track.notes):
            raise GenerationError("MIDI channel or program differs from Sonumio convention")
        mappings.append(Mapping(track.index,channel,name,DEFAULT_PAN[name],program))
    return ImportPlan(tuple(mappings))


def _strict_clock(document,template):
    plan=template.plan
    target=(60_000_000+plan.bpm//2)//plan.bpm
    if _tempo_map(document)!=((0,target),):
        raise GenerationError("MIDI tempo must exactly match the template constant tempo")
    # Accept omitted time signature as the MIDI default 4/4; reject
    # changes and unsupported meters rather than inferring score sections.
    for track in document.tracks:
        for event in track.meta:
            if event.kind==0x58 and (event.tick!=0 or event.payload[:2]!=bytes((4,2))):
                raise GenerationError("MIDI meter must remain 4/4")
            if event.kind in (0x20,0x21):
                raise GenerationError("MIDI channel/port prefix routing is unsupported for import")
        if track.sysex_count:
            raise GenerationError("MIDI SysEx is not supported for arrangement import")
        for event in track.channels:
            if event.kind not in (0x80,0x90,0xc0):
                raise GenerationError("MIDI performance controllers must be removed explicitly before import")
            if event.kind==0xc0 and event.tick!=0:
                raise GenerationError("MIDI program changes after tick zero are unsupported")
    # Compare using exact integer rational timing, independent of PPQ.
    if document.end*PPQ!=midi_tick(plan,plan.ticks)*document.ppq:
        raise GenerationError("MIDI duration differs from template score clock")


class _Grid:
    def __init__(self,template,ppq,tolerance):
        self.positions=tuple(midi_tick(template.plan,tick)*ppq for tick in range(template.plan.ticks+1))
        self.tolerance=tolerance*PPQ

    def resolve(self,source):
        x=source*PPQ
        index=bisect.bisect_left(self.positions,x)
        candidates=[]
        for i in (index-1,index):
            if 0<=i<len(self.positions):
                candidates.append((abs(self.positions[i]-x),i))
        candidates.sort()
        if not candidates or candidates[0][0]>self.tolerance:
            raise GenerationError("MIDI note is off the template tick grid")
        if len(candidates)>1 and candidates[0][0]==candidates[1][0]:
            raise GenerationError("MIDI note grid quantization is ambiguous")
        error,tick=candidates[0]
        return tick,error


def import_arrangement(document,template,plan=None):
    if not isinstance(document,MidiDocument) or not isinstance(template,Arrangement):
        raise GenerationError("Expected decoded MIDI and validated template arrangement")
    plan=infer_sonumio_mapping(document) if plan is None else plan
    if not isinstance(plan,ImportPlan):
        raise GenerationError("Expected MIDI ImportPlan")
    _strict_clock(document,template)
    mapping={row.source:row for row in plan.mappings}
    observed={(note.track,note.channel) for note in document.notes}
    if set(mapping)!=observed:
        raise GenerationError("MIDI mapping must cover exactly the note-bearing track/channel pairs")
    grid=_Grid(template,document.ppq,plan.max_tick_error)
    events=[]
    track_counts=dict.fromkeys(TRACKS,0)
    max_error=0
    quantized=0
    for note in document.notes:
        row=mapping[note.track,note.channel]
        if row.expected_program is not None and note.program!=row.expected_program:
            raise GenerationError("MIDI note program differs from explicit mapping")
        start,start_error=grid.resolve(note.start)
        stop,stop_error=grid.resolve(note.stop)
        if stop<=start:
            raise GenerationError("MIDI note collapses after tick grid quantization")
        max_error=max(max_error,start_error,stop_error)
        quantized+=int(start_error!=0)+int(stop_error!=0)
        # This conversion deliberately does not claim to recover the
        # original six-decimal velocity or per-event pan from MIDI.
        event=Event(row.target,note.pitch,start,stop-start,note.velocity/127.,float(row.pan))
        events.append(event)
        track_counts[row.target]+=1
    result=replace(template,events=tuple(sorted(events,key=event_key)))
    return result,{
        "protocol":PROTOCOL,"source_midi_sha256":document.source_sha256,
        "template_score_sha256":arrangement_digest(template),
        "result_score_sha256":arrangement_digest(result),
        "events":len(events),"events_by_track":track_counts,
        "quantized_endpoints":quantized,"max_source_tick_error":max_error/PPQ,
        "max_tick_error_allowed":plan.max_tick_error,
        "clock_source":"template","harmony_source":"template","sections_source":"template",
        "velocity_source":"7-bit MIDI velocity / 127","pan_source":"explicit track mapping",
        "programs_used_for_validation_only":True,"performance_controls_applied":False,
        "exact_original_score_reconstructed":False,"audio_rendered":False,
    }
