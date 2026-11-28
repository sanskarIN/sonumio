"""Native project assembly with conservative score-clock automation transfer.

The original assembly engine remaps symbolic events. This layer transfers
gain and balance curves across exactly the same bar spans. The source
projects must share a static mixer, and every nonconstant interpolation
interval must preserve its sample duration. Unsupported partial smoothstep
curves and incompatible seam values are rejected, not approximated.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from bisect import bisect_right

from .arrangement import TRACKS, TICKS_PER_BAR
from .errors import GenerationError
from .music_assembly import AssemblyPlan, assemble, encode_assembly, decode_assembly, _alias, _digest
from .music_automation import AutomationPlan, MixLane, MixKey, PARAMETERS, MAX_KEYS, encode_automation
from .music_project import MusicProject, create_project
from .music_score_io import _object, _unique, _nonfinite

FORMAT = "sonumio-native-project-assembly-v1"
MAX_BYTES = 96 * 1024
DEFAULTS = {"gain": 1., "balance": 0.}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


@dataclass(frozen=True, slots=True)
class ProjectPin:
    alias: str
    project_sha256: str

    def __post_init__(self):
        _alias(self.alias)
        _digest(self.project_sha256)


@dataclass(frozen=True, slots=True)
class ProjectAssemblyPlan:
    title: str
    assembly: AssemblyPlan
    projects: tuple[ProjectPin, ...]

    def __post_init__(self):
        from .music_project import _title
        _title(self.title)
        if not isinstance(self.assembly, AssemblyPlan):
            raise GenerationError("Expected validated score assembly recipe")
        if (type(self.projects) is not tuple
                or any(not isinstance(pin, ProjectPin) for pin in self.projects)
                or tuple(pin.alias for pin in self.projects) !=
                   tuple(pin.alias for pin in self.assembly.sources)):
            raise GenerationError("Project pins must match score assembly aliases")

    @property
    def sha256(self):
        return hashlib.sha256(FORMAT.encode("ascii")+b"\0"+_canonical(_payload(self))).hexdigest()


def _payload(plan):
    return {"title": plan.title,
            "assembly": json.loads(encode_assembly(plan.assembly)),
            "projects": [asdict(pin) for pin in plan.projects]}


def encode_project_assembly(plan):
    if not isinstance(plan, ProjectAssemblyPlan):
        raise GenerationError("Expected ProjectAssemblyPlan")
    raw = _canonical({"format": FORMAT, "sha256": plan.sha256, **_payload(plan)})
    if len(raw) > MAX_BYTES:
        raise GenerationError("Project assembly recipe exceeds byte budget")
    return raw


def decode_project_assembly(data, *, expected_sha256=None):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_BYTES:
        raise GenerationError("Invalid project assembly byte count")
    if expected_sha256 is not None:
        _digest(expected_sha256)
    try:
        obj = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
        _object(obj, ("format", "sha256", "title", "assembly", "projects"))
        if obj["format"] != FORMAT or type(obj["projects"]) is not list or not 1 <= len(obj["projects"]) <= 8:
            raise GenerationError("Invalid project assembly format or pins")
        assembly = decode_assembly(_canonical(obj["assembly"]))
        pins = tuple(ProjectPin(**_object(pin, ("alias", "project_sha256"))) for pin in obj["projects"])
        plan = ProjectAssemblyPlan(obj["title"], assembly, pins)
        if obj["sha256"] != plan.sha256 or (expected_sha256 is not None and expected_sha256 != plan.sha256):
            raise GenerationError("Project assembly SHA256 mismatch")
        return plan
    except (UnicodeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed project assembly document") from exc


def read_project_assembly(path, *, expected_sha256=None):
    with open(path, "rb") as stream:
        return decode_project_assembly(stream.read(MAX_BYTES+1), expected_sha256=expected_sha256)


def _lane_map(project):
    return {lane.identity: lane for lane in project.automation.lanes}


def _mixer_identity(project):
    return json.loads(encode_automation(AutomationPlan((), project.automation.mix)))["mix"]


def verify_projects(plan, projects):
    if not isinstance(plan, ProjectAssemblyPlan) or type(projects) is not dict:
        raise GenerationError("Expected project assembly plan and local project bindings")
    if set(projects) != {pin.alias for pin in plan.projects}:
        raise GenerationError("Project assembly bindings differ from recipe")
    identity = None
    for pin in plan.projects:
        project = projects[pin.alias]
        if not isinstance(project, MusicProject) or project.sha256 != pin.project_sha256:
            raise GenerationError("Project assembly source digest mismatch: "+pin.alias)
        mixer = _mixer_identity(project)
        if identity is None:
            identity = mixer
        elif identity != mixer:
            raise GenerationError("Project assembly requires identical static mixer settings")
    return {pin.alias: pin.project_sha256 for pin in plan.projects}


def _value_and_law(score, lane, tick, default):
    """Value at tick and law of the interval starting at tick."""
    if lane is None:
        return default, "step"
    keys = lane.keys
    index = bisect_right(tuple(key.tick for key in keys), tick)-1
    if index < 0:
        return float(keys[0].value), "step"
    left = keys[index]
    if index == len(keys)-1:
        return float(left.value), "step"
    right = keys[index+1]
    if tick == left.tick:
        return float(left.value), left.law
    if left.law == "step" or left.value == right.value:
        return float(left.value), "step"
    if left.law == "smoothstep":
        raise GenerationError("Cannot cut a nonconstant smoothstep inside its interval")
    start = score.plan.frame(left.tick)
    stop = score.plan.frame(right.tick)
    phase = (score.plan.frame(tick)-start)/(stop-start)
    return float(left.value+(right.value-left.value)*phase), "linear"


def crop_lane(score, lane, start, stop, default):
    """Half-open crop with an explicit endpoint value and outgoing laws."""
    if not 0 <= start < stop <= score.plan.ticks:
        raise GenerationError("Invalid automation crop interval")
    points = [start]
    if lane is not None:
        points.extend(key.tick for key in lane.keys if start < key.tick < stop)
    points.append(stop)
    result = []
    for tick in points:
        value, law = _value_and_law(score, lane, tick, default)
        result.append(MixKey(tick-start, value, law))
    # Normalizing equal-value intervals to step makes joins unambiguous;
    # it does not change the waveform. The terminal key has no interval.
    for index in range(len(result)-1):
        if result[index].value == result[index+1].value:
            result[index] = MixKey(result[index].tick, result[index].value, "step")
    last = result[-1]
    result[-1] = MixKey(last.tick, last.value, "step")
    return tuple(result)


def _translated(source, target, keys, start, offset):
    # A bar translation at a nonintegral frames/beat can alter rounded
    # sample spans. Reject changed durations for nonconstant ramps.
    for left, right in zip(keys, keys[1:]):
        if left.law == "step" or left.value == right.value:
            continue
        source_frames = source.plan.frame(start+right.tick)-source.plan.frame(start+left.tick)
        target_frames = target.plan.frame(offset+right.tick)-target.plan.frame(offset+left.tick)
        if source_frames != target_frames:
            raise GenerationError("Automation ramp sample duration changes after assembly")
    return [MixKey(offset+key.tick, key.value, key.law) for key in keys]


def _append_span(output, incoming):
    if not output:
        output.extend(incoming)
        return
    if len(output) < 2 or output[-1].tick != incoming[0].tick:
        raise GenerationError("Automation assembly spans are not contiguous")
    before, end = output[-2:]
    first = incoming[0]
    if before.law != "step" and end.value != first.value:
        raise GenerationError("Automation seam has incompatible continuous endpoint values")
    # The incoming span owns its boundary tick. A preceding step can
    # jump to that value without changing any sample of the previous span.
    output[-1] = first
    output.extend(incoming[1:])


def _compact(keys):
    result = []
    for key in keys:
        if (result and result[-1].law == "step" and key.law == "step"
                and result[-1].value == key.value):
            continue
        result.append(key)
    return tuple(result)


def assemble_projects(plan, projects):
    verified = verify_projects(plan, projects)
    score, score_report = assemble(plan.assembly, {name: project.score for name, project in projects.items()})
    maps = {name: _lane_map(project) for name, project in projects.items()}
    lanes = []
    traces = []
    for track in TRACKS:
        for parameter in PARAMETERS:
            identity = track, parameter
            if not any(identity in mapping for mapping in maps.values()):
                continue
            output = []
            offset = 0
            for segment in plan.assembly.segments:
                source = projects[segment.source].score
                start = segment.first_bar*TICKS_PER_BAR
                stop = segment.last_bar*TICKS_PER_BAR
                try:
                    cropped = crop_lane(source, maps[segment.source].get(identity), start, stop,
                                        DEFAULTS[parameter])
                    for _ in range(segment.repeat):
                        translated = _translated(source, score, cropped, start, offset)
                        _append_span(output, translated)
                        offset += stop-start
                except GenerationError as exc:
                    raise GenerationError(f"Automation assembly {track}/{parameter} at tick {offset}: {exc}") from exc
            if offset != score.plan.ticks:
                raise GenerationError("Automation assembly did not cover output clock")
            compact = _compact(output)
            if len(compact) > MAX_KEYS:
                raise GenerationError(f"Automation assembly {track}/{parameter} exceeds key budget")
            # A constant default lane is semantically identical to absence.
            if all(key.value == DEFAULTS[parameter] for key in compact):
                continue
            lanes.append(MixLane(track, parameter, compact))
            traces.append({"track": track, "parameter": parameter, "keys": len(compact)})
    first = projects[plan.projects[0].alias]
    automation = AutomationPlan(tuple(lanes), first.automation.mix)
    automation.validate_score(score)
    project = create_project(plan.title, score, automation)
    if verify_projects(plan, projects) != verified:
        raise GenerationError("Source project changed during assembly")
    return project, {
        "format": FORMAT, "recipe_sha256": plan.sha256,
        "source_projects": verified, "score_assembly": score_report,
        "result_project_sha256": project.sha256,
        "result_automation_sha256": automation.sha256, "automation_lanes": traces,
        "source_projects_unchanged": True,
        "new_root_revision": True, "static_mixer_preserved": True,
        "nonconstant_ramp_sample_durations_verified": True,
        "floating_point_bitwise_curve_parity_claimed": False,
        "audio_rendered": False, "authenticated": False,
    }
