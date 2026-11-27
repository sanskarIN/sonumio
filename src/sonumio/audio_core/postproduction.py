"""One bounded, deterministic postproduction pipeline for local PCM.

Duration/pitch editing precedes optional circular loop construction,
which precedes optional mastering. All stages share the original sample
clock and channel layout. Intermediate audio is not written or quantized.
The ordered stage receipt binds each operation to its exact PCM digest;
it is reproducibility metadata, not a cryptographic signature.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib

from .contracts import AudioBuffer, AudioSpec
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .hashing import (
    canonical_json_bytes, checksum_record, verify_record, digest_audio, strict_sha256,
)
from .time_scale import TimeScaleSettings, target_frames_for_ratio, stretch_audio
from .pitch_shift import PitchShiftSettings, pitch_shift
from .loop_analysis import LoopSearchPlan, search_loop
from .loop_render import LoopRenderPlan, render_loop, LAWS
from .mastering import MasteringSettings, master_audio
from .contracts import finite_float

MAX_PIPELINE_PCM_BYTES = 24*1024*1024
PROTOCOL = "sonumio-local-postproduction-v1"


@dataclass(frozen=True, slots=True)
class PostProductionPlan:
    duration_ratio: float | None = None
    duration_frames: int | None = None
    pitch_semitones: float | None = None
    time_scale: TimeScaleSettings = TimeScaleSettings()
    pitch_lobes: int = 24
    loop: LoopRenderPlan | None = None
    loop_search: LoopSearchPlan | None = None
    search_render_law: str = "cosine"
    search_render_repeats: int = 1
    search_render_peak_ceiling: float | None = None
    mastering: MasteringSettings | None = None

    def __post_init__(self):
        operations = sum(v is not None for v in (
            self.duration_ratio, self.duration_frames, self.pitch_semitones,
        ))
        if operations > 1:
            raise AudioContractError("Choose at most one duration or pitch operation")
        if not isinstance(self.time_scale, TimeScaleSettings):
            raise AudioContractError("Expected TimeScaleSettings")
        if type(self.pitch_lobes) is not int or not 4 <= self.pitch_lobes <= 128:
            raise AudioContractError("Invalid pipeline pitch resampling lobes")
        if self.duration_ratio is not None:
            target_frames_for_ratio(2,self.duration_ratio)
        if self.duration_frames is not None and (
            type(self.duration_frames) is not int or not 1 <= self.duration_frames <= 2**31-1
        ):
            raise AudioContractError("Invalid pipeline target frame count")
        if self.pitch_semitones is not None:
            self.pitch_settings()
        if self.loop is not None and not isinstance(self.loop, LoopRenderPlan):
            raise AudioContractError("Expected LoopRenderPlan")
        if self.loop_search is not None and not isinstance(self.loop_search, LoopSearchPlan):
            raise AudioContractError("Expected LoopSearchPlan")
        if self.loop is not None and self.loop_search is not None:
            raise AudioContractError("Choose manual loop or loop search, not both")
        if self.loop_search is not None:
            LoopRenderPlan(0,2,law=self.search_render_law,
                           repeats=self.search_render_repeats,
                           peak_ceiling=self.search_render_peak_ceiling)
        elif (self.search_render_law != "cosine" or self.search_render_repeats != 1
              or self.search_render_peak_ceiling is not None):
            raise AudioContractError("Search render options require a loop search")
        if self.mastering is not None and not isinstance(self.mastering, MasteringSettings):
            raise AudioContractError("Expected MasteringSettings")
        if not operations and self.loop is None and self.loop_search is None and self.mastering is None:
            raise AudioContractError("Postproduction plan has no operations")

    def pitch_settings(self):
        from dataclasses import replace
        return PitchShiftSettings(self.pitch_semitones,
            time_scale=replace(self.time_scale,peak_ceiling=None),
            lobes=self.pitch_lobes,peak_ceiling=self.time_scale.peak_ceiling)

    def as_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class StageReceipt:
    operation: str
    input_frames: int
    output_frames: int
    input_pcm_sha256: str
    output_pcm_sha256: str
    details: dict

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PostProductionResult:
    audio: AudioBuffer
    plan: PostProductionPlan
    source_pcm_sha256: str
    source_frames: int
    stages: tuple[StageReceipt,...]

    @property
    def output_pcm_sha256(self):
        return self.stages[-1].output_pcm_sha256

    def manifest(self):
        body = {
            "protocol": PROTOCOL,
            "plan": self.plan.as_dict(),
            "plan_sha256": self.plan.sha256,
            "sample_rate": self.audio.sample_rate,
            "channels": self.audio.channels,
            "source_frames": self.source_frames,
            "output_frames": self.audio.frames,
            "source_pcm_sha256": self.source_pcm_sha256,
            "stages": [s.as_dict() for s in self.stages],
            "intermediate_quantization": False,
            "model_inference": False,
        }
        return checksum_record(body,payload_sha256=self.output_pcm_sha256)


def _budget(audio, frames):
    spec = AudioSpec(audio.sample_rate,audio.spec.layout,frames)
    if spec.byte_count > MAX_PIPELINE_PCM_BYTES:
        raise AudioLimitError("Postproduction intermediate exceeds PCM budget")


def _preflight(audio,plan):
    _budget(audio,audio.frames)
    frames = audio.frames
    if plan.duration_ratio is not None:
        frames = target_frames_for_ratio(frames,plan.duration_ratio)
    elif plan.duration_frames is not None:
        frames = plan.duration_frames
    if frames*2 < audio.frames or frames > 2*audio.frames:
        raise AudioContractError("Pipeline duration is outside supported ratio")
    _budget(audio,frames)
    if plan.pitch_semitones is not None:
        from .contracts import check_rate
        virtual = check_rate(round(audio.sample_rate*2**(plan.pitch_semitones/12)))
        _budget(audio,target_frames_for_ratio(frames,virtual/audio.sample_rate))
    if plan.loop is not None:
        if plan.loop.stop_frame > frames:
            raise AudioContractError("Loop endpoints exceed transformed frame geometry")
        _budget(audio,plan.loop.period_frames*plan.loop.repeats)
    if plan.loop_search is not None:
        search = plan.loop_search
        if search.stop_frame > frames:
            raise AudioContractError("Loop search exceeds transformed frame geometry")
        maximum = min(frames,search.stop_frame+search.stop_radius)-max(0,search.start_frame-search.start_radius)-search.blend_frames
        if search.maximum_period is not None:
            maximum = min(maximum,search.maximum_period)
        if maximum < search.minimum_period:
            raise AudioContractError("No admissible pipeline loop period")
        _budget(audio,maximum*plan.search_render_repeats)


def run_postproduction(audio,plan,*,expected_source_pcm_sha256=None,
                       expected_output_pcm_sha256=None):
    if not isinstance(audio,AudioBuffer) or not isinstance(plan,PostProductionPlan):
        raise AudioContractError("Expected AudioBuffer and PostProductionPlan")
    if expected_source_pcm_sha256 is not None:
        strict_sha256(expected_source_pcm_sha256)
    if expected_output_pcm_sha256 is not None:
        strict_sha256(expected_output_pcm_sha256)
    _preflight(audio,plan)
    source = digest_audio(audio)
    if expected_source_pcm_sha256 is not None and source != expected_source_pcm_sha256:
        raise AudioIntegrityError("Source PCM digest mismatch")
    work = audio
    previous = source
    stages = []

    def accept(operation,result,details):
        nonlocal work,previous
        if (result.sample_rate != audio.sample_rate
                or result.spec.layout != audio.spec.layout):
            raise AudioIntegrityError("Postproduction stage changed channel layout or clock")
        _budget(audio,result.frames)
        current = digest_audio(result)
        stages.append(StageReceipt(operation,work.frames,result.frames,previous,current,details))
        work,previous = result,current

    if plan.duration_ratio is not None or plan.duration_frames is not None:
        target = plan.duration_frames if plan.duration_frames is not None else (
            target_frames_for_ratio(work.frames,plan.duration_ratio)
        )
        result = stretch_audio(work,target,settings=plan.time_scale)
        accept("duration",result.audio,result.report.as_dict())
    elif plan.pitch_semitones is not None:
        result = pitch_shift(work,plan.pitch_settings())
        accept("pitch",result.audio,result.report.as_dict())
    if plan.loop is not None:
        result = render_loop(work,plan.loop)
        accept("loop",result.audio,{"render":result.report.as_dict(),"search":None})
    elif plan.loop_search is not None:
        found = search_loop(work,plan.loop_search)
        render_plan = LoopRenderPlan.from_candidate(found.best,
            law=plan.search_render_law,repeats=plan.search_render_repeats,
            peak_ceiling=plan.search_render_peak_ceiling)
        result = render_loop(work,render_plan)
        accept("loop",result.audio,{"render":result.report.as_dict(),"search":found.as_dict()})
    if plan.mastering is not None:
        result = master_audio(work,plan.mastering)
        accept("master",result.audio,result.report.as_dict())
    if expected_output_pcm_sha256 is not None and previous != expected_output_pcm_sha256:
        raise AudioIntegrityError("Output PCM digest mismatch")
    output = PostProductionResult(work,plan,source,audio.frames,tuple(stages))
    # Fail before returning if the receipt is not canonical primitive JSON.
    output.manifest()
    return output


def verify_postproduction_manifest(record,*,expected_payload=None):
    """Verify checksum, plan identity and the ordered stage digest chain.

    This detects inconsistent receipt geometry; it does not authenticate
    the creator or prove that untrusted metadata describes a real execution.
    """
    body=verify_record(record,expected_payload=expected_payload)
    if body.get("protocol")!=PROTOCOL:
        raise AudioIntegrityError("Unexpected postproduction protocol")
    plan=body.get("plan")
    if type(plan) is not dict:
        raise AudioIntegrityError("Missing postproduction plan")
    if hashlib.sha256(canonical_json_bytes(plan)).hexdigest()!=body.get("plan_sha256"):
        raise AudioIntegrityError("Postproduction plan digest mismatch")
    stages=body.get("stages")
    if type(stages) is not list or not 1<=len(stages)<=3:
        raise AudioIntegrityError("Invalid stage receipt count")
    source_frames=body.get("source_frames")
    if type(source_frames) is not int or source_frames<=0:
        raise AudioIntegrityError("Invalid source frame receipt")
    previous=strict_sha256(body.get("source_pcm_sha256"))
    frames=source_frames
    order={"duration":0,"pitch":0,"loop":1,"master":2}
    prior=-1
    for stage in stages:
        if type(stage) is not dict or set(stage)!={"operation","input_frames","output_frames",
                "input_pcm_sha256","output_pcm_sha256","details"}:
            raise AudioIntegrityError("Unexpected stage receipt fields")
        operation=stage["operation"]
        if type(operation) is not str or operation not in order or order[operation]<=prior:
            raise AudioIntegrityError("Postproduction stage order is invalid")
        if stage["input_pcm_sha256"]!=previous or stage["input_frames"]!=frames:
            raise AudioIntegrityError("Postproduction stage chain is discontinuous")
        new_frames=stage["output_frames"]
        if type(new_frames) is not int or new_frames<=0:
            raise AudioIntegrityError("Invalid stage output frames")
        if operation in ("pitch","master") and new_frames!=frames:
            raise AudioIntegrityError("Fixed-duration stage changed frame count")
        if type(stage["details"]) is not dict:
            raise AudioIntegrityError("Invalid stage details")
        previous=strict_sha256(stage["output_pcm_sha256"])
        frames=new_frames
        prior=order[operation]
    if previous!=body["payload_sha256"] or frames!=body.get("output_frames"):
        raise AudioIntegrityError("Postproduction final receipt differs")
    return body
