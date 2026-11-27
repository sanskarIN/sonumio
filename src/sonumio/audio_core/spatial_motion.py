"""Original offline 2-D moving-source stereo propagation reference.

Distances are measured in metres. Each source-frame emission is mapped
to its receiver arrival clock by a geometric propagation delay. The
monotone arrival map is interpolated at integer receiver frames.
This is a simplified stereo renderer: no measured HRTFs, diffraction,
air-absorption calibration, Doppler anti-alias guarantee or real-time
simulation is implied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import numpy as np

from .contracts import AudioBuffer, AudioLayout, AudioSpec, check_rate, finite_float, make_buffer
from .errors import AudioContractError, AudioLimitError, AudioProcessingError
from .hashing import canonical_json_bytes, digest_audio

MAX_KEYS = 128
MAX_SOURCES = 32
MAX_PATH_FRAMES = 8_000_000
MAX_WORK_BYTES = 160 * 1024 * 1024
MAX_SPEED_MPS = 40.
WALLS = ("left", "right", "bottom", "top")


def _integer(value,name,low,high):
    if type(value) is not int or not low<=value<=high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True,slots=True)
class Position:
    x: float
    y: float

    def __post_init__(self):
        finite_float(self.x,"position x",minimum=-100,maximum=100)
        finite_float(self.y,"position y",minimum=-100,maximum=100)


@dataclass(frozen=True,slots=True)
class MotionKey:
    frame: int
    position: Position

    def __post_init__(self):
        _integer(self.frame,"motion frame",0,2**31-1)
        if not isinstance(self.position,Position):
            raise AudioContractError("Expected Position")


@dataclass(frozen=True,slots=True)
class Trajectory:
    keys: tuple[MotionKey,...]

    def __post_init__(self):
        if (type(self.keys) is not tuple or not 1<=len(self.keys)<=MAX_KEYS
                or any(not isinstance(k,MotionKey) for k in self.keys)):
            raise AudioContractError("Expected bounded motion-key tuple")
        frames=[k.frame for k in self.keys]
        if frames[0]!=0 or frames!=sorted(set(frames)):
            raise AudioContractError("Motion starts at zero and keys strictly increase")

    @classmethod
    def stationary(cls,x,y):
        return cls((MotionKey(0,Position(x,y)),))

    def validate_clock(self,rate,frames):
        check_rate(rate)
        _integer(frames,"source frames",1,2**31-1)
        if self.keys[-1].frame>=frames:
            raise AudioContractError("Motion key lies outside source emission")
        for a,b in zip(self.keys[:-1],self.keys[1:]):
            distance=math.hypot(b.position.x-a.position.x,b.position.y-a.position.y)
            speed=distance*rate/(b.frame-a.frame)
            if speed>MAX_SPEED_MPS:
                raise AudioContractError("Motion exceeds bounded subsonic speed")

    def positions(self,rate,frames):
        self.validate_clock(rate,frames)
        positions=np.arange(frames,dtype=np.float64)
        key_frames=[k.frame for k in self.keys]
        return (
            np.interp(positions,key_frames,[k.position.x for k in self.keys]),
            np.interp(positions,key_frames,[k.position.y for k in self.keys]),
        )


@dataclass(frozen=True,slots=True)
class Room:
    width: float
    depth: float
    left: float = .2
    right: float = .2
    bottom: float = .2
    top: float = .2

    def __post_init__(self):
        finite_float(self.width,"room width",minimum=1,maximum=100)
        finite_float(self.depth,"room depth",minimum=1,maximum=100)
        for wall in WALLS:
            finite_float(getattr(self,wall),wall+" reflection",minimum=0,maximum=.95)

    def contains(self,point):
        return 0<=point.x<=self.width and 0<=point.y<=self.depth

    @property
    def active_walls(self):
        return tuple(w for w in WALLS if getattr(self,w)>0)


@dataclass(frozen=True,slots=True)
class Receiver:
    position: Position = Position(0.,0.)
    heading_degrees: float = 0.
    spacing_m: float = .18
    sound_speed: float = 343.
    reference_distance_m: float = 1.
    rolloff: float = 1.
    directional_mix: float = .5

    def __post_init__(self):
        if not isinstance(self.position,Position):
            raise AudioContractError("Expected receiver Position")
        finite_float(self.heading_degrees,"heading",minimum=-360,maximum=360)
        finite_float(self.spacing_m,"receiver spacing",minimum=0,maximum=1)
        finite_float(self.sound_speed,"sound speed",minimum=300,maximum=380)
        finite_float(self.reference_distance_m,"reference distance",minimum=.05,maximum=20)
        finite_float(self.rolloff,"distance rolloff",minimum=0,maximum=2)
        finite_float(self.directional_mix,"directional mix",minimum=0,maximum=1)

    def geometry(self):
        heading=math.radians(self.heading_degrees)
        # Heading zero faces +y; local right is +x.
        right=np.array([math.cos(heading),-math.sin(heading)])
        center=np.array([self.position.x,self.position.y])
        return center-right*self.spacing_m/2,center+right*self.spacing_m/2,right


@dataclass(frozen=True,slots=True,eq=False)
class MovingSource:
    audio: AudioBuffer
    trajectory: Trajectory
    start_frame: int = 0
    gain_db: float = 0.

    def __post_init__(self):
        if not isinstance(self.audio,AudioBuffer) or self.audio.channels!=1:
            raise AudioContractError("Spatial source requires mono AudioBuffer")
        if not isinstance(self.trajectory,Trajectory):
            raise AudioContractError("Expected Trajectory")
        _integer(self.start_frame,"source start",0,2**31-1)
        finite_float(self.gain_db,"source gain",minimum=-60,maximum=12)
        self.trajectory.validate_clock(self.audio.sample_rate,self.audio.frames)


@dataclass(frozen=True,slots=True)
class SpatialScene:
    sample_rate: int
    frames: int
    sources: tuple[MovingSource,...]
    receiver: Receiver = Receiver()
    room: Room | None = None
    peak_ceiling: float | None = .95

    def __post_init__(self):
        AudioSpec(self.sample_rate,AudioLayout.for_channels(2),self.frames)
        if (type(self.sources) is not tuple or not 1<=len(self.sources)<=MAX_SOURCES
                or any(not isinstance(s,MovingSource) for s in self.sources)):
            raise AudioContractError("Expected bounded moving-source tuple")
        if not isinstance(self.receiver,Receiver):
            raise AudioContractError("Expected Receiver")
        if self.room is not None and not isinstance(self.room,Room):
            raise AudioContractError("Expected Room")
        if any(s.audio.sample_rate!=self.sample_rate for s in self.sources):
            raise AudioContractError("Spatial source sample clock mismatch")
        if any(s.start_frame>=self.frames for s in self.sources):
            raise AudioContractError("Source starts outside receiver scene")
        if self.peak_ceiling is not None:
            finite_float(self.peak_ceiling,"peak ceiling",minimum=.001,maximum=1)
        if self.room is not None:
            left,right,_=self.receiver.geometry()
            if not self.room.contains(self.receiver.position) or any(
                not self.room.contains(Position(float(p[0]),float(p[1]))) for p in (left,right)
            ):
                raise AudioContractError("Receiver lies outside reflection room")
            if any(not self.room.contains(k.position) for s in self.sources for k in s.trajectory.keys):
                raise AudioContractError("Source trajectory leaves reflection room")

    def metadata(self):
        return {
            "sample_rate":self.sample_rate,"frames":self.frames,
            "receiver":asdict(self.receiver),
            "room":asdict(self.room) if self.room else None,
            "peak_ceiling":self.peak_ceiling,
            "sources":[{
                "pcm_sha256":digest_audio(s.audio),"source_frames":s.audio.frames,
                "trajectory":asdict(s.trajectory),"start_frame":s.start_frame,"gain_db":s.gain_db,
            } for s in self.sources],
        }


@dataclass(frozen=True,slots=True)
class PropagationPathReport:
    source_index: int
    path: str
    channel: int
    first_arrival_frame: float
    last_arrival_frame: float
    minimum_distance_m: float
    maximum_distance_m: float
    minimum_arrival_step: float | None
    maximum_arrival_step: float | None
    rendered_frames: int
    truncated: bool
    peak: float


@dataclass(frozen=True,slots=True)
class SpatialReport:
    sample_rate: int
    frames: int
    source_count: int
    paths: tuple[PropagationPathReport,...]
    pre_limit_peak: float
    output_peak: float
    attenuation: float
    scene_sha256: str
    protocol: str = "sonumio-geometric-stereo-motion-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "interpolation":"linear_receiver_clock",
            "reflection_order":1,
            "measured_hrtf":False,
            "physically_calibrated":False,
            "doppler_antialias_guaranteed":False,
            "model_inference":False,
        }


@dataclass(frozen=True,slots=True)
class SpatialResult:
    audio: AudioBuffer
    report: SpatialReport


def image_positions(x,y,room,wall):
    if not isinstance(room,Room) or wall not in WALLS:
        raise AudioContractError("Invalid first-order reflection wall")
    if wall=="left":
        return -x,y
    if wall=="right":
        return 2*room.width-x,y
    if wall=="bottom":
        return x,-y
    return x,2*room.depth-y


def _path_geometry(x,y,ear,receiver,source_start,rate,emission):
    distance=np.hypot(x-ear[0],y-ear[1])
    arrival=source_start+emission+distance*(rate/receiver.sound_speed)
    steps=np.diff(arrival)
    if np.any(steps<=0) or not np.isfinite(arrival).all():
        raise AudioProcessingError("Nonmonotone propagation arrival clock")
    return distance,arrival,steps


def _directional_gains(x,y,receiver,right):
    dx=x-receiver.position.x
    dy=y-receiver.position.y
    distance=np.hypot(dx,dy)
    lateral=np.divide(dx*right[0]+dy*right[1],distance,
                      out=np.zeros_like(distance),where=distance>1e-12)
    lateral=np.clip(lateral,-1,1)
    angle=(lateral+1)*np.pi/4
    # Mix unity-distance stereo with a simple equal-power speaker pan.
    # No frequency-dependent ear filtering is claimed.
    blend=receiver.directional_mix
    base=(1-blend)/math.sqrt(2)
    return base+blend*np.cos(angle),base+blend*np.sin(angle)


def _arrival_mix(output,row,arrival,values):
    """Accumulate only the receiver samples within the actual path domain."""
    first=float(arrival[0])
    last=float(arrival[-1])
    begin=max(0,math.ceil(first))
    stop=min(output.shape[1],math.floor(last)+1)
    if stop<=begin:
        return 0,0.
    if len(arrival)==1:
        # A single sample exists only at its exact arrival coordinate;
        # do not invent a hold interval or a synthetic impulse kernel.
        if not math.isclose(first,begin,rel_tol=0,abs_tol=1e-9):
            return 0,0.
        signal=np.array([values[0]])
    else:
        signal=np.interp(np.arange(begin,stop,dtype=np.float64),arrival,values)
    output[row,begin:stop]+=signal
    return len(signal),float(np.max(np.abs(signal)))


def render_spatial_scene(scene):
    if not isinstance(scene,SpatialScene):
        raise AudioContractError("Expected SpatialScene")
    walls=scene.room.active_walls if scene.room else ()
    path_frames=sum(s.audio.frames for s in scene.sources)*2*(1+len(walls))
    if path_frames>MAX_PATH_FRAMES:
        raise AudioLimitError("Spatial propagation exceeds path-frame budget")
    largest=max(s.audio.frames for s in scene.sources)
    # Includes receiver accumulation and float output copies plus one
    # source/path geometry, direction gain, arrival and interpolation arrays.
    if scene.frames*80+largest*160>MAX_WORK_BYTES:
        raise AudioLimitError("Spatial propagation exceeds working-memory budget")
    left,right,axis=scene.receiver.geometry()
    output=np.zeros((2,scene.frames),dtype=np.float64)
    reports=[]
    receiver=scene.receiver
    for index,source in enumerate(scene.sources):
        x,y=source.trajectory.positions(scene.sample_rate,source.audio.frames)
        emission=np.arange(source.audio.frames,dtype=np.float64)
        source_pcm=source.audio.samples[0].astype(np.float64)*10**(source.gain_db/20)
        for path in ("direct",*walls):
            if path=="direct":
                px,py=x,y
                reflection=1.
            else:
                px,py=image_positions(x,y,scene.room,path)
                reflection=getattr(scene.room,path)
            gains=_directional_gains(px,py,receiver,axis)
            for channel,ear in enumerate((left,right)):
                distance,arrival,steps=_path_geometry(
                    px,py,ear,receiver,source.start_frame,scene.sample_rate,emission)
                attenuation=(receiver.reference_distance_m/
                    np.maximum(receiver.reference_distance_m,distance))**receiver.rolloff
                values=source_pcm*attenuation*gains[channel]*reflection
                frames,peak=_arrival_mix(output,channel,arrival,values)
                reports.append(PropagationPathReport(
                    index,path,channel,float(arrival[0]),float(arrival[-1]),
                    float(distance.min()),float(distance.max()),
                    float(steps.min()) if len(steps) else None,
                    float(steps.max()) if len(steps) else None,
                    frames,bool(arrival[-1]>scene.frames-1),peak,
                ))
    pre=float(np.max(np.abs(output)))
    scale=1.
    if scene.peak_ceiling is not None and pre>scene.peak_ceiling:
        scale=scene.peak_ceiling/pre
        output*=scale
    if not np.isfinite(output).all() or np.max(np.abs(output))>np.finfo(np.float32).max:
        raise AudioProcessingError("Spatial propagation produced invalid PCM")
    result=make_buffer(output.astype(np.float32),scene.sample_rate,
                       source_note="geometric stereo moving-source propagation")
    digest=hashlib.sha256(canonical_json_bytes(scene.metadata())).hexdigest()
    return SpatialResult(result,SpatialReport(scene.sample_rate,scene.frames,len(scene.sources),
        tuple(reports),pre,float(np.max(np.abs(result.samples))),scale,digest))
