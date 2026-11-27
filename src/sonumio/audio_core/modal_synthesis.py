"""Original procedural resonator synthesis for short sound effects.

This is a mathematical, damped-mode reference, not a recording,
a trained model, or a physically calibrated simulation. Excitation
and resonance are separate, deterministic components.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import numpy as np

from .contracts import AudioBuffer, AudioSpec, AudioLayout, check_rate, finite_float, make_buffer
from .errors import AudioContractError, AudioLimitError, AudioProcessingError
from .hashing import canonical_json_bytes

MAX_MODES = 32
MAX_EVENTS = 512
MAX_MODE_PRODUCTS = 60_000_000
MAX_RENDER_BYTES = 96*1024*1024
EXCITATIONS = ("strike", "tap", "scrape", "rattle", "step")


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise AudioContractError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class ResonanceMode:
    frequency_hz: float
    decay_seconds: float
    amplitude: float

    def __post_init__(self):
        finite_float(self.frequency_hz,"mode frequency",minimum=20,maximum=80000)
        finite_float(self.decay_seconds,"mode decay",minimum=.001,maximum=8)
        finite_float(self.amplitude,"mode amplitude",minimum=0,maximum=1)


@dataclass(frozen=True, slots=True)
class Resonator:
    modes: tuple[ResonanceMode,...]
    contact_noise: float = .15
    noise_decay_seconds: float = .02
    noise_color: float = .5

    def __post_init__(self):
        if (type(self.modes) is not tuple or not 1<=len(self.modes)<=MAX_MODES
                or any(not isinstance(m,ResonanceMode) for m in self.modes)):
            raise AudioContractError("Expected bounded tuple of resonance modes")
        if not any(m.amplitude for m in self.modes):
            raise AudioContractError("At least one resonance mode must be active")
        frequencies=[m.frequency_hz for m in self.modes]
        if frequencies!=sorted(set(frequencies)):
            raise AudioContractError("Resonance mode frequencies must increase strictly")
        finite_float(self.contact_noise,"contact noise",minimum=0,maximum=1)
        finite_float(self.noise_decay_seconds,"noise decay",minimum=.001,maximum=2)
        finite_float(self.noise_color,"noise color",minimum=0,maximum=1)


# Explicit authored relative modal profiles, not measurements of real objects.
# Different modal distributions and decay times distinguish the presets.
_PRESETS = {
    "wood": ((1.,1.59,2.31,3.77,5.12),(1.,.62,.38,.18,.08),(.16,.105,.07,.04,.025),.26,.018,.3),
    "metal": ((1.,1.47,2.09,2.76,4.13,5.79),(1.,.67,.5,.32,.2,.12),(.85,.69,.52,.36,.24,.16),.12,.01,.8),
    "glass": ((1.,2.32,3.86,5.43,7.19),(1.,.57,.35,.18,.1),(.6,.44,.29,.17,.10),.16,.008,.9),
    "ceramic": ((1.,1.78,2.63,3.15,4.91),(1.,.55,.35,.27,.12),(.28,.22,.13,.075,.04),.2,.016,.6),
    "stone": ((1.,1.35,2.18,3.02),(1.,.5,.25,.12),(.065,.045,.027,.017),.45,.035,.35),
    "rubber": ((1.,1.29,2.06),(1.,.38,.14),(.045,.028,.018),.18,.04,.1),
    "paper": ((1.,2.22,4.12),(1.,.42,.18),(.018,.012,.008),.9,.055,.8),
    "gravel": ((1.,1.55,2.43,3.65),(1.,.58,.29,.11),(.023,.018,.012,.009),.8,.065,.65),
}
MATERIALS = tuple(_PRESETS)


def material_resonator(material, *, fundamental_hz=240., size=1., damping=1.):
    if material not in MATERIALS:
        raise AudioContractError("Unknown procedural material")
    f=finite_float(fundamental_hz,"fundamental",minimum=40,maximum=4000)
    s=finite_float(size,"material size",minimum=.5,maximum=4)
    d=finite_float(damping,"material damping",minimum=.25,maximum=4)
    ratios,amplitudes,decays,noise,noise_decay,color=_PRESETS[material]
    modes=tuple(ResonanceMode(f*r/s,decay/d,amp)
                for r,amp,decay in zip(ratios,amplitudes,decays,strict=True))
    return Resonator(modes,noise,noise_decay/d,color)


@dataclass(frozen=True, slots=True)
class ContactEvent:
    start_frame: int
    frames: int
    resonator: Resonator
    excitation: str = "strike"
    strength: float = .5
    hardness: float = .5
    pitch_semitones: float = 0.
    pan: float = 0.
    seed: int = 0
    fade_out_frames: int = 64

    def __post_init__(self):
        _integer(self.start_frame,"event start",0,2**31-1)
        _integer(self.frames,"event length",8,2**31-1)
        if not isinstance(self.resonator,Resonator):
            raise AudioContractError("Expected Resonator")
        if self.excitation not in EXCITATIONS:
            raise AudioContractError("Unknown contact excitation")
        finite_float(self.strength,"event strength",minimum=0,maximum=1)
        finite_float(self.hardness,"contact hardness",minimum=0,maximum=1)
        finite_float(self.pitch_semitones,"contact pitch",minimum=-12,maximum=12)
        finite_float(self.pan,"contact pan",minimum=-1,maximum=1)
        _integer(self.seed,"event seed",0,2**32-1)
        _integer(self.fade_out_frames,"fade out",0,self.frames)


@dataclass(frozen=True, slots=True)
class ContactScene:
    sample_rate: int
    frames: int
    events: tuple[ContactEvent,...]
    channels: int = 2
    peak_ceiling: float | None = .9

    def __post_init__(self):
        check_rate(self.sample_rate)
        _integer(self.channels,"channels",1,2)
        AudioSpec(self.sample_rate,AudioLayout.for_channels(self.channels),self.frames)
        if (type(self.events) is not tuple or not 1<=len(self.events)<=MAX_EVENTS
                or any(not isinstance(e,ContactEvent) for e in self.events)):
            raise AudioContractError("Expected bounded contact event tuple")
        if any(e.start_frame+e.frames>self.frames for e in self.events):
            raise AudioContractError("Contact event extends beyond scene")
        if self.channels==1 and any(e.pan!=0 for e in self.events):
            raise AudioContractError("Mono contact scenes do not support panning")
        if self.peak_ceiling is not None:
            finite_float(self.peak_ceiling,"scene peak ceiling",minimum=.001,maximum=1)

    @property
    def sha256(self):
        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class ContactEventReport:
    event_index: int
    start_frame: int
    frames: int
    excitation: str
    active_modes: int
    omitted_modes: int
    impulses: int
    peak: float
    rms: float


@dataclass(frozen=True, slots=True)
class ContactSceneReport:
    sample_rate: int
    frames: int
    channels: int
    event_reports: tuple[ContactEventReport,...]
    pre_limit_peak: float
    output_peak: float
    attenuation: float
    scene_sha256: str
    protocol: str = "sonumio-modal-contact-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "model_inference":False,
            "physically_calibrated":False,
            "perceptual_quality_established":False,
        }


@dataclass(frozen=True, slots=True)
class ContactSceneResult:
    audio: AudioBuffer
    report: ContactSceneReport


def _rng(event, domain):
    return np.random.default_rng(np.random.SeedSequence([event.seed,domain,0x434F4E54]))


def excitation_impulses(event, rate):
    """A bounded original impulse schedule on the event-local clock."""
    if not isinstance(event,ContactEvent):
        raise AudioContractError("Expected ContactEvent")
    check_rate(rate)
    rng=_rng(event,1)
    length=event.frames
    if event.excitation=="strike":
        return ((0,1.),)
    if event.excitation=="tap":
        delay=min(length-1,round(rate*.012))
        return ((0,.75),(delay,.12)) if delay else ((0,.87),)
    if event.excitation=="step":
        # Heel and toe contacts, with independent surface microcontacts.
        centers=[(0,.7),(min(length-1,round(rate*.085)),.48)]
        count=min(16,max(1,round(length/rate*35)))
        centers.extend((int(rng.integers(0,length)),float(rng.uniform(.025,.11)))
                       for _ in range(count))
    elif event.excitation=="rattle":
        count=min(64,max(2,round(length/rate*(8+30*event.hardness))))
        centers=[(int(rng.integers(0,length)),float(rng.uniform(.08,.6)))
                 for _ in range(count)]
        centers.append((0,.35))
    else:
        # Irregular low-strength contact pulses over the declared duration.
        count=min(96,max(2,round(length/rate*(20+60*event.hardness))))
        centers=[(int(rng.integers(0,length)),float(rng.uniform(.03,.18)))
                 for _ in range(count)]
        centers.append((0,.1))
    combined={}
    for index,amplitude in centers:
        combined[index]=combined.get(index,0.)+amplitude
    return tuple(sorted(combined.items()))


def _mode_coefficients(event,rate):
    scale=2**(event.pitch_semitones/12)
    active=[]
    for mode in event.resonator.modes:
        freq=mode.frequency_hz*scale
        if freq>=rate*.46 or not mode.amplitude:
            continue
        # Hard contacts excite higher modes more strongly, while soft
        # contacts use a smoother, lower-frequency distribution.
        corner=150+4500*event.hardness
        weight=mode.amplitude/math.sqrt(1+(freq/corner)**2)
        active.append((freq,mode.decay_seconds,weight))
    if not active:
        raise AudioContractError("No resonance mode below sample-clock guard")
    total=sum(m[2] for m in active)
    return tuple((f,d,w/total) for f,d,w in active)


def _modal_signal(event,rate,impulses,modes):
    n=event.frames
    t=np.arange(n,dtype=np.float64)/rate
    output=np.zeros(n,dtype=np.float64)
    for frequency,decay,weight in modes:
        # A sine impulse starts at zero; exponential decay is a declared
        # amplitude e-folding constant, not a measured acoustic Q.
        response=np.sin(2*np.pi*frequency*t)*np.exp(-t/decay)*weight
        for start,amplitude in impulses:
            output[start:]+=amplitude*response[:n-start]
    return output


def _contact_noise(event,rate,impulses):
    n=event.frames
    rng=_rng(event,2)
    raw=rng.standard_normal(n)
    f=np.fft.rfftfreq(n,1/rate)
    color=event.resonator.noise_color
    cutoff=150+color*min(9000.,rate*.42-150)
    shape=(f/np.maximum(f+50,1)) / np.sqrt(1+(f/cutoff)**4)
    noise=np.fft.irfft(np.fft.rfft(raw)*shape,n=n)
    rms=float(np.sqrt(np.mean(noise*noise)))
    if rms>1e-12:
        noise/=rms
    else:
        noise.fill(0.)
    envelope=np.zeros(n,dtype=np.float64)
    decay=event.resonator.noise_decay_seconds*(1.4-.6*event.hardness)
    length=min(n,max(1,math.ceil(8*decay*rate)))
    tail=np.exp(-np.arange(length,dtype=np.float64)/(rate*decay))
    for start,amplitude in impulses:
        stop=min(n,start+length)
        envelope[start:stop]+=amplitude*tail[:stop-start]
    return noise*envelope*event.resonator.contact_noise


def _render_contact(event,rate,impulses,modes):
    mono=(_modal_signal(event,rate,impulses,modes)
          +_contact_noise(event,rate,impulses))*event.strength
    # Taper the beginning and explicitly requested ending to suppress
    # abrupt boundaries. Long natural decays remain intact until the fade.
    attack=min(event.frames,max(2,round(rate*(.0004+.002*(1-event.hardness)))))
    mono[:attack]*=np.sin(np.linspace(0,np.pi/2,attack))**2
    fade=event.fade_out_frames
    if fade==1:
        mono[-1]=0.
    elif fade:
        gain=np.cos(np.linspace(0,np.pi/2,fade))**2
        gain[-1]=0.
        mono[-fade:]*=gain
    if not np.isfinite(mono).all():
        raise AudioProcessingError("Contact renderer produced nonfinite PCM")
    return mono


def render_contact_scene(scene):
    """Prepare and bound all event schedules before allocating output."""
    if not isinstance(scene,ContactScene):
        raise AudioContractError("Expected ContactScene")
    prepared=[]
    products=0
    maximum_event=0
    for event in scene.events:
        impulses=excitation_impulses(event,scene.sample_rate)
        modes=_mode_coefficients(event,scene.sample_rate)
        products+=event.frames*(len(impulses)*(len(modes)+1)+16)
        maximum_event=max(maximum_event,event.frames)
        if products>MAX_MODE_PRODUCTS:
            raise AudioLimitError("Contact synthesis exceeds mode-product budget")
        prepared.append((event,impulses,modes))
    if scene.frames*scene.channels*24+maximum_event*96>MAX_RENDER_BYTES:
        raise AudioLimitError("Contact synthesis exceeds working-memory budget")
    output=np.zeros((scene.channels,scene.frames),dtype=np.float64)
    reports=[]
    for index,(event,impulses,modes) in enumerate(prepared):
        mono=_render_contact(event,scene.sample_rate,impulses,modes)
        span=slice(event.start_frame,event.start_frame+event.frames)
        if scene.channels==1:
            output[0,span]+=mono
        else:
            angle=(event.pan+1)*np.pi/4
            left=0. if event.pan==1 else math.cos(angle)
            right=0. if event.pan==-1 else math.sin(angle)
            output[0,span]+=mono*left
            output[1,span]+=mono*right
        reports.append(ContactEventReport(
            index,event.start_frame,event.frames,event.excitation,len(modes),
            len(event.resonator.modes)-len(modes),len(impulses),
            float(np.max(np.abs(mono))),float(np.sqrt(np.mean(mono**2)))))
    peak=float(np.max(np.abs(output)))
    attenuation=1.
    if scene.peak_ceiling is not None and peak>scene.peak_ceiling:
        attenuation=scene.peak_ceiling/peak
        output*=attenuation
    if not np.isfinite(output).all() or np.max(np.abs(output))>np.finfo(np.float32).max:
        raise AudioProcessingError("Contact mix exceeds finite PCM representation")
    result=make_buffer(output.astype(np.float32),scene.sample_rate,
                       source_note="original modal contact sound scene")
    return ContactSceneResult(result,ContactSceneReport(
        scene.sample_rate,scene.frames,scene.channels,tuple(reports),peak,
        float(np.max(np.abs(result.samples))),attenuation,scene.sha256))
