"""Original deterministic granular synthesis from locally supplied PCM.

A grain schedule is computed before rendering, independently of source
sample values. Each grain reads a fractional position from the source,
applies an exact envelope and is accumulated in float64. The renderer
is offline, not a trained model, real-time engine or voice synthesizer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

from .contracts import AudioBuffer, AudioLayout, AudioSpec, finite_float, make_buffer
from .errors import AudioContractError, AudioLimitError, AudioProcessingError
from .hashing import canonical_json_bytes
import hashlib

MAX_GRAINS = 8192
MAX_GRAIN_PRODUCTS = 80_000_000
MAX_RENDER_BYTES = 128 * 1024 * 1024
BOUNDARIES = ("wrap", "reflect", "zero")
WINDOWS = ("hann", "triangle", "tukey")
NORMALIZATION = ("none", "overlap")


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class GranularSettings:
    output_frames: int
    grain_frames: int = 1024
    hop_frames: int = 256
    source_start: float = 0.
    source_end: float = 1.
    position_jitter: float = .05
    time_jitter_frames: int = 0
    pitch_semitones: float = 0.
    pitch_jitter_semitones: float = 0.
    reverse_probability: float = 0.
    amplitude: float = .5
    amplitude_jitter: float = 0.
    stereo_spread: float = 0.
    seed: int = 0
    boundary: str = "wrap"
    window: str = "hann"
    normalization: str = "overlap"
    peak_ceiling: float | None = .95

    def __post_init__(self):
        _integer(self.output_frames, "output_frames", 1, 2**31-1)
        _integer(self.grain_frames, "grain_frames", 8, 32768)
        _integer(self.hop_frames, "hop_frames", 1, 32768)
        _integer(self.time_jitter_frames, "time_jitter_frames", 0, self.hop_frames//2)
        _integer(self.seed, "seed", 0, 2**32-1)
        for name, low, high in (
            ("source_start", 0, 1), ("source_end", 0, 1),
            ("position_jitter", 0, 1), ("pitch_semitones", -24, 24),
            ("pitch_jitter_semitones", 0, 12), ("reverse_probability", 0, 1),
            ("amplitude", 0, 2), ("amplitude_jitter", 0, 1),
            ("stereo_spread", 0, 1),
        ):
            finite_float(getattr(self, name), name, minimum=low, maximum=high)
        if abs(self.pitch_semitones)+self.pitch_jitter_semitones > 24:
            raise AudioContractError("Combined grain pitch exceeds two octaves")
        if self.boundary not in BOUNDARIES or self.window not in WINDOWS:
            raise AudioContractError("Unknown grain boundary or window")
        if self.normalization not in NORMALIZATION:
            raise AudioContractError("Unknown grain normalization policy")
        if self.peak_ceiling is not None:
            finite_float(self.peak_ceiling, "peak ceiling", minimum=.001, maximum=1.)

    @property
    def count(self):
        # Include one center at or past the final output sample so that
        # centered grains can cover the right boundary.
        return (self.output_frames-1+self.hop_frames-1)//self.hop_frames+1

    @property
    def sha256(self):
        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class GrainEvent:
    index: int
    center_frame: int
    source_center: float
    rate: float
    amplitude: float
    pan: float


@dataclass(frozen=True, slots=True)
class GranularReport:
    settings: GranularSettings
    source_frames: int
    source_channels: int
    output_channels: int
    sample_rate: int
    grains_scheduled: int
    grains_rendered: int
    reversed_grains: int
    uncovered_frames: int
    maximum_overlap_weight: float
    pre_limit_peak: float
    output_peak: float
    attenuation: float
    schedule_sha256: str
    protocol: str = "sonumio-granular-pcm-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "settings_sha256": self.settings.sha256,
            "model_inference": False,
            "interpolation": "windowed_sinc",
            "perceptual_quality_established": False,
        }


@dataclass(frozen=True, slots=True)
class GranularResult:
    audio: AudioBuffer
    report: GranularReport


def grain_schedule(source_frames, settings):
    """Each independent random stream belongs to one event/property.

    The seed and index determine a grain irrespective of other events'
    random draws. Positions are clamped only at the source-center level;
    individual taps obey the separately selected boundary policy.
    """
    _integer(source_frames, "source_frames", 1, 2**31-1)
    if not isinstance(settings, GranularSettings):
        raise AudioContractError("Expected GranularSettings")
    if settings.count > MAX_GRAINS:
        raise AudioLimitError("Grain schedule exceeds event budget")
    events = []
    for index in range(settings.count):
        nominal = index*settings.hop_frames
        fraction = min(1., nominal/max(1, settings.output_frames-1))
        rng = np.random.default_rng(np.random.SeedSequence([settings.seed,index,0x47524149]))
        # Draw all properties regardless of whether their range is zero.
        jitter, position, pitch, reverse, gain, pan = rng.uniform(-1.,1.,6)
        center = nominal + round(jitter*settings.time_jitter_frames)
        source_fraction = settings.source_start+(settings.source_end-settings.source_start)*fraction
        source_fraction = float(np.clip(source_fraction+position*settings.position_jitter,0,1))
        semitones = settings.pitch_semitones+pitch*settings.pitch_jitter_semitones
        rate = 2**(semitones/12)
        if (reverse+1)/2 < settings.reverse_probability:
            rate = -rate
        amplitude = settings.amplitude*(1+gain*settings.amplitude_jitter)
        events.append(GrainEvent(index,center,source_fraction*(source_frames-1),
                                 float(rate),float(amplitude),float(pan*settings.stereo_spread)))
    return tuple(events)


def grain_window(frames, name):
    _integer(frames,"grain window frames",8,32768)
    if name not in WINDOWS:
        raise AudioContractError("Unknown grain envelope")
    x = np.linspace(0.,1.,frames)
    if name == "hann":
        out = .5-.5*np.cos(2*np.pi*x)
    elif name == "triangle":
        out = 1-np.abs(2*x-1)
    else:
        out = np.ones(frames)
        edge = x < .25
        out[edge] = .5-.5*np.cos(4*np.pi*x[edge])
        edge = x > .75
        out[edge] = .5-.5*np.cos(4*np.pi*(1-x[edge]))
    out[0] = out[-1] = 0.
    return out


def _map_indices(indices, frames, boundary):
    if boundary == "wrap":
        return np.mod(indices,frames), None
    if boundary == "reflect":
        if frames == 1:
            return np.zeros_like(indices), None
        period = 2*(frames-1)
        x = np.mod(indices,period)
        return np.minimum(x,period-x), None
    return np.clip(indices,0,frames-1), (indices>=0)&(indices<frames)


def _read_grain(samples, positions, rate, boundary):
    """Finite low-pass windowed sinc, with rate-dependent cutoff.

    A fixed radius would be too narrow when reading fast. Increase the
    support in source samples to keep a constant number of sinc lobes.
    Interior and periodic/reflect boundaries preserve DC. Zero extension
    deliberately does not renormalize missing source taps.
    """
    cutoff = min(1.,1/abs(rate))
    radius = 12/cutoff
    width = math.ceil(radius)
    taps = np.arange(-width+1,width+1)
    result = np.empty((samples.shape[0],len(positions)),dtype=np.float64)
    block = max(1,65536//len(taps))
    for start in range(0,len(positions),block):
        pos = positions[start:start+block]
        indices = np.floor(pos).astype(np.int64)[:,None]+taps
        distances = pos[:,None]-indices
        window = np.where(np.abs(distances)<radius,
                          .5+.5*np.cos(np.pi*distances/radius),0.)
        weights = cutoff*np.sinc(cutoff*distances)*window
        total = weights.sum(axis=1)
        if np.any(np.abs(total)<1e-12):
            raise AudioProcessingError("Degenerate granular interpolation")
        weights /= total[:,None]
        mapped, valid = _map_indices(indices,samples.shape[1],boundary)
        if valid is not None:
            weights *= valid
        for ch, row in enumerate(samples):
            result[ch,start:start+len(pos)] = np.sum(row[mapped]*weights,axis=1)
    return result


def _preflight(audio, settings):
    stereo = audio.channels == 1 and settings.stereo_spread > 0
    if audio.channels > 2 and settings.stereo_spread:
        raise AudioContractError("Grain panning supports mono and stereo only")
    layout = AudioLayout.for_channels(2) if stereo else audio.spec.layout
    spec = AudioSpec(audio.sample_rate,layout,settings.output_frames)
    work = settings.count*settings.grain_frames*audio.channels*96
    if settings.count > MAX_GRAINS or work > MAX_GRAIN_PRODUCTS:
        raise AudioLimitError("Granular interpolation work exceeds budget")
    memory = settings.output_frames*(layout.channels*24+16)+65536*48
    if memory > MAX_RENDER_BYTES:
        raise AudioLimitError("Granular synthesis exceeds memory budget")
    return spec


def _pan_grain(grain, event, channels):
    if channels == 2 and grain.shape[0] == 1:
        angle = (event.pan+1)*np.pi/4
        return np.vstack((grain[0]*np.cos(angle),grain[0]*np.sin(angle)))
    if grain.shape[0] == 2:
        # Balance existing stereo without cross-channel mixing.
        return grain*np.array([min(1.,1-event.pan),min(1.,1+event.pan)])[:,None]
    return grain


def synthesize_grains(audio, settings):
    """Render exact output geometry, with explicit overlap normalization.

    Normalization divides by max(1, summed window weights), not by the
    number of grains. Consequently isolated grain edges remain tapered,
    dense overlaps avoid unbounded gain and gaps remain silent.
    """
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, GranularSettings):
        raise AudioContractError("Expected AudioBuffer and GranularSettings")
    spec = _preflight(audio,settings)
    events = grain_schedule(audio.frames,settings)
    window = grain_window(settings.grain_frames,settings.window)
    output = np.zeros((spec.channels,spec.frames),dtype=np.float64)
    weights = np.zeros(spec.frames,dtype=np.float64)
    half = settings.grain_frames//2
    rendered = reversed_count = 0
    for event in events:
        begin = event.center_frame-half
        stop = begin+settings.grain_frames
        a,b = max(0,begin),min(spec.frames,stop)
        if a>=b:
            continue
        local = np.arange(a-begin,b-begin,dtype=np.float64)
        positions = event.source_center+(local-half)*event.rate
        grain = _read_grain(audio.samples,positions,event.rate,settings.boundary)
        grain = _pan_grain(grain,event,spec.channels)
        env = window[a-begin:b-begin]
        output[:,a:b] += grain*env*event.amplitude
        weights[a:b] += env
        rendered += 1
        reversed_count += int(event.rate<0)
    if settings.normalization == "overlap":
        output /= np.maximum(1.,weights)
    pre = float(np.max(np.abs(output)))
    attenuation = 1.
    if settings.peak_ceiling is not None and pre>settings.peak_ceiling:
        attenuation = settings.peak_ceiling/pre
        output *= attenuation
    if not np.isfinite(output).all() or np.max(np.abs(output))>np.finfo(np.float32).max:
        raise AudioProcessingError("Granular synthesis produced invalid PCM")
    result = make_buffer(output.astype(np.float32),audio.sample_rate,layout=spec.layout,
                         source_note="original local granular PCM synthesis")
    digest = hashlib.sha256(canonical_json_bytes([asdict(e) for e in events])).hexdigest()
    report = GranularReport(settings,audio.frames,audio.channels,spec.channels,
        audio.sample_rate,len(events),rendered,reversed_count,
        int(np.count_nonzero(weights==0)),float(weights.max()),pre,
        float(np.max(np.abs(result.samples))),attenuation,digest)
    return GranularResult(result,report)
