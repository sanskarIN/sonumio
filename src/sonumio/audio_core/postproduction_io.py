"""Strict JSON plans and verified single-publication postproduction WAV export."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import stat
import numpy as np

from .errors import AudioContractError, AudioFormatError, AudioLimitError
from .hashing import parse_canonical_json, strict_sha256
from .time_scale import TimeScaleSettings
from .loop_analysis import LoopSearchPlan, LoopScoreWeights
from .loop_render import LoopRenderPlan
from .mastering import MasteringSettings
from .interpolated_peak import PeakEstimateSettings
from .gated_energy import GatedEnergySettings
from .mastering_wav import encode_mastered_audio, ENCODED_PEAK_POLICIES
from .postproduction import PostProductionPlan, run_postproduction
from .wave_snapshot import read_verified_pcm
from .dither import QuantizationSettings
from .waveio import encode_wav_bytes

MAX_PLAN_BYTES=64*1024


def _object(value,cls):
    if type(value) is not dict or not set(value)<=set(cls.__dataclass_fields__):
        raise AudioContractError("Unexpected fields for "+cls.__name__)
    return dict(value)


def parse_postproduction_plan(raw):
    data=_object(parse_canonical_json(raw,max_bytes=MAX_PLAN_BYTES),PostProductionPlan)
    if "time_scale" in data:
        data["time_scale"]=TimeScaleSettings(**_object(data["time_scale"],TimeScaleSettings))
    if data.get("loop") is not None:
        data["loop"]=LoopRenderPlan(**_object(data["loop"],LoopRenderPlan))
    if data.get("loop_search") is not None:
        search=_object(data["loop_search"],LoopSearchPlan)
        if "weights" in search:
            search["weights"]=LoopScoreWeights(**_object(search["weights"],LoopScoreWeights))
        data["loop_search"]=LoopSearchPlan(**search)
    if data.get("mastering") is not None:
        master=_object(data["mastering"],MasteringSettings)
        if "peak" in master:
            master["peak"]=PeakEstimateSettings(**_object(master["peak"],PeakEstimateSettings))
        if "energy" in master:
            master["energy"]=GatedEnergySettings(**_object(master["energy"],GatedEnergySettings))
        data["mastering"]=MasteringSettings(**master)
    return PostProductionPlan(**data)


def load_postproduction_plan(path):
    path=Path(path)
    # A bounded descriptor read avoids accepting FIFO or symlink inputs.
    if path.is_symlink() or not path.is_file():
        raise AudioFormatError("Expected ordinary local JSON plan")
    flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0)|getattr(os,"O_BINARY",0)
    fd=os.open(path,flags)
    with os.fdopen(fd,"rb") as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise AudioFormatError("Expected regular JSON descriptor")
        if info.st_size>MAX_PLAN_BYTES:
            raise AudioLimitError("Postproduction plan exceeds byte budget")
        raw=stream.read(MAX_PLAN_BYTES+1)
    return parse_postproduction_plan(raw)


@dataclass(frozen=True,slots=True)
class PostProductionExport:
    source: dict
    manifest: dict
    output_sha256: str
    output_bytes: int
    output_bits: int
    dither_seed: int | None
    encoded_peak: dict | None
    encoding_attenuation_db: float
    protocol: str="sonumio-verified-postproduction-export-v1"

    def as_dict(self):
        return {
            **self.source,
            "protocol":self.protocol,
            "manifest":self.manifest,
            "output_sha256":self.output_sha256,
            "output_bytes":self.output_bytes,
            "output_bits":self.output_bits,
            "dither_seed":self.dither_seed,
            "encoded_peak":self.encoded_peak,
            "encoding_attenuation_db":self.encoding_attenuation_db,
            "intermediate_files_written":False,
        }


def _distinct(source,destination):
    if source.resolve()==destination.resolve():
        raise AudioFormatError("Postproduction source and destination must differ")
    if destination.exists():
        try:
            if os.path.samefile(source,destination):
                raise AudioFormatError("Postproduction paths share a file identity")
        except FileNotFoundError:
            pass


def export_postproduction(source,destination,*,plan,output_bits=None,
        dither_seed=None,encoded_peak_policy="attenuate",packet_frames=8192,
        expected_source_sha256=None,expected_source_pcm_sha256=None,
        expected_output_pcm_sha256=None,expected_output_sha256=None,
        overwrite=False,create_parent=False):
    from ..atomic_cli_wav import publish_generated_wav
    if not isinstance(plan,PostProductionPlan):
        raise AudioContractError("Expected PostProductionPlan")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Invalid postproduction publication flags")
    for pin in (expected_source_sha256,expected_source_pcm_sha256,
                expected_output_pcm_sha256,expected_output_sha256):
        if pin is not None:
            strict_sha256(pin)
    if output_bits is not None:
        QuantizationSettings(bits=output_bits)
    if dither_seed is not None:
        QuantizationSettings(dither=True,seed=dither_seed)
    if encoded_peak_policy not in ENCODED_PEAK_POLICIES:
        raise AudioContractError("Invalid encoded peak policy")
    source,destination=Path(source),Path(destination)
    if destination.suffix.lower()!=".wav":
        raise AudioFormatError("Postproduction destination must end in .wav")
    _distinct(source,destination)
    snapshot=read_verified_pcm(source,expected_sha256=expected_source_sha256,
                               packet_frames=packet_frames)
    result=run_postproduction(snapshot.audio,plan,
        expected_source_pcm_sha256=expected_source_pcm_sha256,
        expected_output_pcm_sha256=expected_output_pcm_sha256)
    bits=snapshot.info.bits_per_sample if output_bits is None else output_bits
    quantization=QuantizationSettings(bits=bits,dither=dither_seed is not None,seed=dither_seed)
    if plan.mastering is not None:
        encoded=encode_mastered_audio(result.audio,quantization=quantization,
                                      settings=plan.mastering,policy=encoded_peak_policy)
        wav=encoded.wav
        peak=encoded.measurement.peak.as_dict()
        attenuation=encoded.attenuation_db
    else:
        if np.max(np.abs(result.audio.samples))>1:
            raise AudioContractError("Postproduction output exceeds normalized PCM")
        wav=encode_wav_bytes(result.audio,settings=quantization)
        peak=None
        attenuation=0.
    _distinct(source,destination)
    published=publish_generated_wav(destination,wav,overwrite=overwrite,
        create_parent=create_parent,expected_sha256=expected_output_sha256)
    return PostProductionExport(snapshot.as_dict(),result.manifest(),
        published.wav_sha256,published.bytes_written,bits,dither_seed,peak,attenuation)
