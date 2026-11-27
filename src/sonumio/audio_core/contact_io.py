"""Strict local contact-scene JSON and verified original WAV publication."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import stat
import numpy as np

from .errors import AudioContractError, AudioFormatError, AudioLimitError
from .hashing import parse_canonical_json, digest_audio, strict_sha256
from .modal_synthesis import (
    ContactScene, ContactEvent, Resonator, ResonanceMode, ContactSceneReport,
    material_resonator, render_contact_scene,
)
from .contact_patterns import ContactPattern, pattern_scene
from .waveio import encode_wav_bytes
from .dither import QuantizationSettings

MAX_JSON_BYTES=256*1024
MAX_EXPORT_BYTES=24*1024*1024


def _object(value,allowed,name):
    if type(value) is not dict or not set(value)<=set(allowed):
        raise AudioContractError("Unexpected fields in "+name)
    return dict(value)


def _resonator(value):
    data=_object(value,("material","fundamental_hz","size","damping","modes",
                        "contact_noise","noise_decay_seconds","noise_color"),"resonator")
    if "material" in data:
        if not set(data)<={"material","fundamental_hz","size","damping"}:
            raise AudioContractError("Preset and custom modes cannot be combined")
        return material_resonator(**data)
    if any(k in data for k in ("fundamental_hz","size","damping")):
        raise AudioContractError("Preset parameters require a material")
    modes=data.get("modes")
    if type(modes) is not list or not 1<=len(modes)<=32:
        raise AudioContractError("Expected bounded custom mode array")
    data["modes"]=tuple(ResonanceMode(**_object(m,ResonanceMode.__dataclass_fields__,"mode"))
                         for m in modes)
    return Resonator(**data)


def parse_contact_scene(raw):
    data=_object(parse_canonical_json(raw,max_bytes=MAX_JSON_BYTES),
                 ("sample_rate","frames","channels","peak_ceiling","events","pattern"),"contact scene")
    if "pattern" in data:
        if set(data)!={"pattern"}:
            raise AudioContractError("Contact pattern is an alternative to an explicit scene")
        return pattern_scene(ContactPattern(**_object(data["pattern"],ContactPattern.__dataclass_fields__,"pattern")))
    events=data.get("events")
    if type(events) is not list or not 1<=len(events)<=512:
        raise AudioContractError("Expected bounded contact event array")
    parsed=[]
    for value in events:
        event=_object(value,ContactEvent.__dataclass_fields__,"contact event")
        if "resonator" not in event:
            raise AudioContractError("Contact event requires resonator")
        event["resonator"]=_resonator(event["resonator"])
        parsed.append(ContactEvent(**event))
    data["events"]=tuple(parsed)
    return ContactScene(**data)


def load_contact_scene(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise AudioFormatError("Expected ordinary local contact JSON file")
    flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0)|getattr(os,"O_BINARY",0)
    fd=os.open(path,flags)
    with os.fdopen(fd,"rb") as f:
        info=os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise AudioFormatError("Expected regular contact JSON descriptor")
        if info.st_size>MAX_JSON_BYTES:
            raise AudioLimitError("Contact JSON exceeds byte budget")
        raw=f.read(MAX_JSON_BYTES+1)
    return parse_contact_scene(raw)


@dataclass(frozen=True, slots=True)
class ContactExport:
    output_sha256: str
    output_pcm_sha256: str
    output_bytes: int
    output_bits: int
    dither_seed: int | None
    render: ContactSceneReport
    protocol: str = "sonumio-contact-export-v1"

    def as_dict(self):
        return {
            "protocol":self.protocol,
            "output_sha256":self.output_sha256,
            "pre_encoding_pcm_sha256":self.output_pcm_sha256,
            "output_bytes":self.output_bytes,
            "output_bits":self.output_bits,
            "dither_seed":self.dither_seed,
            "render":self.render.as_dict(),
            "model_inference":False,
        }


def export_contact_scene(scene,destination,*,output_bits=16,dither_seed=None,
                         expected_output_sha256=None,overwrite=False,create_parent=False):
    from ..atomic_cli_wav import publish_generated_wav
    if not isinstance(scene,ContactScene):
        raise AudioContractError("Expected ContactScene")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Invalid contact publication flags")
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    destination=Path(destination)
    if destination.suffix.lower()!=".wav":
        raise AudioFormatError("Contact output must end in .wav")
    settings=QuantizationSettings(bits=output_bits,dither=dither_seed is not None,seed=dither_seed)
    if scene.frames*scene.channels*4>MAX_EXPORT_BYTES:
        raise AudioLimitError("Contact export exceeds offline PCM budget")
    result=render_contact_scene(scene)
    if np.max(np.abs(result.audio.samples))>1:
        raise AudioContractError("Contact output exceeds normalized PCM")
    encoded=encode_wav_bytes(result.audio,settings=settings)
    published=publish_generated_wav(destination,encoded,overwrite=overwrite,
        create_parent=create_parent,expected_sha256=expected_output_sha256)
    return ContactExport(published.wav_sha256,digest_audio(result.audio),
                         published.bytes_written,output_bits,dither_seed,result.report)
