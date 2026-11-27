"""Verified canonical WAV loop search and no-clobber export."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np

from .contracts import AudioSpec
from .errors import AudioContractError, AudioFormatError, AudioLimitError
from .hashing import strict_sha256
from .wave_snapshot import read_verified_pcm, MAX_SNAPSHOT_BYTES
from .loop_analysis import LoopSearchPlan, LoopSearchResult, search_loop
from .loop_render import LoopRenderPlan, LoopRenderReport, render_loop, LAWS
from .dither import QuantizationSettings
from .waveio import encode_wav_bytes


@dataclass(frozen=True, slots=True)
class VerifiedLoopSearch:
    source: dict
    result: LoopSearchResult

    def as_dict(self):
        return {**self.source, "search": self.result.as_dict(),
                "protocol": "sonumio-verified-loop-search-v1"}


@dataclass(frozen=True, slots=True)
class VerifiedLoopExport:
    source: dict
    output_sha256: str
    output_bytes: int
    output_bits: int
    dither_seed: int | None
    render: LoopRenderReport
    search: LoopSearchResult | None = None

    def as_dict(self):
        return {
            **self.source,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
            "output_bits": self.output_bits,
            "dither_seed": self.dither_seed,
            "render": self.render.as_dict(),
            "search": self.search.as_dict() if self.search else None,
            "protocol": "sonumio-verified-loop-export-v1",
        }


def search_wav_loop(source, plan, *, expected_sha256=None, packet_frames=8192):
    if not isinstance(plan, LoopSearchPlan):
        raise AudioContractError("Expected LoopSearchPlan")
    snapshot = read_verified_pcm(source, expected_sha256=expected_sha256,
                                 packet_frames=packet_frames)
    return VerifiedLoopSearch(snapshot.as_dict(), search_loop(snapshot.audio, plan))


def _distinct_paths(source, destination):
    if source.resolve() == destination.resolve():
        raise AudioFormatError("Loop source and destination must differ")
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                raise AudioFormatError("Loop source and destination have the same file identity")
        except FileNotFoundError:
            pass


def export_wav_loop(source, destination, *, plan=None, search=None,
                    law="cosine", repeats=1, peak_ceiling=None,
                    output_bits=None, dither_seed=None, packet_frames=8192,
                    expected_sha256=None, expected_output_sha256=None,
                    overwrite=False, create_parent=False):
    """Use exactly one manual plan or automatic bounded endpoint search.

    The source is fully verified before the optional search and render.
    File publication is intentionally separate from the pure PCM engine.
    No output or parent directory is touched on failed input validation.
    """
    from ..atomic_cli_wav import publish_generated_wav
    if (plan is None) == (search is None):
        raise AudioContractError("Select either manual loop plan or endpoint search")
    if plan is not None and not isinstance(plan, LoopRenderPlan):
        raise AudioContractError("Expected LoopRenderPlan")
    if search is not None and not isinstance(search, LoopSearchPlan):
        raise AudioContractError("Expected LoopSearchPlan")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Loop publication flags must be Boolean")
    if plan is not None and (law != "cosine" or repeats != 1 or peak_ceiling is not None):
        raise AudioContractError("Manual plan already defines law, repeats and peak ceiling")
    if search is not None:
        # Validate automatic render settings before opening the source.
        LoopRenderPlan(0, 2, law=law, repeats=repeats, peak_ceiling=peak_ceiling)
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    if output_bits is not None:
        QuantizationSettings(bits=output_bits)
    if dither_seed is not None:
        QuantizationSettings(dither=True, seed=dither_seed)
    source, destination = Path(source), Path(destination)
    if destination.suffix.lower() != ".wav":
        raise AudioFormatError("Loop destination must end in .wav")
    _distinct_paths(source, destination)
    snapshot = read_verified_pcm(source, expected_sha256=expected_sha256,
                                 packet_frames=packet_frames)
    selected = None
    if search is not None:
        selected = search_loop(snapshot.audio, search)
        plan = LoopRenderPlan.from_candidate(selected.best, law=law, repeats=repeats,
                                             peak_ceiling=peak_ceiling)
    spec = AudioSpec(snapshot.audio.sample_rate, snapshot.audio.spec.layout,
                     plan.period_frames*plan.repeats)
    if spec.byte_count > MAX_SNAPSHOT_BYTES:
        raise AudioLimitError("Loop export exceeds offline PCM byte budget")
    result = render_loop(snapshot.audio, plan)
    if np.max(np.abs(result.audio.samples)) > 1:
        raise AudioContractError("Loop exceeds normalized PCM; specify a peak ceiling")
    bits = snapshot.info.bits_per_sample if output_bits is None else output_bits
    wav = encode_wav_bytes(result.audio, settings=QuantizationSettings(
        bits=bits, dither=dither_seed is not None, seed=dither_seed,
    ))
    _distinct_paths(source, destination)
    published = publish_generated_wav(
        destination, wav, overwrite=overwrite, create_parent=create_parent,
        expected_sha256=expected_output_sha256,
    )
    return VerifiedLoopExport(
        snapshot.as_dict(), published.wav_sha256, published.bytes_written,
        bits, dither_seed, result.report, selected,
    )
