"""Explicit MusicXML-to-original-audio preview through Sonumio's audio core.

A MusicXML file is never treated as an executable program or as a MIDI
synthesizer preset. The validated pitched score is converted into deterministic
finite-Fourier waveforms by the existing opt-in score_preview implementation.
Notational omissions and model-provenance caveats are returned with WAV hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import MidiWorkstationError, sha256
from .musicxml_common import MusicXmlReceipt, fingerprint
from .musicxml_import import decode_musicxml
from .mxl import decode_mxl


@dataclass(frozen=True, slots=True)
class XmlPreviewReceipt:
    source_sha256: str
    score_sha256: str
    wav_sha256: str
    output_frames: int
    output_channels: int
    sample_rate: int
    notes: int
    losses: tuple[str, ...]
    waveform: str
    protocol: str = "sonumio-musicxml-wavetable-preview-v1"

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "source_sha256": self.source_sha256,
            "score_sha256": self.score_sha256,
            "wav_sha256": self.wav_sha256,
            "output_frames": self.output_frames,
            "output_channels": self.output_channels,
            "sample_rate": self.sample_rate,
            "notes": self.notes,
            "losses": list(self.losses),
            "waveform": self.waveform,
            "trained_model_inference": False,
            "external_samples_used": False,
            "source_modified": False,
            "perceptual_quality_established": False,
        }


def export_musicxml_preview(
    payload: bytes, destination: str | Path, *,
    compressed: bool = False, options=None,
    source_path: str | Path | None = None,
) -> tuple[XmlPreviewReceipt, dict]:
    """Render after complete validation; keep the raw source untouched.

    ``compressed`` must be supplied explicitly by the caller. ZIP magic is
    never enough to implicitly select a format with different resource rules.
    """
    if type(compressed) is not bool:
        raise MidiWorkstationError("Compressed MusicXML flag must be boolean")
    source_hash = fingerprint(payload)
    if compressed:
        raw, mxl_receipt = decode_mxl(payload)
    else:
        raw, mxl_receipt = payload, None
    score, import_receipt = decode_musicxml(raw)
    # Late import allows compatibility tests and score inspection to run in
    # source-only environments without the optional audio core installed.
    from .score_preview import ScorePreviewOptions, export_score_preview
    if options is None:
        options = ScorePreviewOptions()
    if not isinstance(options, ScorePreviewOptions):
        raise MidiWorkstationError("MusicXML preview options must use ScorePreviewOptions")
    if score.note_count > options.bridge.maximum_notes:
        raise MidiWorkstationError("MusicXML note count exceeds preview bridge budget")
    rendered = export_score_preview(score, destination, options,
                                    source_path=source_path)
    receipt = XmlPreviewReceipt(
        source_hash, score.sha256, rendered.wav_sha256,
        rendered.output_frames, rendered.output_channels,
        rendered.sample_rate, score.note_count, import_receipt.losses,
        options.bridge.waveform,
    )
    return receipt, {
        "import": import_receipt.as_dict(),
        "package": mxl_receipt.as_dict() if mxl_receipt else None,
        "render": rendered.as_dict(),
    }
