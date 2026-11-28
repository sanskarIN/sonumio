"""MusicXML-to-existing-PCM pipeline tests without synthesizer installation."""
from dataclasses import dataclass
from hashlib import sha256
import json
import sys
from types import ModuleType

import pytest

from sonumio.midi_workstation.contracts import MidiWorkstationError
from sonumio.midi_workstation.musicxml_export import encode_musicxml
from sonumio.midi_workstation.musicxml_preview import export_musicxml_preview
from sonumio.midi_workstation.mxl import encode_mxl
from sonumio.midi_workstation.score import Score, ScoreNote
from sonumio.midi_workstation.musicxml_cli import main


@dataclass(frozen=True)
class FakeBridge:
    sample_rate: int = 24000
    channels: int = 2
    waveform: str = "triangle"
    master_gain: float = .15
    maximum_notes: int = 2048

    def __post_init__(self):
        if self.maximum_notes < 1 or self.sample_rate < 8000:
            raise MidiWorkstationError("Invalid fake audio bridge")


@dataclass(frozen=True)
class FakeOptions:
    bridge: FakeBridge = FakeBridge()
    output_bits: int = 16
    dither_seed: int | None = None
    overwrite: bool = False
    create_parent: bool = False
    expected_output_sha256: str | None = None


class FakeRenderReport:
    wav_sha256 = "a"*64
    output_frames = 24000
    output_channels = 2
    sample_rate = 24000

    def as_dict(self):
        return {"protocol":"mock-render", "wav_sha256":self.wav_sha256,
                "output_frames":self.output_frames, "model_inference":False}


@pytest.fixture
def mock_preview(monkeypatch):
    module = ModuleType("sonumio.midi_workstation.score_preview")
    module.ScorePreviewOptions = FakeOptions
    recorded = []

    def fake_export(score, destination, options, *, source_path):
        recorded.append((score, destination, options, source_path))
        return FakeRenderReport()

    module.export_score_preview = fake_export
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.recorded = recorded
    return module


def test_incoming_xml_passes_validated_score_to_existing_engine(mock_preview):
    raw, _ = encode_musicxml(Score.empty())
    receipt, information = export_musicxml_preview(raw, "output.wav",
        options=FakeOptions(), source_path="input.musicxml")
    assert receipt.source_sha256 == sha256(raw).hexdigest()
    assert receipt.notes == 0
    assert receipt.waveform == "triangle"
    assert receipt.wav_sha256 == "a"*64
    assert information["render"]["model_inference"] is False
    assert mock_preview.recorded[0][0].note_count == 0
    assert mock_preview.recorded[0][1] == "output.wav"
    assert mock_preview.recorded[0][3] == "input.musicxml"


def test_compressed_xml_routes_through_archive_validator(mock_preview):
    raw, _ = encode_musicxml(Score.empty())
    archive, _ = encode_mxl(raw)
    receipt, information = export_musicxml_preview(archive, "music.wav",
        compressed=True, options=FakeOptions(),source_path="input.mxl")
    assert receipt.source_sha256 == sha256(archive).hexdigest()
    assert information["package"]["compressed"] is True
    assert len(mock_preview.recorded) == 1


def test_bad_compressed_flag_is_not_treated_as_truthy(mock_preview):
    raw, _ = encode_musicxml(Score.empty())
    with pytest.raises(MidiWorkstationError):
        export_musicxml_preview(raw, "result.wav", compressed=1)
    assert not mock_preview.recorded


def test_fake_options_cannot_cross_real_type_boundary(mock_preview):
    raw, _ = encode_musicxml(Score.empty())
    with pytest.raises(MidiWorkstationError):
        export_musicxml_preview(raw,"result.wav",options=object())
    assert not mock_preview.recorded


def test_no_synthesis_attempt_for_invalid_xml(mock_preview):
    with pytest.raises(MidiWorkstationError):
        export_musicxml_preview(b"<score-partwise version='4.0'>", "result.wav")
    assert not mock_preview.recorded


def test_real_audio_core_is_not_implicitly_required_by_inspection(mock_preview):
    raw, _ = encode_musicxml(Score.empty())
    receipt, _ = export_musicxml_preview(raw,"result.wav",options=FakeOptions())
    assert receipt.score_sha256 == Score.empty().sha256


def test_preview_can_enforce_user_note_limit_before_render(mock_preview):
    s = Score.empty()
    original = ScoreNote("note", 0, 480, 60)
    from dataclasses import replace
    score = replace(s,tracks=(replace(s.tracks[0],notes=(original,)),))
    raw,_=encode_musicxml(score)
    options=FakeOptions(FakeBridge(maximum_notes=1))
    assert export_musicxml_preview(raw,"result.wav",options=options)[0].notes==1
    assert len(mock_preview.recorded)==1


@pytest.mark.parametrize("fake_data", [b"", b"PK\x03\x04"*100, b"random"*200])
def test_bad_archive_fails_before_renderer(fake_data,mock_preview):
    with pytest.raises(MidiWorkstationError):
        export_musicxml_preview(fake_data,"result.wav",compressed=True)
    assert not mock_preview.recorded


def test_no_output_reached_on_wrong_format_cli(tmp_path,capsys,mock_preview):
    source = tmp_path / "source.musicxml"
    source.write_bytes(encode_musicxml(Score.empty())[0])
    output = tmp_path / "out.wav"
    assert main(["preview-mxl", str(source),"--output",str(output)]) == 2
    assert not mock_preview.recorded
    assert not output.exists()
    assert capsys.readouterr().out == ""


def test_no_output_reached_on_sha_mismatch_cli(tmp_path,capsys,mock_preview):
    source = tmp_path / "source.musicxml"
    source.write_bytes(encode_musicxml(Score.empty())[0])
    output = tmp_path / "out.wav"
    assert main(["preview-xml",str(source),"--output",str(output),
                 "--expected-sha256","0"*64]) == 2
    assert not mock_preview.recorded
    assert not output.exists()
    assert capsys.readouterr().out == ""


def test_default_options_selected_when_omitted(mock_preview):
    raw,_ = encode_musicxml(Score.empty())
    receipt,_ = export_musicxml_preview(raw,"result.wav")
    assert mock_preview.recorded[0][2] == FakeOptions()
    assert receipt.waveform == "triangle"


def test_receipt_does_not_claim_neural_inference_or_quality(mock_preview):
    raw,_ = encode_musicxml(Score.empty())
    receipt,_=export_musicxml_preview(raw,"result.wav")
    content=receipt.as_dict()
    assert content["trained_model_inference"] is False
    assert content["perceptual_quality_established"] is False
    assert content["source_modified"] is False
    assert content["external_samples_used"] is False
    assert content["losses"]
