"""Preview contract and deterministic safety boundaries without the full audio core."""
import pytest

from sonumio.midi_workstation.contracts import MidiWorkstationError
from sonumio.midi_workstation.score import Score, ScoreNote
from sonumio.midi_workstation.score_preview import (
    MAX_PREVIEW_FLOAT_BYTES, ScorePreviewOptions, export_score_preview,
)
from sonumio.midi_workstation.wavetable_bridge import WavetableBridgeSettings


@pytest.mark.parametrize("bad", [-1, 0, 7, 15, 17, 25, 64, True, "16", None])
def test_invalid_output_depth_fails_closed(bad):
    with pytest.raises(MidiWorkstationError):
        ScorePreviewOptions(output_bits=bad)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 0.0, "3"])
def test_dither_seed_requires_exact_bounded_integer(seed):
    with pytest.raises(MidiWorkstationError):
        ScorePreviewOptions(dither_seed=seed)


def test_all_integer_pcm_depths_are_supported():
    assert [ScorePreviewOptions(output_bits=bits).output_bits
            for bits in (8, 16, 24, 32)] == [8, 16, 24, 32]


def test_random_seed_within_bounded_range():
    assert ScorePreviewOptions(dither_seed=0).dither_seed == 0
    assert ScorePreviewOptions(dither_seed=2**32 - 1).dither_seed == 2**32 - 1


@pytest.mark.parametrize("flag", ["true", 1, 0, None])
def test_publishing_flags_are_strict_boolean(flag):
    with pytest.raises(MidiWorkstationError):
        ScorePreviewOptions(overwrite=flag)
    with pytest.raises(MidiWorkstationError):
        ScorePreviewOptions(create_parent=flag)


def test_strict_output_hash_validation():
    with pytest.raises(MidiWorkstationError):
        ScorePreviewOptions(expected_output_sha256="f" * 63)
    with pytest.raises(MidiWorkstationError):
        ScorePreviewOptions(expected_output_sha256="F" * 64)
    assert ScorePreviewOptions(expected_output_sha256="f" * 64).expected_output_sha256


def test_bridge_type_must_be_checked():
    with pytest.raises(MidiWorkstationError):
        ScorePreviewOptions(bridge="wavetable")


def test_preview_extension_checked_before_audio_import(tmp_path):
    score = Score.empty(end=480)
    with pytest.raises(MidiWorkstationError, match="must be a .wav"):
        export_score_preview(score, tmp_path / "not-wav.mid")


def test_input_output_identity_refused_before_audio_import(tmp_path):
    score = Score.empty(end=480)
    source = tmp_path / "same.wav"
    source.write_bytes(b"existing input, not touched")
    with pytest.raises(MidiWorkstationError, match="different"):
        export_score_preview(score, source, source_path=source)
    assert source.read_bytes() == b"existing input, not touched"


def test_non_score_rejected_before_audio_import(tmp_path):
    with pytest.raises(MidiWorkstationError):
        export_score_preview(None, tmp_path / "output.wav")


def test_large_preflight_exits_before_audio_import(tmp_path):
    score = Score.empty(end=480 * 200)
    with pytest.raises(MidiWorkstationError, match="float32 PCM budget"):
        export_score_preview(
            score, tmp_path / "large.wav",
            ScorePreviewOptions(bridge=WavetableBridgeSettings(
                sample_rate=192000, channels=2,
            )),
        )
    assert not (tmp_path / "large.wav").exists()


def test_sound_preview_has_explicit_maximum_bytes():
    assert MAX_PREVIEW_FLOAT_BYTES == 24 * 1024 * 1024
