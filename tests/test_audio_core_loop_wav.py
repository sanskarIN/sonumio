"""Verified snapshot and loop export regression tests using synthetic PCM."""
import hashlib
import json
import os
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.waveio import encode_wav_bytes, read_wav_file
from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError, AudioLimitError
from sonumio.audio_core.wave_snapshot import read_verified_pcm
from sonumio.audio_core.loop_analysis import LoopSearchPlan
from sonumio.audio_core.loop_render import LoopRenderPlan
from sonumio.audio_core.loop_wav import search_wav_loop, export_wav_loop
from sonumio.audio_core.loop_cli import main


@pytest.fixture
def source(tmp_path):
    t = np.arange(4097)
    x = (.2*np.sin(2*np.pi*t/64)).astype(np.float32)
    data = encode_wav_bytes(make_buffer(np.stack((x,-x)), 24000),
                            settings=QuantizationSettings(bits=24))
    path = tmp_path/"source.wav"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("size", [1, 17, 128, 8192])
def test_verified_snapshot_matches_decoder(source, size):
    path, pin = source
    snapshot = read_verified_pcm(path, expected_sha256=pin, packet_frames=size)
    np.testing.assert_array_equal(snapshot.audio.samples, read_wav_file(path).samples)
    assert snapshot.sha256 == pin
    assert snapshot.info.frames == 4097
    assert snapshot.as_dict()["source_verified"]


def test_snapshot_budget_and_digest(source):
    path, _ = source
    with pytest.raises(AudioLimitError):
        read_verified_pcm(path, maximum_pcm_bytes=8)
    with pytest.raises(AudioIntegrityError):
        read_verified_pcm(path, expected_sha256="0"*64)


def test_search_and_export_are_consistent(source, tmp_path):
    path, pin = source
    plan = LoopSearchPlan(128, 1111, start_radius=4, stop_radius=32,
                          context_frames=64, blend_frames=32)
    found = search_wav_loop(path, plan, expected_sha256=pin)
    destination = tmp_path/"auto.wav"
    receipt = export_wav_loop(path, destination, search=plan, repeats=3,
                              expected_sha256=pin)
    assert receipt.search == found.result
    assert receipt.render.plan.start_frame == found.result.best.start_frame
    assert receipt.render.plan.stop_frame == found.result.best.stop_frame
    assert receipt.render.output_frames == 3*found.result.best.period_frames
    assert receipt.output_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert receipt.source["source_sha256"] == pin
    audio = read_wav_file(destination)
    n = receipt.render.period_frames
    for i in (1, 2):
        np.testing.assert_array_equal(audio.samples[:, :n], audio.samples[:, i*n:(i+1)*n])
    json.dumps(receipt.as_dict(), allow_nan=False)


@pytest.mark.parametrize("bits", [8,16,24,32])
def test_dither_and_reader_packet_partition(source, tmp_path, bits):
    path, pin = source
    plan = LoopRenderPlan(100, 1111, 32, repeats=2)
    a = export_wav_loop(path, tmp_path/"a.wav", plan=plan, output_bits=bits,
                        dither_seed=32, packet_frames=13)
    b = export_wav_loop(path, tmp_path/"b.wav", plan=plan, output_bits=bits,
                        dither_seed=32, packet_frames=8192,
                        expected_output_sha256=a.output_sha256)
    assert a.output_sha256 == b.output_sha256
    assert (tmp_path/"a.wav").read_bytes() == (tmp_path/"b.wav").read_bytes()


def test_invalid_pin_does_not_create_parent(source, tmp_path):
    path, _ = source
    output = tmp_path/"missing"/"result.wav"
    with pytest.raises(AudioIntegrityError):
        export_wav_loop(path, output, plan=LoopRenderPlan(0, 100),
                        expected_sha256="0"*64, create_parent=True)
    assert not output.parent.exists()


def test_no_clobber_and_output_digest(source, tmp_path):
    path, pin = source
    output = tmp_path/"existing.wav"
    output.write_bytes(b"old")
    plan = LoopRenderPlan(100, 1100, 32)
    with pytest.raises(AudioContractError):
        export_wav_loop(path, output, plan=plan)
    with pytest.raises(AudioIntegrityError):
        export_wav_loop(path, output, plan=plan, overwrite=True,
                        expected_output_sha256="0"*64)
    assert output.read_bytes() == b"old"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin


def test_source_alias_rejected(source, tmp_path):
    path, pin = source
    plan = LoopRenderPlan(100, 1100)
    with pytest.raises(AudioContractError, match="must differ"):
        export_wav_loop(path, path, plan=plan, overwrite=True)
    alias = tmp_path/"alias.wav"
    try:
        os.link(path, alias)
    except OSError:
        pytest.skip("hard links unavailable")
    with pytest.raises(AudioContractError, match="file identity"):
        export_wav_loop(path, alias, plan=plan, overwrite=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin


def test_reject_overrange_and_allow_attenuation(tmp_path):
    source = tmp_path/"constant.wav"
    source.write_bytes(encode_wav_bytes(make_buffer(
        np.full(128, .9, dtype=np.float32), 8000)))
    output = tmp_path/"out.wav"
    with pytest.raises(AudioContractError, match="peak ceiling"):
        export_wav_loop(source, output, plan=LoopRenderPlan(0, 128, 32, "equal_power"))
    assert not output.exists()
    r = export_wav_loop(source, output,
        plan=LoopRenderPlan(0, 128, 32, "equal_power", peak_ceiling=.9))
    assert r.render.attenuation < 1


@pytest.mark.parametrize("kwargs", [
    {}, {"plan": object()}, {"search": object()},
    {"plan": LoopRenderPlan(0, 10), "search": LoopSearchPlan(0, 10)},
    {"plan": LoopRenderPlan(0, 10), "repeats": 2},
    {"search": LoopSearchPlan(0, 10), "law": "unknown"},
])
def test_options_validated_before_read(kwargs, tmp_path):
    with pytest.raises(AudioContractError):
        export_wav_loop(tmp_path/"missing.wav", tmp_path/"out.wav", **kwargs)


def test_cli_search_and_auto_render(source, tmp_path, capsys):
    path, pin = source
    common = [str(path), "--start-frame", "100", "--stop-frame", "1000",
              "--context-frames", "64", "--start-radius", "4",
              "--stop-radius", "16", "--blend-frames", "32",
              "--expected-sha256", pin]
    assert main(["loop-search", *common]) == 0
    result = json.loads(capsys.readouterr().out)
    best = result["search"]["candidates"][0]
    output = tmp_path/"loop.wav"
    assert main(["loop-render", *common, "--auto-search", "--output", str(output)]) == 0
    export = json.loads(capsys.readouterr().out)
    assert export["render"]["period_frames"] == best["period_frames"]


def test_cli_reports_failed_integrity(source, tmp_path, capsys):
    path, _ = source
    output = tmp_path/"bad.wav"
    assert main(["loop-render", str(path), "--start-frame", "0",
        "--stop-frame", "100", "--output", str(output),
        "--expected-sha256", "0"*64]) == 2
    assert "SHA256" in capsys.readouterr().err
    assert not output.exists()
