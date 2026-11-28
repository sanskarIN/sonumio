"""Blockwise score rendering must reproduce the original PCM contract."""
from dataclasses import replace
import hashlib
import io
import json
import wave

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan, TRACKS, compose_arrangement, arrangement_digest
from sonumio.audio import SAMPLE_RATE
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.music_blockwise import (
    generate_blockwise_music, iter_stem_blocks, iter_wet_blocks, _Room, _Edges,
    MIN_BLOCK, MAX_BLOCK, ROOM_HISTORY,
)
from sonumio.music_instruments import smooth_edges
from sonumio.music_mix import MixPlan, generate_arrangement_audio, render_stems, room_response
from sonumio.music_score_io import encode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(style="rhythmic", root=57,
        mode="minor", bars=4, bpm=160, seed=31, swing=.23))


@pytest.fixture(scope="module")
def reference(score):
    return render_stems(score)


@pytest.mark.parametrize("block",[256,997,4096,8192,16384])
def test_stem_partition_is_sample_exact(score,reference,block):
    received = {track: [] for track in TRACKS}
    cursor = 0
    for start, stems in iter_stem_blocks(score,block):
        assert cursor == start
        count = min(block,score.plan.frame(score.plan.ticks)-start)
        for track in TRACKS:
            assert stems[track].shape == (2,count)
            assert stems[track].dtype == np.float32
            assert not stems[track].flags.writeable
            received[track].append(stems[track])
        cursor += count
    assert cursor == score.plan.frame(score.plan.ticks)
    for track in TRACKS:
        np.testing.assert_array_equal(np.concatenate(received[track],axis=1),reference[track])


@pytest.mark.parametrize("frames",[2,17,239,240,241,1200,1439,1440,1441,12001])
def test_fade_equivalence_even_for_short_geometry(frames):
    edges = _Edges(frames)
    output = []
    for start in range(0,frames,137):
        data = np.ones((2,min(137,frames-start)),dtype=np.float64)
        edges.apply(start,data)
        output.append(data)
    expected = np.broadcast_to(smooth_edges(frames,240,1200),(2,frames))
    np.testing.assert_array_equal(np.concatenate(output,axis=1),expected)


@pytest.mark.parametrize("block",[1,257,1024,6024,8192])
@pytest.mark.parametrize("amount",[0.,.15,1.])
def test_room_history_matches_whole_array(block,amount):
    rng = np.random.default_rng(7)
    dry = rng.normal(size=(2,14031))*.1
    room = _Room(amount,dry.shape[1])
    chunks = []
    for start in range(0,dry.shape[1],block):
        chunks.append(room.apply(start,dry[:,start:start+block]))
        assert room.history.shape[1] <= ROOM_HISTORY
    np.testing.assert_array_equal(np.concatenate(chunks,axis=1),room_response(dry,amount))


@pytest.mark.parametrize("mix",[
    MixPlan(),
    MixPlan(solo=("lead",),room=0.),
    MixPlan(mute=("pad","kick"),room=1.),
    MixPlan(mute=TRACKS),
    MixPlan(gains=(2.,)*8,peak_ceiling=.05),
])
def test_final_wav_is_identical_to_original(score,mix):
    expected, prior = generate_arrangement_audio(score,mix)
    actual, report = generate_blockwise_music(score,mix,block_frames=997)
    assert actual == expected
    for key in ("score_sha256","wav_sha256","pre_master_peak","master_attenuation",
                "room_amount","peak_ceiling","frames","sample_rate","channels","metrics"):
        assert report[key] == prior[key],key
    for track in TRACKS:
        for key in ("active","gain","dry_peak"):
            assert report["tracks"][track][key] == prior["tracks"][track][key]
        assert report["tracks"][track]["dry_rms"] == pytest.approx(prior["tracks"][track]["dry_rms"],abs=1e-14)
    assert report["replay_verified"]
    assert report["render_passes"] == 2
    assert report["retains_full_float_stems"] is False
    assert report["retains_final_wav_bytes"] is True


def test_partition_changes_receipt_not_wav(score):
    a,ar = generate_blockwise_music(score,block_frames=256)
    b,br = generate_blockwise_music(score,block_frames=MAX_BLOCK)
    assert a == b
    assert ar["wav_sha256"] == br["wav_sha256"] == hashlib.sha256(a).hexdigest()
    assert ar["unscaled_stream_sha256"] != br["unscaled_stream_sha256"]
    with wave.open(io.BytesIO(a),"rb") as reader:
        assert reader.getnchannels() == 2
        assert reader.getframerate() == SAMPLE_RATE
        assert reader.getsampwidth() == 2
        assert reader.getnframes() == score.plan.frame(score.plan.ticks)


@pytest.mark.parametrize("block",[False,0,MIN_BLOCK-1,MAX_BLOCK+1,256.0])
def test_rejects_invalid_block_sizes(score,block):
    with pytest.raises(GenerationError):
        generate_blockwise_music(score,block_frames=block)


def test_second_pass_mismatch_rejected(score,monkeypatch):
    import sonumio.music_blockwise as blockwise
    original = blockwise.iter_wet_blocks
    calls = 0

    def changed(*args,**kwargs):
        nonlocal calls
        calls += 1
        for start,wet in original(*args,**kwargs):
            if calls == 2 and start == 0:
                wet = wet.copy()
                wet[0,0] += .001
            yield start,wet

    monkeypatch.setattr(blockwise,"iter_wet_blocks",changed)
    with pytest.raises(GenerationError,match="replay"):
        generate_blockwise_music(score)


def test_cli_atomic_publication_and_pins(score,tmp_path,capsys):
    source = tmp_path/"score.json"
    source.write_bytes(encode_score(score))
    target = tmp_path/"render.wav"
    args = ["music-render-blockwise",str(source),"--output",str(target),
            "--expected-score-sha256",arrangement_digest(score),"--block-frames","997"]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    content = target.read_bytes()
    assert report["output_sha256"] == hashlib.sha256(content).hexdigest()
    assert report["bytes_written"] == len(content)
    assert main(args) == 2
    assert target.read_bytes() == content
    assert main(args+["--overwrite","--expected-output-sha256","0"*64]) == 2
    assert target.read_bytes() == content
    assert main(args+["--overwrite","--expected-output-sha256",report["output_sha256"]]) == 0
    assert target.read_bytes() == content


def test_cli_does_not_replace_input_alias(score,tmp_path):
    source = tmp_path/"score.wav"
    data = encode_score(score)
    source.write_bytes(data)
    assert main(["music-render-blockwise",str(source),"--output",str(source),"--overwrite"]) == 2
    assert source.read_bytes() == data


def test_long_score_without_whole_song_stem_dependency(monkeypatch):
    import sonumio.music_mix as mixer
    def forbidden(*args,**kwargs):
        raise AssertionError("Full float stems must not be allocated")
    monkeypatch.setattr(mixer,"render_stems",forbidden)
    score = compose_arrangement(ArrangementPlan(bars=24,bpm=120,seed=71))
    wav, report = generate_blockwise_music(score,block_frames=8192)
    assert report["frames"] == 48*SAMPLE_RATE
    assert len(wav) == 44+48*SAMPLE_RATE*4
    assert report["replay_verified"]
