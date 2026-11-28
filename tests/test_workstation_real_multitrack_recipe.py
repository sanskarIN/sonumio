"""Live Sonumio audio-core tests of checksum-bound multitrack recipes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

pytest.importorskip('sonumio.audio_core.wavetable')
from sonumio.audio_core.waveio import read_wav_file
from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.multitrack_preview import (
    MultiTrackPreviewSettings, TrackPatch,
)
from sonumio.midi_workstation.multitrack_recipe import MultitrackRecipe
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack
from sonumio.midi_workstation.score_io import encode_score


def fixture(tmp_path):
    original=Score.empty(end=960)
    score=replace(original,tracks=(
        ScoreTrack('lead','Lead',0,80,0,0,
                   (ScoreNote('lead-note',0,480,69),)),
        ScoreTrack('pad','Pad',1,89,0,0,
                   (ScoreNote('pad-note',480,480,60),)),
    ))
    recipe=MultitrackRecipe(score.sha256,MultiTrackPreviewSettings(
        (TrackPatch('lead','sine',.09,attack_frames=0,decay_frames=0,
                    sustain_level=1.,release_frames=0),
         TrackPatch('pad','triangle',.07,attack_frames=0,decay_frames=0,
                    sustain_level=1.,release_frames=0)),
        sample_rate=8000,channels=2,
    ))
    source=tmp_path/'score.json'
    instrument=tmp_path/'patches.json'
    source.write_bytes(encode_score(score))
    instrument.write_bytes(recipe.to_bytes())
    return source,instrument,recipe


def argv(source,recipe,destination):
    return ['multitrack-preview-from-recipe',str(source),
            '--recipe',str(recipe),'--output',str(destination)]


def test_real_pinned_preview_is_deterministic_and_stereo(tmp_path,capsys):
    source,patches,recipe=fixture(tmp_path)
    first=tmp_path/'one.wav'
    second=tmp_path/'two.wav'
    assert main(argv(source,patches,first))==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['recipe']['canonical_sha256']==recipe.sha256
    assert receipt['preview']['render']['rendered_notes']==2
    assert receipt['preview']['render']['tracks_rendered']==2
    assert receipt['preview']['render']['trained_model_inference'] is False
    wav=read_wav_file(first)
    assert (wav.sample_rate,wav.frames,wav.channels)==(8000,8000,2)
    assert main(argv(source,patches,second))==0
    capsys.readouterr()
    assert first.read_bytes()==second.read_bytes()
    assert receipt['preview']['wav_sha256']==hashlib.sha256(first.read_bytes()).hexdigest()


def test_wrong_pinned_recipe_digest_fails_before_publication(tmp_path,capsys):
    source,patches,recipe=fixture(tmp_path)
    out=tmp_path/'no.wav'
    assert main(argv(source,patches,out)+['--expected-recipe-sha256','0'*64])==2
    assert not out.exists()
    assert capsys.readouterr().out==''


def test_wrong_expected_output_hash_refuses_to_commit(tmp_path,capsys):
    source,patches,_=fixture(tmp_path)
    out=tmp_path/'no.wav'
    assert main(argv(source,patches,out)+['--expected-output-sha256','f'*64])==2
    assert not out.exists()
    assert capsys.readouterr().out==''


def test_recipe_and_output_hardlink_cannot_clobber_source(tmp_path,capsys):
    source,patches,_=fixture(tmp_path)
    out=tmp_path/'recipe-output.wav'
    out.hardlink_to(patches)
    original=patches.read_bytes()
    assert main(argv(source,patches,out)+['--overwrite'])==2
    assert out.read_bytes()==original
    assert patches.read_bytes()==original
    assert capsys.readouterr().out==''


def test_source_score_and_output_hardlink_cannot_clobber_source(tmp_path,capsys):
    source,patches,_=fixture(tmp_path)
    out=tmp_path/'score-output.wav'
    out.hardlink_to(source)
    original=source.read_bytes()
    assert main(argv(source,patches,out)+['--overwrite'])==2
    assert out.read_bytes()==original
    assert source.read_bytes()==original
    assert capsys.readouterr().out==''


def test_existing_destination_refused_without_overwrite(tmp_path,capsys):
    source,patches,_=fixture(tmp_path)
    out=tmp_path/'mix.wav'
    out.write_bytes(b'old destination')
    assert main(argv(source,patches,out))==2
    assert out.read_bytes()==b'old destination'
    assert capsys.readouterr().out==''


def test_stale_source_score_rejected_before_render(tmp_path,capsys):
    source,patches,_=fixture(tmp_path)
    old=source.read_bytes()
    score=Score.empty(end=480)
    source.write_bytes(encode_score(score))
    out=tmp_path/'no.wav'
    assert main(argv(source,patches,out))==2
    assert not out.exists()
    assert source.read_bytes()!=old
    assert capsys.readouterr().out==''


def test_invalid_dither_seed_does_not_create_output(tmp_path,capsys):
    source,patches,_=fixture(tmp_path)
    out=tmp_path/'no.wav'
    assert main(argv(source,patches,out)+['--dither-seed','-1'])==2
    assert not out.exists()
    assert capsys.readouterr().out==''
