"""CLI accepts hash-pinned recipes; no unsafe path replacement or hidden outputs."""
from dataclasses import replace
import json

import pytest

from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.score import Score, ScoreTrack, ScoreNote
from sonumio.midi_workstation.score_io import encode_score
from sonumio.midi_workstation.multitrack_preview import TrackPatch, MultiTrackPreviewSettings
from sonumio.midi_workstation.multitrack_recipe import MultitrackRecipe


def fixtures(tmp_path):
    empty=Score.empty(end=960)
    s=replace(empty, tracks=(ScoreTrack('melody','Melody',0,0,0,0,
                                     (ScoreNote('n',0,480,69),)),))
    score=tmp_path/'score.json'
    score.write_bytes(encode_score(s))
    recipe=MultitrackRecipe(s.sha256,MultiTrackPreviewSettings(
        (TrackPatch('melody','sine',.05),), sample_rate=8000, channels=2,
    ))
    path=tmp_path/'recipe.json'
    path.write_bytes(recipe.to_bytes())
    return score,path,recipe


def command(score,recipe,output):
    return ['multitrack-preview-from-recipe',str(score),
            '--recipe',str(recipe),'--output',str(output)]


def test_recipe_inspection_is_validated_and_does_not_render(tmp_path,capsys):
    score,path,recipe=fixtures(tmp_path)
    assert main(['multitrack-recipe-inspect',str(path),
                 '--expected-sha256',recipe.sha256])==0
    body=json.loads(capsys.readouterr().out)
    assert body['patches']==1
    assert body['score_sha256']==recipe.score_sha256
    assert body['audio_rendered'] is False
    assert not (tmp_path/'out.wav').exists()


def test_inspection_rejects_wrong_expected_sha(tmp_path,capsys):
    _,recipe,_=fixtures(tmp_path)
    assert main(['multitrack-recipe-inspect',str(recipe),
                 '--expected-sha256','f'*64])==2
    assert capsys.readouterr().out==''


def test_mismatched_score_digest_aborts_without_output(tmp_path,capsys):
    source,patches,_=fixtures(tmp_path)
    other=tmp_path/'different.json'
    other.write_bytes(encode_score(Score.empty(end=480)))
    output=tmp_path/'render.wav'
    assert main(command(other,patches,output))==2
    assert not output.exists()
    assert capsys.readouterr().out==''


def test_cli_passes_protected_recipe_path_to_publisher(tmp_path,capsys,monkeypatch):
    source,patches,recipe=fixtures(tmp_path)
    output=tmp_path/'mix.wav'
    called={}
    def fake_export(score,path,settings,**keywords):
        called.update(path=path,settings=settings,keywords=keywords,
                      digest=score.sha256)
        return {'wav_sha256':'e'*64}
    monkeypatch.setattr('sonumio.midi_workstation.multitrack_recipe_cli.export_multitrack_preview',fake_export)
    assert main(command(source,patches,output)+['--expected-recipe-sha256',recipe.sha256])==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['preview']['wav_sha256']=='e'*64
    assert called['digest']==recipe.score_sha256
    assert called['settings']==recipe.settings
    assert called['keywords']['protected_paths']==(patches,)
    assert called['keywords']['overwrite'] is False
    assert called['keywords']['source_path']==source
    assert not output.exists()  # fake publisher proves dispatch only


def test_cli_rejects_destination_equal_to_recipe(tmp_path,capsys,monkeypatch):
    source,path,_=fixtures(tmp_path)
    old=path.read_bytes()
    output=path
    assert main(command(source,path,output))==2
    assert path.read_bytes()==old
    assert capsys.readouterr().out==''


def test_cli_rejects_destination_hardlinked_to_recipe(tmp_path,capsys):
    source,path,_=fixtures(tmp_path)
    alias=tmp_path/'alias.wav'
    alias.hardlink_to(path)
    assert main(command(source,path,alias)+['--overwrite'])==2
    assert alias.read_bytes()==path.read_bytes()
    assert capsys.readouterr().out==''


def test_cli_rejects_input_file_symlink(tmp_path,capsys):
    source,path,_=fixtures(tmp_path)
    alias=tmp_path/'alias.json'
    alias.symlink_to(path)
    assert main(command(source,alias,tmp_path/'mix.wav'))==2
    assert capsys.readouterr().out==''


def test_cli_rejects_nonfinite_dither_or_unknown_score(tmp_path,capsys):
    source,path,_=fixtures(tmp_path)
    target=tmp_path/'mix.wav'
    assert main(command(source,path,target)+['--dither-seed','-1'])==2
    assert not target.exists()
    assert capsys.readouterr().out==''
    assert main(command(tmp_path/'missing.json',path,target))==2
    assert not target.exists()


def test_cli_does_not_accept_invalid_recipe_protocol(tmp_path,capsys):
    source,path,_=fixtures(tmp_path)
    data=json.loads(path.read_text())
    data['protocol']='evil'
    path.write_text(json.dumps(data))
    assert main(command(source,path,tmp_path/'out.wav'))==2
    assert capsys.readouterr().out==''
