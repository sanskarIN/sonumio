"""Dry-run CLI can produce resource estimates in a pure workstation install."""
from dataclasses import replace
import json

import pytest

from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.contracts import MidiWorkstationError
from sonumio.midi_workstation.multitrack_preview import TrackPatch,MultiTrackPreviewSettings
from sonumio.midi_workstation.multitrack_recipe import MultitrackRecipe
from sonumio.midi_workstation.score import Score,ScoreNote,ScoreTrack
from sonumio.midi_workstation.score_io import encode_score


def artifacts(tmp_path):
    score=replace(Score.empty(end=960),tracks=(ScoreTrack(
        'melody','Melody',0,0,0,0,(ScoreNote('a',0,480,69),)),))
    source=tmp_path/'original.json'
    source.write_bytes(encode_score(score))
    manifest=MultitrackRecipe(score.sha256,MultiTrackPreviewSettings(
        (TrackPatch('melody','sine',.1,release_frames=0),),sample_rate=8000,
    ))
    recipe=tmp_path/'patches.json'
    recipe.write_bytes(manifest.to_bytes())
    return source,recipe,score,manifest


def test_dry_run_cli_works_without_audio_engine(tmp_path,capsys):
    source,recipe,score,manifest=artifacts(tmp_path)
    assert main(['multitrack-preview-plan',str(source),
                 '--patch','melody:sine:.1','--sample-rate','8000'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['plan']['source_score_sha256']==score.sha256
    assert result['plan']['rendered_notes']==1
    assert result['plan']['audio_rendered'] is False
    assert result['plan']['output_frames']>0
    assert not (tmp_path/'output.wav').exists()


def test_recipe_dry_run_matches_manual_same_settings(tmp_path,capsys):
    source,recipe,score,manifest=artifacts(tmp_path)
    assert main(['multitrack-plan-from-recipe',str(source),
                 '--recipe',str(recipe),
                 '--expected-recipe-sha256',manifest.sha256])==0
    result=json.loads(capsys.readouterr().out)
    assert result['recipe']['canonical_sha256']==manifest.sha256
    assert result['plan']['rendered_notes']==1
    assert result['plan']['stems'][0]['waveform']=='sine'
    assert result['plan']['work_samples']>0


def test_pinned_score_rejects_different_content(tmp_path,capsys):
    source,recipe,score,manifest=artifacts(tmp_path)
    changed=tmp_path/'changed.json'
    changed.write_bytes(encode_score(Score.empty(end=480)))
    assert main(['multitrack-plan-from-recipe',str(changed),
                 '--recipe',str(recipe)])==2
    assert capsys.readouterr().out==''


def test_wrong_file_digest_rejected(tmp_path,capsys):
    source,recipe,score,manifest=artifacts(tmp_path)
    assert main(['multitrack-plan-from-recipe',str(source),
                 '--recipe',str(recipe),'--expected-recipe-sha256','0'*64])==2
    assert capsys.readouterr().out==''


def test_missing_instrument_mapping_aborts(tmp_path,capsys):
    source,recipe,score,manifest=artifacts(tmp_path)
    assert main(['multitrack-preview-plan',str(source),
                 '--patch','elsewhere:saw:.1','--sample-rate','8000'])==2
    assert capsys.readouterr().out==''


def test_bad_patch_syntax_aborts_with_no_success_receipt(tmp_path,capsys):
    source,recipe,score,manifest=artifacts(tmp_path)
    assert main(['multitrack-preview-plan',str(source),
                 '--patch','../invalid:broken:.1'])==2
    assert capsys.readouterr().out==''
