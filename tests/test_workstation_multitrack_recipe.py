"""Strict deterministic recipe interoperability, no audio-core installation needed."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from sonumio.midi_workstation.contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
)
from sonumio.midi_workstation.multitrack_preview import (
    MultiTrackPreviewSettings, TrackPatch,
)
from sonumio.midi_workstation.multitrack_recipe import (
    RECIPE_PROTOCOL, MAX_RECIPE_BYTES, MultitrackRecipe,
    decode_multitrack_recipe, load_multitrack_recipe,
)
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack


def score():
    blank = Score.empty(end=960)
    return replace(blank, tracks=(ScoreTrack(
        'piano', 'Piano', 0, 1, 0, 0,
        (ScoreNote('a', 0, 480, 69),),
    ),))


def recipe():
    return MultitrackRecipe(score().sha256, MultiTrackPreviewSettings(
        patches=(TrackPatch('piano', 'sine', .05),),
        sample_rate=8000, channels=2, normalization='peak',
    ))


def to_dict(r):
    return json.loads(r.to_bytes())


def reencode(value):
    return json.dumps(value, sort_keys=True).encode('utf-8')


def test_digest_is_stable_and_roundtrip_semantics_are_exact():
    r = recipe()
    second = decode_multitrack_recipe(r.to_bytes())
    assert second == r
    assert second.sha256 == r.sha256
    assert second.to_bytes() == r.to_bytes()
    r.check_score(score())


def test_canonical_serialization_is_independent_of_json_field_order():
    r=recipe()
    source=json.dumps(to_dict(r),sort_keys=False,indent=3).encode()
    restored=decode_multitrack_recipe(source)
    assert restored.to_bytes() == r.to_bytes()
    assert restored.sha256 == r.sha256


def test_recipe_is_pinned_to_exact_score():
    r=recipe()
    wrong=Score.empty(end=960)
    with pytest.raises(MidiIntegrityError,match='another score'):
        r.check_score(wrong)
    with pytest.raises(MidiWorkstationError,match='immutable'):
        r.check_score(None)


def test_invalid_sha256_is_rejected_on_construction_and_decode():
    with pytest.raises(MidiWorkstationError):
        MultitrackRecipe('x'*64,recipe().settings)
    document=to_dict(recipe())
    document['score_sha256']='f'*63
    with pytest.raises(MidiWorkstationError):
        decode_multitrack_recipe(reencode(document))


def test_typed_settings_are_required():
    with pytest.raises(MidiWorkstationError):
        MultitrackRecipe(score().sha256, {'patches': []})


@pytest.mark.parametrize('fault',[
    lambda o:o.update(protocol='another-version'),
    lambda o:o.update(uri='https://remote.example/audio.wav'),
    lambda o:o['render'].update(arbitrary=1),
    lambda o:o['render'].pop('channels'),
    lambda o:o['render'].update(patches=[]),
    lambda o:o['render'].update(patches=[None]),
    lambda o:o['render']['patches'][0].update(script='import os'),
    lambda o:o['render']['patches'][0].pop('waveform'),
    lambda o:o['render']['patches'][0].update(waveform='external'),
    lambda o:o['render']['patches'][0].update(gain=float('nan')),
    lambda o:o['render']['patches'][0].update(table_size=True),
    lambda o:o['render'].update(maximum_notes=5000),
    lambda o:o['render'].update(sample_rate='8000'),
    lambda o:o['render'].update(normalization='compress'),
    lambda o:o['render'].update(include_muted=1),
])
def test_invalid_or_unsafe_recipe_fields_rejected(fault):
    data=to_dict(recipe())
    fault(data)
    with pytest.raises((MidiWorkstationError,MidiFormatError,TypeError)):
        decode_multitrack_recipe(reencode(data))


@pytest.mark.parametrize('bad',[
    b'', b'null', b'[]', b'{}', b'{',
    b'\xff', b'{"protocol":NaN}',
    b'{"protocol":1,"protocol":2}',
    b'{"protocol":"a","score_sha256":"b","render":{"x":true}}',
])
def test_invalid_json_is_rejected(bad):
    with pytest.raises((MidiWorkstationError,MidiFormatError)):
        decode_multitrack_recipe(bad)


def test_recipe_byte_budget_rejects_oversized_input():
    with pytest.raises(MidiLimitError):
        decode_multitrack_recipe(b'X'*(MAX_RECIPE_BYTES+1))


def test_patches_require_canonical_order():
    original=to_dict(recipe())
    first=dict(original['render']['patches'][0])
    first['track_id']='b'
    second=dict(first)
    second['track_id']='a'
    original['render']['patches']=[first,second]
    with pytest.raises(MidiWorkstationError, match='unique and sorted'):
        decode_multitrack_recipe(reencode(original))


def test_no_silent_default_values_during_decode():
    data=to_dict(recipe())
    data['render']['patches'][0].pop('gain')
    with pytest.raises(MidiFormatError):
        decode_multitrack_recipe(reencode(data))


def test_regular_file_snapshot_and_expected_sha256(tmp_path):
    original=recipe()
    file=tmp_path/'patches.json'
    file.write_bytes(original.to_bytes())
    found,receipt=load_multitrack_recipe(file, expected_sha256=original.sha256)
    assert found == original
    assert receipt['source_sha256']==original.sha256
    assert receipt['source_bytes']==len(original.to_bytes())
    assert receipt['canonical'] is True
    file.write_bytes(json.dumps(to_dict(original),indent=2).encode())
    found,receipt=load_multitrack_recipe(file)
    assert found == original
    assert receipt['canonical'] is False
    with pytest.raises(MidiIntegrityError):
        load_multitrack_recipe(file, expected_sha256=original.sha256)


def test_leaf_symlink_recipe_is_rejected(tmp_path):
    source=tmp_path/'source.json'
    source.write_bytes(recipe().to_bytes())
    alias=tmp_path/'alias.json'
    alias.symlink_to(source)
    with pytest.raises(MidiFormatError,match='symbolic-link'):
        load_multitrack_recipe(alias)
