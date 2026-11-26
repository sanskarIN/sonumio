"""Controlled-language parser invariants, bilingual equivalence and output tests."""
import hashlib
import json
import wave
import pytest

from sonumio.audio_prompt import parse_prompt, canonical_prompt, build_source, tokenize
from sonumio.audio_prompt_cli import main
from sonumio.arrangement import ArrangementPlan
from sonumio.nature_scene import scene_digest
from sonumio.errors import GenerationError


def test_bilingual_music_prompt_has_same_semantics():
    a=parse_prompt("music calm minor key C# bpm 120 bars 8 swing 0.15 seed 17")
    b=parse_prompt("संगीत शांत माइनर कुंजी C♯ गति 120 बार 8 स्विंग 0.15 सीड 17")
    assert a==b
    assert a.music==ArrangementPlan("calm",61,"minor",120,8,17,.15)
    assert build_source(a)==build_source(b)
    assert parse_prompt(canonical_prompt(a))==a


def test_bilingual_nature_prompt_has_same_semantics():
    a=parse_prompt("nature forest seconds 2.125 seed 13")
    b=parse_prompt("प्रकृति जंगल सेकंड 2.125 सीड 13")
    assert a==b
    assert scene_digest(a.nature)==scene_digest(b.nature)
    assert parse_prompt(canonical_prompt(a))==a


@pytest.mark.parametrize("key,note",[
    ("c",60),("C♯",61),("Db",61),("d",62),("E♭",63),("e",64),
    ("f",65),("f#",66),("Gb",66),("g",67),("Ab",68),
    ("a",69),("Bb",70),("b",71),("48",48),("59",59),("71",71),
])
def test_key_aliases(key,note):
    assert parse_prompt("music key "+key).music.root==note


@pytest.mark.parametrize("style,alias",[
    ("calm","शांत"),("cinematic","सिनेमैटिक"),("rhythmic","लयबद्ध"),
])
def test_all_style_aliases(style,alias):
    assert parse_prompt("music "+style)==parse_prompt("वाद्य "+alias)


@pytest.mark.parametrize("preset,alias",[
    ("forest","जंगल"),("storm","तूफान"),("coast","समुद्र"),
    ("creek","झरना"),("campfire","अलाव"),
])
def test_all_preset_aliases(preset,alias):
    a=parse_prompt("nature "+preset+" seconds 1 seed 4294967295")
    assert a==parse_prompt("ध्वनि "+alias+" सेकंड 1 सीड 4294967295")
    assert parse_prompt(canonical_prompt(a))==a


def test_default_seed_and_explicit_override():
    a=parse_prompt("music calm",default_seed=29)
    assert a.music.seed==29
    assert parse_prompt("music calm seed 3",default_seed=29).music.seed==3
    n=parse_prompt("nature storm",default_seed=29)
    assert parse_prompt(canonical_prompt(n))==n
    with pytest.raises(GenerationError):
        parse_prompt("music",default_seed=True)


@pytest.mark.parametrize("prompt",[
    "", " ", "x", "nature", "music rock", "music calm rhythmic",
    "music major minor", "music bpm", "music bpm 59", "music bpm 161",
    "music bars 3", "music bars 33", "music key 72", "music key H",
    "music swing -0.1", "music swing 0.601", "music swing NaN",
    "music swing 1e-1", "music seed -1", "music seed 4294967296",
    "music tempo 120 bpm 130", "music seconds 2",
    "nature forest storm", "nature forest bpm 100",
    "nature forest seconds 0.9", "nature forest seconds 31",
    "nature forest seconds 1.0001", "nature forest seed 2 seed 3",
    "nature forest extra words", "music\x00", "music\u200b",
    "music "+("a "*50), "music "+("a"*801),
])
def test_bad_prompts_fail_closed(prompt):
    with pytest.raises(GenerationError):
        parse_prompt(prompt)


def test_normalization_only_changes_supported_syntax():
    assert tokenize(" MUSIC\t CALM \n KEY C♯ ")==("music","calm","key","c#")
    assert parse_prompt("music calm bars 4")!=parse_prompt("music calm bars 5")


def test_controlled_prompt_cli_music_receipt(tmp_path,capsys):
    path=tmp_path/"music.wav"
    args=["music calm bars 4 bpm 160 seed 17","--output",str(path)]
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert report["prompt_kind"]=="music"
    assert report["controlled_language"] and not report["language_model_used"]
    assert report["output_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert parse_prompt(report["canonical_prompt"]).music.bars==4
    with wave.open(str(path),"rb") as wav:
        assert wav.getnchannels()==2 and wav.getnframes()==144000
    original=path.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    with pytest.raises(SystemExit):
        main(args+["--overwrite","--expected-output-sha256","0"*64])
    assert path.read_bytes()==original


def test_controlled_prompt_cli_nature_and_rejected_prompt(tmp_path,capsys):
    path=tmp_path/"nature.wav"
    args=["प्रकृति जंगल सेकंड 1 सीड 17","--output",str(path)]
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert report["prompt_kind"]=="nature" and report["frames"]==24000
    assert report["output_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    bad=tmp_path/"bad.wav"
    with pytest.raises(SystemExit):
        main(["music unknown","--output",str(bad)])
    assert not bad.exists()
