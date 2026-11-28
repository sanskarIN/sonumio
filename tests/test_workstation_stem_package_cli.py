"""End-to-end strict recipe and CLI protection for source-only stem packages."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from sonumio.midi_workstation.contracts import MidiFormatError, MidiIntegrityError, MidiWorkstationError
from sonumio.midi_workstation.publish import PublicationOptions
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack
from sonumio.midi_workstation.score_io import encode_score
from sonumio.midi_workstation.stem_package import (
    StemPackagePlan, StemPatch,
)
from sonumio.midi_workstation.stem_package_cli import (
    RECIPE_PROTOCOL, encode_recipe, read_recipe, main,
)
from test_workstation_stem_package import fake_engine, plan, score


def files(tmp_path, *, score_override=None, settings=None):
    original=score() if score_override is None else score_override
    output=tmp_path/"score.json"
    output.write_bytes(encode_score(original))
    recipe=tmp_path/"recipe.json"
    recipe.write_bytes(encode_recipe(plan() if settings is None else settings, original.sha256))
    return original,output,recipe


def call(argv,capsys):
    status=main([str(x) for x in argv])
    output=capsys.readouterr()
    return status,output


def test_canonical_recipe_encodes_all_explicit_patches():
    original=score()
    binary=encode_recipe(plan(),original.sha256)
    options,pin=read_recipe(binary)
    assert options==plan()
    assert pin==original.sha256
    assert encode_recipe(options,pin)==binary
    assert json.loads(binary)["protocol"]==RECIPE_PROTOCOL
    assert json.loads(binary)["patches"][0]["track_id"]=="alpha"


def test_read_only_preflight_has_no_output(fake_engine,tmp_path,capsys):
    original,source,recipe=files(tmp_path)
    status,stdout=call(["plan",source,"--recipe",recipe],capsys)
    assert status==0
    report=json.loads(stdout.out)
    assert report["preflight"]["frames"]==4064
    assert report["preflight"]["source_score_sha256"]==original.sha256
    assert report["published"] is False
    assert fake_engine==[]
    assert sorted(p.name for p in tmp_path.iterdir())==["recipe.json","score.json"]


def test_render_inspect_and_digest_pinning(fake_engine,tmp_path,capsys):
    original,source,recipe=files(tmp_path)
    destination=tmp_path/"stems.zip"
    args=["render",source,"--recipe",recipe,"--output",destination]
    status,out=call(args,capsys)
    assert status==0,out.err
    rendered=json.loads(out.out)
    assert destination.is_file()
    assert rendered["render"]["stems"]==2
    assert rendered["render"]["source_score_sha256"]==original.sha256
    status,out=call(["inspect",destination,"--expected-sha256",
                     sha256(destination.read_bytes()).hexdigest()],capsys)
    assert status==0
    inspected=json.loads(out.out)
    assert inspected["source_score_sha256"]==original.sha256
    assert inspected["original_mathematical_synthesis"] is True
    assert len(fake_engine)==2


def test_cli_bad_score_pin_never_invokes_engine(fake_engine,tmp_path,capsys):
    _,source,recipe=files(tmp_path)
    destination=tmp_path/"stems.zip"
    status,out=call(["render",source,"--recipe",recipe,"--output",destination,
                     "--expected-sha256","0"*64],capsys)
    assert status==2
    assert not destination.exists()
    assert fake_engine==[]
    assert out.out==""


def test_cli_bad_recipe_pin_never_invokes_engine(fake_engine,tmp_path,capsys):
    _,source,recipe=files(tmp_path)
    destination=tmp_path/"stems.zip"
    status,out=call(["render",source,"--recipe",recipe,"--output",destination,
                     "--expected-recipe-sha256","0"*64],capsys)
    assert status==2
    assert not destination.exists()
    assert fake_engine==[]
    assert out.out==""


def test_recipe_wrong_score_digest_is_fatal(fake_engine,tmp_path,capsys):
    _,source,recipe=files(tmp_path)
    recipe.write_bytes(encode_recipe(plan(),"f"*64))
    destination=tmp_path/"stems.zip"
    status,out=call(["render",source,"--recipe",recipe,"--output",destination],capsys)
    assert status==2
    assert "different canonical score" in out.err
    assert not destination.exists()
    assert fake_engine==[]


def test_render_no_clobber_and_explicit_replace(fake_engine,tmp_path,capsys):
    _,source,recipe=files(tmp_path)
    path=tmp_path/"mix.zip"
    args=["render",source,"--recipe",recipe,"--output",path]
    assert call(args,capsys)[0]==0
    before=path.read_bytes()
    assert call(args,capsys)[0]==2
    assert path.read_bytes()==before
    status,out=call(args+["--overwrite","--expected-previous-sha256",
                         sha256(before).hexdigest()],capsys)
    assert status==0,out.err
    assert path.read_bytes()==before


@pytest.mark.parametrize("kind",["score","recipe"])
def test_source_file_cannot_be_overwritten_with_output(fake_engine,tmp_path,capsys,kind):
    _,source,recipe=files(tmp_path)
    destination=source if kind=="score" else recipe
    before=destination.read_bytes()
    status,out=call(["render",source,"--recipe",recipe,"--output",destination,
                     "--overwrite"],capsys)
    assert status==2
    assert not out.out
    assert destination.read_bytes()==before


def test_cli_preflight_does_not_mask_muted_policy(tmp_path,capsys):
    original,source,recipe=files(tmp_path,score_override=score(muted=True))
    status,out=call(["plan",source,"--recipe",recipe],capsys)
    assert status==2
    assert "all active pitched tracks" in out.err
    assert not out.out


def test_cli_does_not_make_missing_parent_by_default(fake_engine,tmp_path,capsys):
    _,source,recipe=files(tmp_path)
    destination=tmp_path/"missing"/"stems.zip"
    status,out=call(["render",source,"--recipe",recipe,"--output",destination],capsys)
    assert status==2
    assert not destination.exists()
    assert not destination.parent.exists()
    assert not out.out
    status,out=call(["render",source,"--recipe",recipe,"--output",destination,
                     "--create-parent"],capsys)
    assert status==0,out.err
    assert destination.is_file()


@pytest.mark.parametrize("bad",[
    b"{}", b"null", b"[]", b"not-json", b"{\"protocol\":\"wrong\"}",
    b"{\"x\":1,\"x\":2}", b"{\"x\":NaN}",
])
def test_invalid_recipe_document_rejected(bad):
    with pytest.raises((MidiFormatError,MidiWorkstationError)):
        read_recipe(bad)


def test_noncanonical_recipe_refused_even_if_parseable():
    original=score()
    content=json.loads(encode_recipe(plan(),original.sha256))
    bad=json.dumps(content,indent=2).encode()
    with pytest.raises(MidiFormatError,match="canonical"):
        read_recipe(bad)


def test_receipt_has_no_remote_or_neural_claims(fake_engine,tmp_path,capsys):
    _,source,recipe=files(tmp_path)
    status,out=call(["render",source,"--recipe",recipe,
                     "--output",tmp_path/"stems.zip"],capsys)
    assert status==0,out.err
    receipt=json.loads(out.out)
    assert receipt["model_inference"] is False
    assert receipt["remote_publication"] is False
    assert receipt["render"]["trained_model_inference"] is False


def test_archive_inspector_rejects_wrong_digest_before_parsing(tmp_path,capsys):
    path=tmp_path/"output.zip";path.write_bytes(b"garbage")
    status,out=call(["inspect",path,"--expected-sha256","0"*64],capsys)
    assert status==2
    assert not out.out


def test_suspicious_recipe_path_or_symlink_fails_closed(tmp_path,capsys):
    _,source,recipe=files(tmp_path)
    link=tmp_path/"recipe-link.json"
    try:link.symlink_to(recipe)
    except (NotImplementedError,OSError):pytest.skip("No symlink privileges")
    status,out=call(["plan",source,"--recipe",link],capsys)
    assert status==2
    assert "regular" in out.err.lower() or "symbolic-link" in out.err.lower()


def test_missing_output_required_by_parser(tmp_path):
    _,source,recipe=files(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["render",str(source),"--recipe",str(recipe)])
    assert exc.value.code==2
