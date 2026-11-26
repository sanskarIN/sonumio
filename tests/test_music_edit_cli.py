"""Strict edit document parsing and atomic edited-score export."""
import hashlib
import json
import pytest

from sonumio.arrangement import ArrangementPlan,compose_arrangement,arrangement_digest
from sonumio.music_score_io import encode_score,decode_score
from sonumio.music_edit import Retime,Transpose,LeadBar
from sonumio.music_edit_cli import decode_edits,MAX_EDIT_BYTES,main
from sonumio.errors import GenerationError


def document(rows):
    return json.dumps({"format":"sonumio-score-edits-v1","operations":rows}).encode()


def row(kind,**params):
    return {"type":kind,"parameters":params}


def test_document_preserves_operation_order():
    data=document([row("retime",bpm=140),row("transpose",semitones=1),
                   row("lead_bar",bar=0,notes=[[0,8,61,.5]])])
    ops=decode_edits(data)
    assert ops==(Retime(140),Transpose(1),LeadBar(0,((0,8,61,.5),)))


@pytest.mark.parametrize("data",[
    b"",b"null",b"[]",b'{"format":"x","format":"y"}',
    b'{"x":NaN}',b"\xff",b" "*(MAX_EDIT_BYTES+1),
    document([]),document([row("unknown")]),
    document([row("retime",bpm=140,unexpected=1)]),
    document([row("transpose")]),document([row("lead_bar",bar=0,notes={})]),
    document([row("retime",bpm=140)]*33),
])
def test_reject_invalid_edit_document(data):
    with pytest.raises(GenerationError):
        decode_edits(data)


def test_cli_edits_and_preserves_original(tmp_path,capsys):
    source=tmp_path/"source.json"
    edits=tmp_path/"edits.json"
    output=tmp_path/"edited.json"
    score=compose_arrangement(ArrangementPlan(bars=4,bpm=120,seed=17))
    original=encode_score(score)
    source.write_bytes(original)
    edits.write_bytes(document([row("retime",bpm=140),row("transpose",semitones=1)]))
    args=[str(source),str(edits),"--expected-score-sha256",arrangement_digest(score),
          "--output",str(output)]
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    result=decode_score(output.read_bytes())
    assert result.plan.bpm==140 and result.plan.root==61
    assert report["output_sha256"]==hashlib.sha256(output.read_bytes()).hexdigest()
    assert source.read_bytes()==original
    before=output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    with pytest.raises(SystemExit):
        main(args+["--overwrite","--expected-output-sha256","0"*64])
    assert output.read_bytes()==before
    edits.write_bytes(document([row("retime",bpm=140),row("transpose",semitones=23)]))
    with pytest.raises(SystemExit):
        main(args+["--overwrite"])
    assert output.read_bytes()==before
