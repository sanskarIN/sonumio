"""Reversible score patch identity, transactional replay and three-way merge."""
from dataclasses import replace
import hashlib
import json
import os

import pytest

from sonumio.arrangement import ArrangementPlan, compose_arrangement, arrangement_digest, event_key
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.music_edit import Retime, TrackArticulation, QuantizeVelocity, RemoveTrack
from sonumio.music_score_io import encode_score, read_score
from sonumio.music_score_patch import (
    ScorePatch, Endpoint, ScoreMetadata, make_patch, apply_patch, apply_with_receipt,
    inspect_patch, chain_patches, encode_patch, decode_patch, read_patch, merge_scores,
    MAX_PATCH_BYTES, MAX_CHAIN,
)


@pytest.fixture(scope="module")
def base():
    return compose_arrangement(ArrangementPlan(style="rhythmic",bars=4,bpm=120,seed=17))


def alter(score,track,velocity):
    return TrackArticulation(track,0,score.plan.bars,velocity).apply(score)


def test_exact_diff_apply_reverse_and_source_immutability(base):
    target=alter(base,"lead",.7)
    source_digest=arrangement_digest(base)
    patch=make_patch(base,target)
    report=inspect_patch(patch)
    assert report["tracks"]["lead"]["removed"]>0
    assert all(row["removed"]==row["added"]==0 for name,row in report["tracks"].items() if name!="lead")
    assert report["endpoint_verified"] is False
    result,receipt=apply_with_receipt(base,patch)
    assert encode_score(result)==encode_score(target)
    assert receipt["endpoint_verified"] and receipt["original_unchanged"]
    assert arrangement_digest(base)==source_digest
    assert encode_score(apply_patch(result,patch.reversed()))==encode_score(base)
    assert patch.reversed().reversed().sha256==patch.sha256


def test_metadata_only_patch(base):
    target=Retime(bpm=128,swing=.1).apply(base)
    patch=make_patch(base,target)
    assert patch.removed==patch.added==()
    assert inspect_patch(patch)["metadata_changed"]==["plan"]
    assert encode_score(apply_patch(base,patch))==encode_score(target)


def test_all_structural_components_can_change(base):
    target=compose_arrangement(ArrangementPlan(style="calm",mode="major",root=54,bars=8,bpm=140,seed=91))
    patch=make_patch(base,target)
    assert inspect_patch(patch)["metadata_changed"]==["plan","sections","harmony"]
    assert encode_score(apply_patch(base,patch))==encode_score(target)
    assert encode_score(apply_patch(target,patch.reversed()))==encode_score(base)


def test_noop_and_canonical_json_roundtrip(base):
    patch=make_patch(base,base)
    assert inspect_patch(patch)["noop"]
    assert patch.removed==patch.added==()
    assert decode_patch(encode_patch(patch),expected_sha256=patch.sha256)==patch
    assert encode_patch(decode_patch(encode_patch(patch)))==encode_patch(patch)
    assert encode_score(apply_patch(base,patch))==encode_score(base)


def test_numeric_representation_is_not_silently_collapsed(base):
    old=replace(base.events[0],pan=0)
    new=replace(old,pan=0.)
    a=replace(base,events=tuple(sorted((old,*base.events[1:]),key=event_key)))
    b=replace(a,events=tuple(sorted((new,*base.events[1:]),key=event_key)))
    assert old==new
    assert arrangement_digest(a)!=arrangement_digest(b)
    patch=make_patch(a,b)
    assert len(patch.removed)==len(patch.added)==1
    assert encode_score(apply_patch(a,patch))==encode_score(b)
    assert encode_score(apply_patch(b,patch.reversed()))==encode_score(a)


def test_stale_source_and_forged_endpoints_fail(base):
    target=alter(base,"lead",.7)
    patch=make_patch(base,target)
    with pytest.raises(GenerationError,match="digest mismatch"):
        apply_patch(alter(base,"bass",.8),patch)
    forged=replace(patch,target=replace(patch.target,score_sha256="a"*64))
    with pytest.raises(GenerationError,match="digest mismatch"):
        apply_patch(base,forged)
    wrong_metadata=replace(patch.source.metadata,plan=replace(base.plan,seed=18))
    forged=replace(patch,source=replace(patch.source,metadata=wrong_metadata))
    with pytest.raises(GenerationError,match="metadata or count"):
        apply_patch(base,forged)


def test_redundant_or_noncanonical_event_changes_rejected(base):
    patch=make_patch(base,alter(base,"lead",.7))
    with pytest.raises(GenerationError,match="canonical"):
        replace(patch,removed=tuple(reversed(patch.removed)))
    with pytest.raises(GenerationError,match="Duplicate"):
        replace(patch,removed=(patch.removed[0],patch.removed[0]))
    with pytest.raises(GenerationError,match="cancelling"):
        replace(patch,added=patch.removed)
    with pytest.raises(GenerationError,match="count"):
        replace(patch,target=replace(patch.target,event_count=patch.target.event_count+1))


def test_apply_rejects_missing_removal_even_with_valid_count(base):
    target=alter(base,"lead",.7)
    patch=make_patch(base,target)
    absent=replace(patch.removed[0],velocity=.001)
    removed=tuple(sorted((absent,*patch.removed[1:]),key=event_key))
    forged=replace(patch,removed=removed)
    with pytest.raises(GenerationError,match="missing source event"):
        apply_patch(base,forged)


def test_chain_verified_and_minimized(base):
    middle=alter(base,"lead",.7)
    end=QuantizeVelocity(9).apply(middle)
    first,second=make_patch(base,middle),make_patch(middle,end)
    combined,report=chain_patches(base,(first,second))
    assert combined.sha256==make_patch(base,end).sha256
    assert report["intermediate_scores_verified"]==2
    assert report["chain"][0]["result_score_sha256"]==arrangement_digest(middle)
    assert encode_score(apply_patch(base,combined))==encode_score(end)
    with pytest.raises(GenerationError,match="digest mismatch"):
        chain_patches(base,(second,first))
    with pytest.raises(GenerationError,match="bounded"):
        chain_patches(base,(first,)*(MAX_CHAIN+1))


def test_chain_forward_and_reverse_collapses_noop(base):
    patch=make_patch(base,alter(base,"lead",.7))
    combined,receipt=chain_patches(base,(patch,patch.reversed()))
    assert receipt["noop"] and receipt["endpoint_verified"]
    assert combined.sha256==make_patch(base,base).sha256


def test_json_digest_pin_and_mutation(base):
    patch=make_patch(base,alter(base,"lead",.7))
    data=encode_patch(patch)
    with pytest.raises(GenerationError,match="digest mismatch"):
        decode_patch(data,expected_sha256="0"*64)
    obj=json.loads(data)
    obj["patch"]["target"]["score_sha256"]="a"*64
    with pytest.raises(GenerationError,match="digest mismatch"):
        decode_patch(json.dumps(obj).encode())
    obj=json.loads(data)
    obj["patch"]["added"][0]["unknown"]=1
    with pytest.raises(GenerationError,match="fields"):
        decode_patch(json.dumps(obj).encode())


@pytest.mark.parametrize("data",[
    b"",b"x"*(MAX_PATCH_BYTES+1),b"\xff",
    b'{"format":"sonumio-score-patch-v1","format":"duplicate"}',
    b'{"format":"sonumio-score-patch-v1","sha256":NaN,"patch":{}}',
    b'{"format":"sonumio-score-patch-v1","sha256":"bad","patch":[]}',
])
def test_malformed_patch_bytes(data):
    with pytest.raises(GenerationError):
        decode_patch(data)


def test_independent_track_merge(base):
    left=alter(base,"lead",.6)
    right=alter(base,"bass",.7)
    expected=alter(left,"bass",.7)
    merged,report=merge_scores(base,left,right)
    assert report["merged"] and not report["conflicts"]
    assert report["track_sources"]["lead"]=="left"
    assert report["track_sources"]["bass"]=="right"
    assert encode_score(merged)==encode_score(expected)
    assert encode_score(merge_scores(base,right,left)[0])==encode_score(expected)


def test_same_track_conflict_and_identical_changes(base):
    left=alter(base,"lead",.6)
    right=alter(base,"lead",.8)
    result,report=merge_scores(base,left,right)
    assert result is None and not report["merged"]
    assert report["conflicts"]==[{"domain":"lead","reason":"both branches changed this track differently"}]
    assert encode_score(merge_scores(base,left,left)[0])==encode_score(left)
    assert encode_score(merge_scores(base,left,base)[0])==encode_score(left)


def test_global_metadata_change_conflicts_with_other_track_edit(base):
    left=Retime(bpm=128).apply(base)
    right=alter(base,"bass",.7)
    result,report=merge_scores(base,left,right)
    assert result is None
    assert any(conflict["domain"]=="metadata" for conflict in report["conflicts"])
    right=Retime(bpm=136).apply(base)
    assert merge_scores(base,left,right)[0] is None


def test_identical_metadata_and_separate_track_changes_merge(base):
    retimed=Retime(bpm=128).apply(base)
    left=alter(retimed,"lead",.6)
    right=alter(retimed,"bass",.7)
    merged,report=merge_scores(base,left,right)
    assert report["merged"] and report["metadata_source"]=="both"
    assert encode_score(merged)==encode_score(alter(left,"bass",.7))


def test_cli_complete_diff_apply_reverse_chain_and_inspection(base,tmp_path,capsys):
    source=tmp_path/"source.json"
    target=tmp_path/"target.json"
    patch=tmp_path/"patch.json"
    result=tmp_path/"result.json"
    reverse=tmp_path/"reverse.json"
    chain=tmp_path/"chain.json"
    source.write_bytes(encode_score(base))
    edited=alter(base,"lead",.7)
    target.write_bytes(encode_score(edited))
    args=["music-score-diff",str(source),str(target),"--output",str(patch)]
    assert main(args)==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt["endpoint_verified"]
    assert receipt["output_sha256"]==hashlib.sha256(patch.read_bytes()).hexdigest()
    assert main(["music-patch-inspect",str(patch),"--expected-patch-sha256",receipt["patch_sha256"]])==0
    assert json.loads(capsys.readouterr().out)["endpoint_verified"] is False
    assert main(["music-patch-apply",str(source),str(patch),"--output",str(result)])==0
    assert result.read_bytes()==target.read_bytes()
    assert main(["music-patch-reverse",str(source),str(patch),"--output",str(reverse)])==0
    assert main(["music-patch-chain",str(source),str(patch),str(reverse),"--output",str(chain)])==0
    assert inspect_patch(read_patch(chain))["noop"]
    assert main(args)==2
    assert main(args+["--overwrite","--expected-output-sha256","0"*64])==2
    assert hashlib.sha256(patch.read_bytes()).hexdigest()==receipt["output_sha256"]
    assert main(["music-patch-apply",str(source),str(patch),"--output",str(source),"--overwrite"])==2
    assert source.read_bytes()==encode_score(base)


def test_cli_conflicting_merge_preserves_existing_output(base,tmp_path,capsys):
    paths=[tmp_path/(name+".json") for name in ("base","left","right","result")]
    for path,score in zip(paths,(base,alter(base,"lead",.6),alter(base,"lead",.8),base),strict=True):
        path.write_bytes(encode_score(score))
    args=["music-score-merge",*(str(path) for path in paths[:3]),"--output",str(paths[3]),"--overwrite"]
    assert main(args)==2
    report=json.loads(capsys.readouterr().out)
    assert report["conflicts"][0]["domain"]=="lead"
    assert paths[3].read_bytes()==encode_score(base)
    paths[2].write_bytes(encode_score(alter(base,"bass",.7)))
    assert main(args)==0
    assert json.loads(capsys.readouterr().out)["merged"]
    assert paths[3].read_bytes()==encode_score(alter(alter(base,"lead",.6),"bass",.7))
