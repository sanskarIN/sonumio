"""Independent domain and replay oracles for sibling project merges."""
from dataclasses import replace
import json

import pytest

from sonumio.arrangement import ArrangementPlan, TRACKS, compose_arrangement, event_key
from sonumio.errors import GenerationError
from sonumio.music_automation import AutomationPlan, MixLane, MixKey, PARAMETERS
from sonumio.music_mix import MixPlan
from sonumio.music_project import create_project, revise_project, verify_revision, encode_project
from sonumio.music_project_merge import (
    describe_project_changes, merge_projects, verify_merge_receipt,
)
from sonumio.music_project_merge_cli import main


@pytest.fixture
def base():
    return create_project("Sibling test", compose_arrangement(ArrangementPlan(bars=4, bpm=120, seed=17)))


def mix_edit(project, **changes):
    return revise_project(project, automation=replace(project.automation,
                                                       mix=replace(project.automation.mix, **changes)))


def gain_edit(project, track, gain):
    gains = list(project.automation.mix.gains)
    gains[TRACKS.index(track)] = gain
    return mix_edit(project, gains=tuple(gains))


def lane_edit(project, track, parameter="gain", value=.4, tick=0):
    lanes = {lane.identity: lane for lane in project.automation.lanes}
    lanes[track, parameter] = MixLane(track, parameter, (MixKey(tick, value),))
    ordered = tuple(lanes[t, p] for t in TRACKS for p in PARAMETERS if (t, p) in lanes)
    return revise_project(project, automation=replace(project.automation, lanes=ordered))


def track_edit(project, track, velocity):
    score = project.score
    index = next(i for i, event in enumerate(score.events) if event.track == track)
    events = list(score.events)
    events[index] = replace(events[index], velocity=velocity)
    return revise_project(project, score=replace(score, events=tuple(sorted(events, key=event_key))))


def test_title_and_room_merge(base):
    left = revise_project(base, title="New title")
    right = mix_edit(base, room=.5)
    result, receipt = merge_projects(base, left, right)
    assert result.title == left.title
    assert result.automation.mix.room == .5
    assert result.revision == 2 and result.parent_project_sha256 == left.sha256
    assert receipt["domains"]["title"] == "left"
    assert receipt["domains"]["mix.room"] == "right"
    assert receipt["sibling_links_verified"]
    assert receipt["authenticated"] is False
    assert verify_revision(left, result)["parent_verified"]
    assert verify_merge_receipt(base, left, right, result, receipt) == receipt


def test_independent_track_gains_merge(base):
    left = gain_edit(base, "lead", .7)
    right = gain_edit(base, "bass", .1)
    result, receipt = merge_projects(base, left, right)
    assert result.automation.mix.gains[TRACKS.index("lead")] == .7
    assert result.automation.mix.gains[TRACKS.index("bass")] == .1
    assert receipt["domains"]["gain.lead"] == "left"
    assert receipt["domains"]["gain.bass"] == "right"


def test_different_track_edits_merge_with_mixer(base):
    left = track_edit(base, "lead", .777)
    right = track_edit(base, "keys", .123)
    result, receipt = merge_projects(base, left, right)
    assert result.score_sha256 not in (left.score_sha256, right.score_sha256)
    assert receipt["domains"]["score"] == "combined"
    assert any(e.velocity == .777 for e in result.score.events if e.track == "lead")
    assert any(e.velocity == .123 for e in result.score.events if e.track == "keys")


def test_automation_lanes_independent(base):
    left = lane_edit(base, "lead")
    right = lane_edit(base, "lead", parameter="balance", value=-.4)
    result, receipt = merge_projects(base, left, right)
    assert len(result.automation.lanes) == 2
    assert receipt["domains"]["lane.lead.gain"] == "left"
    assert receipt["domains"]["lane.lead.balance"] == "right"


def test_lane_deletion_and_unrelated_edit(base):
    original = create_project("With lane", base.score, lane_edit(base, "keys").automation)
    left = revise_project(original, automation=AutomationPlan((), original.automation.mix))
    right = lane_edit(original, "lead")
    result, receipt = merge_projects(original, left, right)
    assert tuple(lane.track for lane in result.automation.lanes) == ("lead",)
    assert receipt["domains"]["lane.keys.gain"] == "left"


@pytest.mark.parametrize("field", ["title", "room", "gain", "lane", "routing", "score"])
def test_same_domain_conflicts_fail_closed(base, field):
    if field == "title":
        left, right = revise_project(base, title="L"), revise_project(base, title="R")
        domain = "title"
    elif field == "room":
        left, right = mix_edit(base, room=.3), mix_edit(base, room=.4)
        domain = "mix.room"
    elif field == "gain":
        left, right = gain_edit(base, "keys", .1), gain_edit(base, "keys", .3)
        domain = "gain.keys"
    elif field == "lane":
        left, right = lane_edit(base, "keys", value=.5), lane_edit(base, "keys", value=.7)
        domain = "lane.keys.gain"
    elif field == "routing":
        left, right = mix_edit(base, mute=("keys",)), mix_edit(base, solo=("lead",))
        domain = "mix.routing"
    else:
        left, right = track_edit(base, "lead", .111), track_edit(base, "lead", .222)
        domain = "score.lead"
    result, receipt = merge_projects(base, left, right)
    assert result is None and receipt["merged"] is False
    assert domain in [c["domain"] for c in receipt["conflicts"]]


def test_noop_reconciliation_returns_left(base):
    left = revise_project(base, title="Same")
    result, receipt = merge_projects(base, left, left)
    assert result is left
    assert receipt["already_in_left"]
    assert receipt["result_revision"] == 1


def test_identical_lane_edits_merge_without_conflict(base):
    left = lane_edit(base, "keys")
    right = lane_edit(base, "keys")
    result, receipt = merge_projects(base, left, right)
    assert result.sha256 == left.sha256
    assert receipt["domains"]["lane.keys.gain"] == "both"


def test_clock_edit_and_concurrent_automation_conflict(base):
    left = revise_project(base, score=replace(base.score, plan=replace(base.score.plan, bpm=128)))
    right = lane_edit(base, "keys", tick=32)
    result, receipt = merge_projects(base, left, right)
    assert result is None
    assert any("different score clock" in c["reason"] for c in receipt["conflicts"])


def test_clock_edit_and_static_gain_can_merge(base):
    left = revise_project(base, score=replace(base.score, plan=replace(base.score.plan, bpm=128)))
    right = gain_edit(base, "bass", .5)
    result, receipt = merge_projects(base, left, right)
    assert result.score.plan.bpm == 128
    assert result.automation.mix.gains[TRACKS.index("bass")] == .5
    assert receipt["merged"]


def test_direct_sibling_links_required(base):
    left = revise_project(base, title="Left")
    descendant = revise_project(left, title="Descendant")
    with pytest.raises(GenerationError, match="link"):
        merge_projects(base, left, descendant)


def test_inputs_immutable_and_diff_replay(base):
    left = revise_project(base, title="Changed")
    right = lane_edit(base, "bass")
    before = [encode_project(p) for p in (base, left, right)]
    result, receipt = merge_projects(base, left, right)
    diff = describe_project_changes(base, result)
    assert [row["domain"] for row in diff["changed_domains"]] == ["title", "lane.bass.gain"]
    assert diff["revision_link_declared"] is False
    assert before == [encode_project(p) for p in (base, left, right)]
    assert merge_projects(base, left, right)[1] == receipt
    with pytest.raises(GenerationError, match="receipt"):
        verify_merge_receipt(base, left, right, result, {**receipt, "authenticated": True})
    with pytest.raises(GenerationError, match="result"):
        verify_merge_receipt(base, left, right, left, receipt)


def _write(tmp_path, name, project):
    path = tmp_path / (name + ".sonumio")
    path.write_bytes(encode_project(project))
    return str(path)


def test_cli_preview_merge_verify_and_conflicts(base, tmp_path, capsys):
    left = revise_project(base, title="Left")
    right = mix_edit(base, room=.5)
    inputs = [_write(tmp_path, n, p) for n, p in zip(("base", "left", "right"), (base, left, right))]
    output = tmp_path / "out.sonumio"
    assert main(["preview", *inputs]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert main(["merge", *inputs, "--output", str(output)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["receipt_sha256"] == preview["receipt_sha256"]
    report = tmp_path / "receipt.json"
    report.write_text(json.dumps(receipt))
    assert main(["verify", *inputs, str(output), str(report)]) == 0
    assert json.loads(capsys.readouterr().out)["merged"]
    assert main(["merge", *inputs, "--output", inputs[0], "--overwrite"]) == 2
    assert main(["merge", *inputs, "--output", str(output)]) == 2
    conflict = _write(tmp_path, "conflict", revise_project(base, title="Right"))
    assert main(["merge", inputs[0], inputs[1], conflict, "--output", str(tmp_path / "conflict-output.sonumio")]) == 3
    assert not (tmp_path / "conflict-output.sonumio").exists()
