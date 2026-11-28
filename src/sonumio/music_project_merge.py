"""Conservative content-linked project revision reconciliation.

The v1 project format has one parent. A successful merge therefore creates
a revision of the left project, with the right and common parent recorded
in a separate deterministic receipt. Only direct siblings are accepted.
A receipt is a reproducibility record, not a signature or ancestry proof.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from .arrangement import TRACKS
from .errors import GenerationError
from .music_automation import AutomationPlan, PARAMETERS, encode_automation
from .music_mix import MixPlan
from .music_project import MusicProject, revise_project, verify_revision
from .music_score_patch import make_patch, inspect_patch, merge_scores, ScoreMetadata

PROTOCOL = "sonumio-project-merge-v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _identity(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_project(value):
    if not isinstance(value, MusicProject):
        raise GenerationError("Expected validated music project")
    return value


def _automation_fields(automation):
    """Canonical semantic fields; track gain and lane edits are independent.

    Mute and solo are a single routing domain because an empty solo set
    activates all unmuted tracks and combining independently modified sets
    can radically change that meaning.
    """
    obj = json.loads(encode_automation(automation))
    mix = obj["mix"]
    fields = {
        "mix.room": mix["room"],
        "mix.peak_ceiling": mix["peak_ceiling"],
        "mix.routing": {"mute": mix["mute"], "solo": mix["solo"]},
    }
    for track, gain in zip(TRACKS, mix["gains"], strict=True):
        fields["gain." + track] = gain
    lanes = {(lane["track"], lane["parameter"]): lane for lane in obj["lanes"]}
    for track in TRACKS:
        for parameter in PARAMETERS:
            fields[f"lane.{track}.{parameter}"] = lanes.get((track, parameter))
    return fields


def _field_order():
    return ("title", "score", "mix.room", "mix.peak_ceiling", "mix.routing",
            *("gain." + track for track in TRACKS),
            *(f"lane.{track}.{parameter}" for track in TRACKS for parameter in PARAMETERS))


def _project_fields(project):
    return {"title": project.title, "score": project.score_sha256,
            **_automation_fields(project.automation)}


def describe_project_changes(source, target):
    """Describe semantic changes without pretending to verify ancestry."""
    _require_project(source)
    _require_project(target)
    before, after = map(_project_fields, (source, target))
    changes = []
    for domain in _field_order():
        if _canonical(before[domain]) != _canonical(after[domain]):
            changes.append({"domain": domain, "before_sha256": _identity(before[domain]),
                            "after_sha256": _identity(after[domain])})
    patch = make_patch(source.score, target.score)
    return {
        "protocol": PROTOCOL, "source_project_sha256": source.sha256,
        "target_project_sha256": target.sha256, "changed_domains": changes,
        "score_patch": inspect_patch(patch),
        "revision_link_declared": target.parent_project_sha256 == source.sha256,
        "semantic_noop": not changes, "authenticated": False,
    }


def _pick(base, left, right):
    b, l, r = map(_canonical, (base, left, right))
    if l == r:
        return left, "base" if l == b else "both"
    if l == b:
        return right, "right"
    if r == b:
        return left, "left"
    return None, "conflict"


def _changed_lanes(a, b):
    left, right = map(_automation_fields, (a, b))
    return {name for name in left if name.startswith("lane.")
            and _canonical(left[name]) != _canonical(right[name])}


def _reconstruct_automation(values, left, right, base):
    # Values have already been canonicalized by encode_automation. Use
    # the original validated lane objects rather than manufacturing keys.
    lookup = {}
    for project in (base, left, right):
        fields = _automation_fields(project.automation)
        for lane in project.automation.lanes:
            name = f"lane.{lane.track}.{lane.parameter}"
            lookup[(name, _canonical(fields[name]))] = lane
    lanes = []
    for track in TRACKS:
        for parameter in PARAMETERS:
            name = f"lane.{track}.{parameter}"
            if values[name] is not None:
                lanes.append(lookup[(name, _canonical(values[name]))])
    routing = values["mix.routing"]
    mix = MixPlan(tuple(values["gain." + track] for track in TRACKS),
                  tuple(routing["mute"]), tuple(routing["solo"]),
                  values["mix.room"], values["mix.peak_ceiling"])
    return AutomationPlan(tuple(lanes), mix)


def _clock_identity(score):
    # BPM and swing alter the mapping from ticks to frames; bar count
    # alters the permitted tick domain. Other metadata remains under the
    # existing conservative score merge policy.
    return (score.plan.bpm, round(score.plan.swing * 1000), score.plan.bars)


def _receipt(base, left, right, result, domains, conflicts, score_receipt):
    record = {
        "protocol": PROTOCOL,
        "base_project_sha256": base.sha256,
        "left_project_sha256": left.sha256,
        "right_project_sha256": right.sha256,
        "result_project_sha256": result.sha256 if result is not None else None,
        "result_revision": result.revision if result is not None else None,
        "result_parent_project_sha256": result.parent_project_sha256 if result is not None else None,
        "merged": result is not None,
        "already_in_left": result is not None and result.sha256 == left.sha256,
        "domains": domains,
        "conflicts": conflicts,
        "score_merge": score_receipt,
        "sibling_links_verified": True,
        "authenticated": False,
    }
    return {**record, "receipt_sha256": _identity(record)}


def merge_projects(base, left, right):
    """Return (project or None, receipt), with no mutation or file writes.

    Conflicts are reported rather than automatically resolved. A result
    that already equals left is returned unchanged, avoiding meaningless
    revisions. A successful new revision always has left as its parent.
    """
    for project in (base, left, right):
        _require_project(project)
    verify_revision(base, left)
    verify_revision(base, right)
    before, ours, theirs = map(_project_fields, (base, left, right))
    values = {}
    domains = {}
    conflicts = []

    for domain in _field_order():
        if domain == "score":
            continue
        chosen, source = _pick(before[domain], ours[domain], theirs[domain])
        domains[domain] = source
        if source == "conflict":
            conflicts.append({"domain": domain, "reason": "different concurrent edits"})
        else:
            values[domain] = chosen

    score, score_receipt = merge_scores(base.score, left.score, right.score)
    if score is None:
        domains["score"] = "conflict"
        for conflict in score_receipt["conflicts"]:
            conflicts.append({"domain": "score." + conflict["domain"],
                              "reason": conflict["reason"]})
    else:
        if score_receipt["result_score_sha256"] == base.score_sha256:
            domains["score"] = "base"
        elif left.score_sha256 == right.score_sha256:
            domains["score"] = "both"
        elif score_receipt["result_score_sha256"] == left.score_sha256:
            domains["score"] = "left"
        elif score_receipt["result_score_sha256"] == right.score_sha256:
            domains["score"] = "right"
        else:
            domains["score"] = "combined"

        # A concurrent automation edit made against a different clock
        # needs explicit reconciliation, even when its tick range fits.
        target_clock = _clock_identity(score)
        for side, project in (("left", left), ("right", right)):
            if _clock_identity(project.score) != target_clock:
                for name in sorted(_changed_lanes(base.automation, project.automation)):
                    if domains.get(name) in (side, "both"):
                        conflicts.append({"domain": name,
                            "reason": f"{side} automation edit uses a different score clock"})

    if conflicts:
        return None, _receipt(base, left, right, None, domains, conflicts, score_receipt)

    try:
        automation = _reconstruct_automation(values, left, right, base)
        automation.validate_score(score)
    except GenerationError:
        conflicts.append({"domain": "automation", "reason": "combined automation violates score bounds"})
        return None, _receipt(base, left, right, None, domains, conflicts, score_receipt)

    if (values["title"] == left.title and score_receipt["result_score_sha256"] == left.score_sha256
            and automation.sha256 == left.automation_sha256):
        result = left
    else:
        result = revise_project(left, title=values["title"], score=score, automation=automation)
        verify_revision(left, result)
    return result, _receipt(base, left, right, result, domains, conflicts, score_receipt)


def verify_merge_receipt(base, left, right, result, receipt):
    """Replay a claimed successful reconciliation against all three inputs."""
    _require_project(result)
    if type(receipt) is not dict or receipt.get("protocol") != PROTOCOL:
        raise GenerationError("Invalid project merge receipt")
    expected, report = merge_projects(base, left, right)
    if expected is None or expected.sha256 != result.sha256:
        raise GenerationError("Project merge result differs from replay")
    if _canonical(report) != _canonical(receipt):
        raise GenerationError("Project merge receipt differs from replay")
    return report
