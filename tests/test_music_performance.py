"""Sample-clock performance and deterministic expressive timing regression tests."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from sonumio.arrangement import (
    Arrangement, ArrangementPlan, Event, TRACKS, arrangement_digest,
    compose_arrangement, event_key,
)
from sonumio.errors import GenerationError
from sonumio.music_automation import AutomationPlan, MixLane, MixKey, generate_automated_music
from sonumio.music_humanize import HumanizePlan, TrackFeel, humanize
from sonumio.music_mix import render_stems, generate_arrangement_audio
from sonumio.music_performance import (
    Expression, Performance, compile_performance, describe_performance,
    encode_performance, decode_performance, render_performance_stems,
    generate_performance_music, MAX_DOCUMENT_BYTES, MAX_OFFSET,
)
from sonumio.music_performance_cli import main
from sonumio.music_score_io import encode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4, bpm=160, seed=79, swing=.4))


def small_score(score, *events):
    return replace(score, events=tuple(sorted(events, key=event_key)))


def test_identity_matches_original_stems_and_wav(score):
    plan = Performance(arrangement_digest(score))
    original = render_stems(score)
    rendered = render_performance_stems(score, plan)
    for track in TRACKS:
        np.testing.assert_array_equal(rendered[track], original[track])
        assert not rendered[track].flags.writeable
    wav, report = generate_performance_music(score, plan)
    expected, _ = generate_arrangement_audio(score)
    assert wav == expected
    assert report["wav_sha256"] == hashlib.sha256(wav).hexdigest()
    assert report["symbolic_score_changed"] is False


def test_identity_matches_automated_wav(score):
    auto = AutomationPlan((MixLane("lead", "gain", (MixKey(0, .5), MixKey(32, 1.))),))
    performance = Performance(arrangement_digest(score))
    expected, _ = generate_automated_music(score, auto)
    actual, _ = generate_performance_music(score, performance, auto)
    assert actual == expected


def test_shift_is_exact_sample_translation(score):
    s = small_score(score, Event("lead", 60, 8, 4, .5, -.2))
    baseline = render_stems(s)["lead"]
    p = Performance(arrangement_digest(s), (Expression(0, 67, 67),))
    shifted = render_performance_stems(s, p)["lead"]
    np.testing.assert_array_equal(shifted[:, 67:], baseline[:, :-67])
    assert not np.any(shifted[:, :67])
    assert describe_performance(s, p)["maximum_onset_offset"] == 67


def test_percussion_seed_unchanged_when_shifted(score):
    s = small_score(score, Event("hat", 42, 8, 2, .5, 0.))
    baseline = render_stems(s)["hat"]
    plan = Performance(arrangement_digest(s), (Expression(0, -37, -37),))
    shifted = render_performance_stems(s, plan)["hat"]
    np.testing.assert_array_equal(shifted[:, :-37], baseline[:, 37:])


def test_scale_and_pan_are_bounded(score):
    s = small_score(score, Event("lead", 60, 8, 4, .9, .8))
    p = Performance(arrangement_digest(s), (Expression(0, 0, 0, 2., 1.),))
    row, = compile_performance(s, p)
    assert row.velocity == 1. and row.pan == 1.
    assert s.events[0].velocity == .9


def test_overlap_rejected_for_monophonic_track(score):
    s = small_score(score, Event("lead", 60, 4, 4, .5, 0.),
                    Event("lead", 62, 8, 4, .5, 0.))
    plan = Performance(arrangement_digest(s), (Expression(0, end_delta=1),))
    with pytest.raises(GenerationError, match="polyphony"):
        compile_performance(s, plan)


def test_same_pitch_overlap_rejected_for_polyphonic_track(score):
    s = small_score(score, Event("keys", 60, 4, 4, .5, 0.),
                    Event("keys", 60, 8, 4, .5, 0.))
    with pytest.raises(GenerationError, match="polyphony"):
        compile_performance(s, Performance(arrangement_digest(s), (Expression(1, -1),)))


def test_five_concurrent_keys_rejected(score):
    events = [Event("keys", p, 4, 4, .5, 0.) for p in (60, 62, 63, 65)]
    events.append(Event("keys", 67, 8, 4, .5, 0.))
    s = small_score(score, *events)
    with pytest.raises(GenerationError, match="polyphony"):
        compile_performance(s, Performance(arrangement_digest(s), (Expression(4, -1),)))


def test_bounds_and_wrong_score(score):
    s = small_score(score, Event("lead", 60, 0, 1, .5, 0.))
    with pytest.raises(GenerationError, match="frame"):
        compile_performance(s, Performance(arrangement_digest(s), (Expression(0, -1),)))
    with pytest.raises(GenerationError, match="SHA256"):
        compile_performance(s, Performance("0"*64))
    with pytest.raises(GenerationError, match="missing"):
        compile_performance(s, Performance(arrangement_digest(s), (Expression(1, 1),)))


@pytest.mark.parametrize("kwargs", [
    {"index": True}, {"index": -1}, {"onset_delta": True}, {"end_delta": MAX_OFFSET+1},
    {"velocity_scale": 0.}, {"velocity_scale": float("nan")}, {"pan_delta": 1.1},
])
def test_expression_rejects_bad_values(kwargs):
    with pytest.raises(GenerationError):
        Expression(**({"index": 0} | kwargs))


def test_document_canonical_identity_and_order(score):
    digest = arrangement_digest(score)
    p = Performance(digest, (Expression(0), Expression(1, 10, 10, 1, 0)))
    assert len(p.expressions) == 1
    assert decode_performance(encode_performance(p)) == p
    assert p.sha256 == hashlib.sha256(encode_performance(p)).hexdigest()
    with pytest.raises(GenerationError):
        Performance(digest, (Expression(1), Expression(0)))
    with pytest.raises(GenerationError):
        Performance(digest, (Expression(0), Expression(0)))


@pytest.mark.parametrize("data", [b"", b"{}", b"[]", b"\xff", b"x"*(MAX_DOCUMENT_BYTES+1),
                                  b'{"format":1,"format":2}'])
def test_invalid_documents(data):
    with pytest.raises(GenerationError):
        decode_performance(data)


def test_humanization_is_repeatable_and_changes_with_seed(score):
    feel = tuple(TrackFeel(track, 200, 90, .15, .1) for track in TRACKS)
    plan = HumanizePlan(13, feel)
    p1, r1 = humanize(score, plan)
    p2, r2 = humanize(score, plan)
    assert p1 == p2 and r1 == r2
    p3, _ = humanize(score, replace(plan, seed=14))
    assert p3.sha256 != p1.sha256
    assert r1["polyphony_verified"]
    assert len(compile_performance(score, p1)) == len(score.events)


def test_humanization_preserves_chord_onset_and_track_independence(score):
    s = small_score(score, Event("keys", 60, 8, 4, .5, 0.),
                    Event("keys", 63, 8, 4, .5, 0.),
                    Event("lead", 60, 8, 4, .5, 0.))
    keys = TrackFeel("keys", 500, 100, .1, .1)
    only, _ = humanize(s, HumanizePlan(13, (keys,)))
    both, _ = humanize(s, HumanizePlan(13, (keys, TrackFeel("lead", 500, 100))))
    a, b, _ = compile_performance(s, both)
    assert a.start == b.start
    assert tuple(e for e in both.expressions if e.index < 2) == only.expressions


def test_humanization_tight_boundary_and_shortening(score):
    s = small_score(score, Event("lead", 60, 0, 1, .5, 0.),
                    Event("lead", 62, 1, 1, .5, 0.),
                    Event("lead", 63, 2, 1, .5, 0.))
    for seed in range(30):
        p, _ = humanize(s, HumanizePlan(seed, (TrackFeel("lead", 1200, 1200),)))
        rows = compile_performance(s, p)
        assert rows[0].start == 0
        assert all(a.stop <= b.start for a, b in zip(rows, rows[1:]))
        assert all(row.stop-row.start >= 2 for row in rows)


def test_parameter_domain_separation(score):
    feel = TrackFeel("lead", 100, 40, .1, .1)
    a, _ = humanize(score, HumanizePlan(8, (feel,)))
    b, _ = humanize(score, HumanizePlan(8, (replace(feel, shortening_frames=200),)))
    left = {e.index: e for e in a.expressions}
    right = {e.index: e for e in b.expressions}
    for index in left.keys() & right.keys():
        assert left[index].onset_delta == right[index].onset_delta
        assert left[index].velocity_scale == right[index].velocity_scale
        assert left[index].pan_delta == right[index].pan_delta


def test_empty_feel_is_exact_identity(score):
    result, _ = humanize(score, HumanizePlan(4, tuple(TrackFeel(t) for t in TRACKS)))
    assert result == Performance(arrangement_digest(score))


def test_cli_end_to_end_and_no_overwrite(score, tmp_path, capsys):
    source = tmp_path / "score.json"
    perf = tmp_path / "performance.json"
    target = tmp_path / "performance.wav"
    source.write_bytes(encode_score(score))
    assert main(["humanize", str(source), "--seed", "42", "--track", "lead",
                 "--timing-frames", "100", "--velocity-variation", ".1"]) == 0
    perf.write_text(capsys.readouterr().out)
    plan = decode_performance(perf.read_bytes())
    assert main(["inspect", str(source), str(perf)]) == 0
    assert not json.loads(capsys.readouterr().out)["rendered"]
    args = ["render", str(source), str(perf), "--output", str(target)]
    assert main(args + ["--expected-performance-sha256", "0"*64]) == 2
    assert not target.exists()
    assert main(args + ["--expected-performance-sha256", plan.sha256]) == 0
    report = json.loads(capsys.readouterr().out)
    original = target.read_bytes()
    assert report["output_sha256"] == hashlib.sha256(original).hexdigest()
    assert main(args) == 2
    assert target.read_bytes() == original
