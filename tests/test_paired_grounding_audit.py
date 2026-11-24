import copy
import json

import pytest

from research import paired_grounding_audit as audit


def decision(prediction, target=None, accepted=True):
    row = {"closest": prediction, "within_reference_radius": accepted and prediction is not None}
    if target is not None:
        row["target_rank"] = None if prediction is None else (1 if prediction == target else 2)
    return row


def report(mode="perfect", confounded=False):
    categories = audit.CATEGORIES
    noise = {
        family: [{"seed": seed, **decision("rain", accepted=confounded)}
                 for seed in (731, 732, 733)]
        for family in ("white_8k", "filtered_to_8k")
    }
    cases = []
    checks = []
    for arm in audit.ARMS:
        checks.append({"arm": arm, "sampling_seed": 301, "caption_invariant": True})
        for language in audit.LANGUAGES:
            for i, target in enumerate(categories):
                alt = categories[(i + 1) % len(categories)]
                prediction = target if mode == "perfect" else "rain" if mode == "collapsed" else None
                cases.append({
                    "arm": arm, "language": language, "sampling_seed": 301,
                    "requested": target, "rotated_label": alt,
                    "matched": decision(prediction, target),
                    "rotated": decision(prediction, alt),
                    "null_guidance": decision("rain", target),
                })
    return {
        "protocol": audit.SOURCE_PROTOCOL, "status": "scored",
        "configuration": {"categories": list(categories), "model_seed": 17, "sampling_seeds": [301]},
        "reference_probe": {
            "overlap_with_generator_data": False,
            "supports_interpreting_generated_scores": True,
            "noise_controls": noise,
            "categories_accepted_for_noise": ["rain"] if confounded else [],
        },
        "weight_or_audio_files_saved": False,
        "cases": cases, "null_guidance_checks": checks,
    }


def test_perfect_match_exact_assignment_distribution():
    result = audit.audit_report(report())
    assert result["panels_passing"] == 4
    assert result["all_panels_pass_descriptive_screen"]
    assert result["model_readiness_established"] is False
    for panel in result["panels"]:
        exact = panel["accepted_nonconfounded"]
        assert exact["observed"] == 6
        assert exact["permutations"] == 720
        assert exact["tail_count"] == 1
        assert exact["tail_fraction"] == 1 / 720
        assert exact["expected_correct"] == 1
        assert exact["distribution"] == {"0": 265, "1": 264, "2": 135, "3": 40,
                                         "4": 15, "5": 0, "6": 1}
        assert panel["matched_minus_null"] == 5
        assert panel["matched_minus_rotated"] == 6
    json.dumps(result, allow_nan=False)


def test_mode_collapse_is_not_grounding_even_with_accepted_predictions():
    result = audit.audit_report(report("collapsed"))
    assert result["panels_passing"] == 0
    for panel in result["panels"]:
        assert panel["matched_correct"] == panel["rotated_correct"] == panel["null_correct"] == 1
        assert panel["accepted_nonconfounded"]["tail_fraction"] == 1
        assert panel["distinct_accepted_categories"] == 1


def test_noise_confounds_are_excluded_from_all_controls():
    result = audit.audit_report(report("collapsed", confounded=True))
    assert result["noise_confounded_categories"] == ["rain"]
    for panel in result["panels"]:
        assert panel["forced_rank_one"]["observed"] == 1
        assert panel["accepted_nonconfounded"]["observed"] == 0
        assert panel["accepted_coverage"] == 0
        assert panel["matched_correct"] == panel["rotated_correct"] == panel["null_correct"] == 0
        assert panel["accepted_nonconfounded"]["tail_fraction"] == 1


def test_partial_coverage_adjusts_random_label_expectation():
    result = audit.audit_report(report(confounded=True))
    for panel in result["panels"]:
        assert panel["accepted_coverage"] == 5
        assert panel["accepted_nonconfounded"]["expected_correct"] == 5 / 6
        assert panel["accepted_nonconfounded"]["observed"] == 5


def test_abstention_never_creates_grounding():
    for panel in audit.audit_report(report("silent"))["panels"]:
        assert panel["accepted_coverage"] == 0
        assert panel["accepted_nonconfounded"]["distribution"]["0"] == 720
        assert panel["matched_minus_null"] == -1
        assert not panel["passes_descriptive_screen"]


def test_failed_reference_gate_does_not_promote_results():
    raw = report()
    raw["status"] = "reference_gate_failed"
    raw["reference_probe"]["supports_interpreting_generated_scores"] = False
    raw["cases"] = []
    raw["null_guidance_checks"] = []
    result = audit.audit_report(raw)
    assert result["status"] == "reference_gate_failed"
    assert result["panels"] == []


@pytest.mark.parametrize("corruption, message", [
    ("missing_case", "Incomplete"),
    ("duplicate_case", "duplicate"),
    ("wrong_rotation", "rotated label"),
    ("changed_rotation", "Rotated control"),
    ("wrong_rank", "Target rank"),
    ("false_null_check", "invariance"),
    ("changed_null", "Null prediction"),
    ("cross_language_null", "across languages"),
    ("false_confounds", "contradict controls"),
    ("overlap", "overlap"),
    ("duplicate_noise_seed", "Duplicate noise"),
])
def test_invalid_report_fails_closed(corruption, message):
    raw = report()
    first = raw["cases"][0]
    if corruption == "missing_case":
        raw["cases"].pop()
    elif corruption == "duplicate_case":
        raw["cases"][-1] = copy.deepcopy(first)
    elif corruption == "wrong_rotation":
        first["rotated_label"] = "rain"
    elif corruption == "changed_rotation":
        first["rotated"] = decision("thunder", "thunder")
    elif corruption == "wrong_rank":
        first["matched"]["target_rank"] = 2
    elif corruption == "false_null_check":
        raw["null_guidance_checks"][0]["caption_invariant"] = False
    elif corruption == "changed_null":
        first["null_guidance"] = decision("wind", "rain")
    elif corruption == "cross_language_null":
        for row in raw["cases"]:
            if row["arm"] == "raw" and row["language"] == "hi":
                row["null_guidance"] = decision("wind", row["requested"])
    elif corruption == "false_confounds":
        raw["reference_probe"]["categories_accepted_for_noise"] = ["rain"]
    elif corruption == "overlap":
        raw["reference_probe"]["overlap_with_generator_data"] = True
    elif corruption == "duplicate_noise_seed":
        rows = raw["reference_probe"]["noise_controls"]["white_8k"]
        rows[1]["seed"] = rows[0]["seed"]
    with pytest.raises(ValueError, match=message):
        audit.audit_report(raw)


def test_reader_rejects_nonfinite_and_oversized_json(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text('{"bad": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        audit.read_report(path)
    monkeypatch.setattr(audit, "MAX_REPORT_BYTES", 8)
    with pytest.raises(ValueError, match="size limit"):
        audit.read_report(path)


def test_cli_outputs_numeric_json(tmp_path, monkeypatch, capsys):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report()), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["paired_grounding_audit", str(path)])
    audit.main()
    assert json.loads(capsys.readouterr().out)["panels_passing"] == 4
