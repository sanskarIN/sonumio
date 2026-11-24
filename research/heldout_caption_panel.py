"""Held-out lexical intervention panel for waveform text conditioning.

The panel is a fixed six-category synthetic-language diagnostic, not an
open-vocabulary benchmark. Alternative captions are excluded from the
existing template corpus and compared at identical initial noise.
"""
from __future__ import annotations

from itertools import combinations
from statistics import mean

import torch

from research.generation_audit import _compare_waveforms
from research.paired_grounding_audit import _exact_assignment
from research.text_audio import ByteTextTokenizer, SYNTHETIC_COMBINATIONS, _synth_prompts
from research.velocity_ddim import sample_velocity_ddim
from research.wave_diffusion import CATEGORIES

LANGUAGES = ("en", "hi")
PARAPHRASES = {
    "en": (
        "steady rainfall outdoors", "a low thunder rumble", "gusts of moving air",
        "birds chirping among trees", "waves rolling along the shore",
        "a sustained ambient tone",
    ),
    "hi": (
        "लगातार होती बारिश", "बादलों की गहरी गरज", "चलती हवा के झोंके",
        "पेड़ों के बीच चिड़ियों की चहचहाहट", "किनारे पर आती समुद्री लहरें",
        "धीरे बजती मधुर धुन",
    ),
}


def validate_caption_panel() -> None:
    """Check token identities, not merely spelling, against all train templates."""
    tokenizer = ByteTextTokenizer()
    known = tokenizer.encode([
        _synth_prompts(members, language, variant)
        for members in SYNTHETIC_COMBINATIONS
        for language in LANGUAGES for variant in (0, 1)
    ])
    known_rows = {tuple(row.tolist()) for row in known}
    alternative = tokenizer.encode([
        caption for language in LANGUAGES for caption in PARAPHRASES[language]
    ])
    alternatives = {tuple(row.tolist()) for row in alternative}
    if len(alternatives) != 2 * len(CATEGORIES) or alternatives & known_rows:
        raise ValueError("Held-out caption panel overlaps known training templates")


def _accepted(decision: dict, category: str) -> bool:
    return decision["within_reference_radius"] and decision["closest"] == category


def _summarize(rows: list[dict], references: list[torch.Tensor], alternatives: list[torch.Tensor]) -> dict:
    if len(rows) != len(CATEGORIES):
        raise ValueError("Incomplete lexical panel")
    ref_predictions = tuple(
        row["reference"]["closest"] if row["reference"]["within_reference_radius"] else None
        for row in rows
    )
    alt_predictions = tuple(
        row["alternative"]["closest"] if row["alternative"]["within_reference_radius"] else None
        for row in rows
    )
    ref_correct = [_accepted(row["reference"], row["category"]) for row in rows]
    alt_correct = [_accepted(row["alternative"], row["category"]) for row in rows]
    within = mean(row["same_category_waveform_mae"] for row in rows)
    between = mean(
        _compare_waveforms(references[i], alternatives[j])["mean_absolute_difference"]
        for i in range(len(rows)) for j in range(len(rows)) if i != j
    )
    alt_pair = mean(
        _compare_waveforms(alternatives[i], alternatives[j])["mean_absolute_difference"]
        for i, j in combinations(range(len(rows)), 2)
    )
    return {
        "reference_correct_accepted": sum(ref_correct),
        "alternative_correct_accepted": sum(alt_correct),
        "both_correct_accepted": sum(a and b for a, b in zip(ref_correct, alt_correct, strict=True)),
        "reference_only_correct": sum(a and not b for a, b in zip(ref_correct, alt_correct, strict=True)),
        "alternative_only_correct": sum(b and not a for a, b in zip(ref_correct, alt_correct, strict=True)),
        "distinct_reference_accepted_categories": len(set(ref_predictions) - {None}),
        "distinct_alternative_accepted_categories": len(set(alt_predictions) - {None}),
        "reference_assignment": _exact_assignment(ref_predictions),
        "alternative_assignment": _exact_assignment(alt_predictions),
        "same_category_reference_to_alternative_mae": within,
        "different_category_reference_to_alternative_mae": between,
        "different_category_alternative_mae": alt_pair,
        "different_minus_same_category_mae": between - within,
        "identical_same_category_waveforms": sum(row["identical_same_category_waveforms"] for row in rows),
        "waveform_difference_is_not_semantic_quality": True,
    }


@torch.no_grad()
def evaluate_caption_panel(model, schedule, probe, *, evaluations: int, seed: int) -> dict:
    """Compare same-category paraphrases and cross-category controls."""
    validate_caption_panel()
    settings = {"evaluations": evaluations, "seed": seed}
    panels = {}
    null_reference = sample_velocity_ddim(
        model, schedule, _synth_prompts((0,), "en", seed), guidance=0, **settings,
    )
    null_alternative = sample_velocity_ddim(
        model, schedule, PARAPHRASES["hi"][1], guidance=0, **settings,
    )
    if not torch.equal(null_reference, null_alternative):
        raise ValueError("Zero guidance changed with held-out caption")
    for language in LANGUAGES:
        rows, references, alternatives = [], [], []
        for index, category in enumerate(CATEGORIES):
            reference = sample_velocity_ddim(
                model, schedule, _synth_prompts((index,), language, seed), guidance=1, **settings,
            ).clamp(-1, 1)
            alternative = sample_velocity_ddim(
                model, schedule, PARAPHRASES[language][index], guidance=1, **settings,
            ).clamp(-1, 1)
            reference_decision = probe.detailed_score(reference).as_dict(index)
            alternative_decision = probe.detailed_score(alternative).as_dict(index)
            comparison = _compare_waveforms(reference, alternative)
            rows.append({
                "category": category,
                "reference": reference_decision,
                "alternative": alternative_decision,
                "same_category_waveform_mae": comparison["mean_absolute_difference"],
                "identical_same_category_waveforms": comparison["exact_same_waveform"],
            })
            references.append(reference)
            alternatives.append(alternative)
        panels[language] = {
            "cases": rows,
            "summary": _summarize(rows, references, alternatives),
        }
    return {
        "sampling_seed": seed, "zero_guidance_caption_invariant": True,
        "alternative_templates_token_disjoint_from_training": True,
        "same_initial_noise": True, "by_language": panels,
        "general_open_vocabulary_quality_established": False,
    }
