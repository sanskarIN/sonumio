"""Compositional text/audio controls for original Sonumio DSP sound mixtures.

The project's 21 synthetic sound groups are six atomic categories plus
their 15 unordered pairs. 'Rain' and 'rain with wind' are related, not
independent unrelated negative examples; but they are NOT identical
positive recordings either.

This module supplies:
1. Audited exact / partially shared / disjoint sound-group relations;
2. A symmetric InfoNCE variant that masks *partial-overlap* groups
   from the denominator rather than pushing related concepts apart;
3. A source-group-aware 21-category similarity/retrieval diagnostic
   separating exact, partial and disjoint matches in English/Hindi;
4. A no-pretrained-model in-memory held-out evaluation of the original
   learned audio and UTF-8 text towers on the same audio recordings.

The semantic groups are only first-party procedural DSP generation
metadata. They are NOT proof of general natural-language understanding
or perceptually satisfactory music/speech generation.
"""
from __future__ import annotations

import hashlib
import json
import math
from statistics import fmean

import torch
from torch.nn import functional as F

from research.audio_text_alignment import OriginalAudioTextAligner
from research.audio_text_alignment_study import _encode_in_batches
from research.latent_audio import FRAMES, synthetic_bilingual_waves
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-compositional-bilingual-audio-text-evaluation-v1"
LOSS_PROTOCOL = "sonumio-overlap-masked-symmetric-infonce-v1"
RELATIONSHIP_PROTOCOL = "sonumio-original-dsp-source-composition-v1"
SOUND_COUNT = 6
MAX_GROUPS = 21
MIN_GROUPS = 2
RELATIONS = ("exact", "partially_shared", "disjoint")
SOURCE_GROUPS = tuple(SYNTHETIC_COMBINATIONS)
METRICS = (
    "exact_similarity",
    "partially_shared_similarity",
    "disjoint_similarity",
    "exact_minus_disjoint",
    "partial_minus_disjoint",
    "exact_minus_partial",
)


def _groups(groups: tuple[tuple[int, ...], ...]) -> None:
    if (type(groups) is not tuple
            or not MIN_GROUPS <= len(groups) <= MAX_GROUPS
            or any(type(members) is not tuple
                   or not 1 <= len(members) <= 2
                   or any(type(x) is not int or not 0 <= x < SOUND_COUNT
                          for x in members)
                   or members != tuple(sorted(set(members)))
                   or members not in SOURCE_GROUPS
                   for members in groups)
            or len(set(groups)) != len(groups)):
        raise ValueError(
            "Require unique first-party DSP sound categories or 2-type mixtures"
        )


def relation(
    left: tuple[int, ...], right: tuple[int, ...],
) -> str:
    _groups((left, right)) if left != right else _groups(
        (left, tuple(x for x in range(SOUND_COUNT) if x not in left)[:1])
    )
    if left == right:
        return "exact"
    if set(left) & set(right):
        return "partially_shared"
    return "disjoint"


def category_relation_matrix(
    groups: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, dict]:
    """Each ordered category pair has a deterministic 0/1/2 relation."""
    _groups(groups)
    n = len(groups)
    matrix = torch.zeros((n, n), dtype=torch.int8)
    totals = {name: 0 for name in RELATIONS}
    source_keys = ["-".join(map(str, group)) for group in groups]
    for index, left in enumerate(groups):
        for j, right in enumerate(groups):
            if left == right:
                code, name = 2, "exact"
            elif set(left) & set(right):
                code, name = 1, "partially_shared"
            else:
                code, name = 0, "disjoint"
            matrix[index, j] = code
            totals[name] += 1
    if (totals["exact"] != n
            or totals["exact"] + totals["partially_shared"]
            + totals["disjoint"] != n * n):
        raise ValueError("Unstable original DSP category relation topology")
    return matrix, {
        "protocol": RELATIONSHIP_PROTOCOL,
        "sound_combination_groups": source_keys,
        "source_category_ids": [list(members) for members in groups],
        "number_of_groups": n,
        "ordered_exact_relations": totals["exact"],
        "ordered_partially_shared_relations": totals["partially_shared"],
        "ordered_disjoint_relations": totals["disjoint"],
        "each_group_exactly_matches_only_itself": True,
        "partially_shared_categories_not_identical_positive_recordings": True,
        "structural_dsp_group_metadata_only_not_human_semantics": True,
    }


def _embeddings(
    audio: torch.Tensor, text: torch.Tensor,
    *, maximum: int,
) -> None:
    if (not isinstance(audio, torch.Tensor)
            or not isinstance(text, torch.Tensor)
            or audio.ndim != 2 or audio.shape != text.shape
            or not 2 <= len(audio) <= maximum
            or audio.device.type != "cpu"
            or text.device.type != "cpu"
            or audio.dtype != torch.float32
            or text.dtype != torch.float32
            or not torch.isfinite(audio).all()
            or not torch.isfinite(text).all()):
        raise ValueError("Compositional metric requires paired finite float32 CPU embeddings")


def overlap_masked_info_nce(
    audio_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    groups: tuple[tuple[int, ...], ...], *,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Symmetric InfoNCE: matched positive + truly disjoint negatives only.

    Pairs of *partially overlapping* source categories are deliberately
    not treated as negatives, because both can legitimately contain a
    shared source sound. They are not promoted to exact positives.
    Each row must have at least one truly disjoint negative category.
    This is not a universal semantic classifier or perceptual metric.
    """
    _embeddings(audio_embeddings, text_embeddings, maximum=MAX_GROUPS)
    if (type(groups) is not tuple or len(groups) != len(audio_embeddings)
            or type(temperature) not in (float, int)
            or not math.isfinite(temperature)
            or not 0.02 <= temperature <= 1):
        raise ValueError("Invalid category-disjoint InfoNCE temperature or groups")
    relationship, summary = category_relation_matrix(groups)
    exact = relationship == 2
    disjoint = relationship == 0
    partial = relationship == 1
    if (not torch.equal(exact, torch.eye(
            len(groups), dtype=torch.bool))
            or not disjoint.any(dim=1).all()
            or not disjoint.any(dim=0).all()):
        raise ValueError(
            "A compositional contrastive pair needs one nonoverlapping negative per group"
        )
    a = F.normalize(audio_embeddings, dim=-1, eps=1e-8)
    t = F.normalize(text_embeddings, dim=-1, eps=1e-8)
    scores = a @ t.T / float(temperature)
    permitted = exact | disjoint
    masked = scores.masked_fill(~permitted, float("-inf"))
    positive = torch.arange(len(groups), dtype=torch.long)
    loss = (
        F.cross_entropy(masked, positive)
        + F.cross_entropy(masked.T, positive)
    ) / 2
    if not torch.isfinite(loss) or loss < 0:
        raise ValueError("Invalid or nonfinite compositional loss")
    with torch.no_grad():
        original = (
            F.cross_entropy(scores, positive)
            + F.cross_entropy(scores.T, positive)
        ) / 2
        positive_cosine = torch.diagonal(a @ t.T)
        distant_cosine = (a @ t.T)[disjoint]
        partial_cosine = (a @ t.T)[partial]
        numeric = {
            "protocol": LOSS_PROTOCOL,
            "category_groups": summary["sound_combination_groups"],
            "groups": len(groups),
            "exact_positive_edges": int(exact.sum()),
            "ignored_partially_shared_negative_edges": int(partial.sum()),
            "true_disjoint_negative_edges": int(disjoint.sum()),
            "mean_disjoint_negative_count_per_anchor": float(
                disjoint.sum(dim=1).float().mean()
            ),
            "masked_symmetric_info_nce": float(loss),
            "ordinary_symmetric_info_nce_with_partials": float(original),
            "mean_exact_positive_cosine": float(positive_cosine.mean()),
            "mean_true_disjoint_cosine": float(distant_cosine.mean()),
            "mean_related_cosine": (
                float(partial_cosine.mean())
                if len(partial_cosine) > 0 else None
            ),
            "partial_overlap_groups_used_as_negatives": False,
            "same_group_english_hindi_duplicates_used_as_negatives": False,
            "no_external_pretrained_semantic_model": True,
            "audio_quality_or_open_vocabulary_understanding_proven": False,
        }
    for key in (
        "masked_symmetric_info_nce",
        "ordinary_symmetric_info_nce_with_partials",
        "mean_exact_positive_cosine",
        "mean_true_disjoint_cosine",
    ):
        if not math.isfinite(numeric[key]):
            raise ValueError("Nonfinite overlap-aware contrastive research diagnostic")
    if (numeric["mean_related_cosine"] is not None
            and not math.isfinite(numeric["mean_related_cosine"])):
        raise ValueError("Invalid related-sound comparison")
    return loss, numeric


@torch.no_grad()
def compositional_retrieval(
    audio_features: torch.Tensor,
    text_features: torch.Tensor,
    groups: tuple[tuple[int, ...], ...],
) -> dict:
    """Per-source exact/partial/disjoint embeddings without human quality claims."""
    _embeddings(audio_features, text_features, maximum=MAX_GROUPS)
    if type(groups) is not tuple or len(groups) != len(audio_features):
        raise ValueError("One source group per aligned embedding pair required")
    relationship, _ = category_relation_matrix(groups)
    a = F.normalize(audio_features, dim=-1, eps=1e-8)
    t = F.normalize(text_features, dim=-1, eps=1e-8)
    cosine = a @ t.T
    per_source = []
    for index, members in enumerate(groups):
        row = cosine[index]
        exact = float(row[index])
        partial = row[relationship[index] == 1]
        disjoint = row[relationship[index] == 0]
        if disjoint.numel() < 1:
            raise ValueError("Compositional retrieval needs a disjoint negative")
        if partial.numel() < 1:
            # Not all minimal group panels have partial relationships.
            partial_mean = None
        else:
            partial_mean = float(partial.mean())
        disjoint_mean = float(disjoint.mean())
        order = sorted(
            range(len(groups)),
            key=lambda position: (-float(row[position]), position),
        )
        rank = order.index(index) + 1
        candidate = {
            "source_group_id": "-".join(map(str, members)),
            "source_category_ids": list(members),
            "exact_rank_in_complete_category_panel": rank,
            "exact_similarity": exact,
            "partially_shared_similarity": partial_mean,
            "disjoint_similarity": disjoint_mean,
            "exact_minus_disjoint": exact - disjoint_mean,
            "partial_minus_disjoint": (
                partial_mean - disjoint_mean
                if partial_mean is not None else None
            ),
            "exact_minus_partial": (
                exact - partial_mean
                if partial_mean is not None else None
            ),
            "partially_shared_candidate_count": int(
                (relationship[index] == 1).sum()
            ),
            "disjoint_candidate_count": int(
                (relationship[index] == 0).sum()
            ),
            "exact_candidate_count": 1,
            "related_categories_are_neither_false_exact_matches_nor_disjoint_negatives": True,
        }
        for name in METRICS:
            value = candidate[name]
            if value is not None and not math.isfinite(value):
                raise ValueError("Nonfinite compositional related-source score")
        per_source.append(candidate)

    overview = {}
    for metric in METRICS:
        values = [
            row[metric] for row in per_source
            if row[metric] is not None
        ]
        overview[metric] = {
            "groups_available": len(values),
            "mean": fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    rank1 = sum(row["exact_rank_in_complete_category_panel"] == 1
                for row in per_source)
    return {
        "protocol": PROTOCOL,
        "candidate_categories": len(groups),
        "category_groups": [
            "-".join(map(str, members)) for members in groups
        ],
        "exact_rank_one_count": rank1,
        "exact_rank_one_fraction": rank1 / len(groups),
        "per_acoustic_category": per_source,
        "relationship_descriptive": overview,
        "groups_with_partial_relations": sum(
            row["partially_shared_candidate_count"] > 0
            for row in per_source
        ),
        "every_category_has_disjoint_negative": True,
        "no_external_pretrained_sound_embeddings": True,
        "semantic_accuracy_or_perceptual_quality_proven": False,
    }


@torch.no_grad()
def evaluate_bilingual_composition(
    model: OriginalAudioTextAligner, *,
    validation_seeds: tuple[int, ...] = (101,),
) -> dict:
    """Evaluate same 21 acoustic recordings in both English/Hindi languages.

    No model training or inference checkpoint is created; this is an
    objective function usable for future original, genuinely trained
    audio/text encoders. A random model's ranking is not informative.
    """
    if (not isinstance(model, OriginalAudioTextAligner)
            or next(model.parameters()).device.type != "cpu"
            or type(validation_seeds) is not tuple
            or not 1 <= len(validation_seeds) <= 4
            or any(type(x) is not int or not 0 <= x < 2**32
                   for x in validation_seeds)
            or len(set(validation_seeds)) != len(validation_seeds)):
        raise ValueError("Original CPU model and distinct bounded evaluation seeds required")
    source = synthetic_bilingual_waves(validation_seeds)
    expected = len(SOURCE_GROUPS) * len(validation_seeds) * 2
    if source.examples != expected or source.waves.shape[0] != expected:
        raise ValueError("Incomplete original bilingual acoustic dataset")
    audio, english, hindi, hashes = [], [], [], []
    for family in range(len(SOURCE_GROUPS)):
        position = family * len(validation_seeds) * 2
        original = source.waves[position]
        duplicate = source.waves[position + 1]
        if not torch.equal(original, duplicate):
            raise ValueError("Bilingual captions lost exact sound recording correspondence")
        audio.append(original)
        english.append(source.tokens[position])
        hindi.append(source.tokens[position + 1])
        hashes.append(hashlib.sha256(
            original.contiguous().numpy().tobytes()
        ).hexdigest())
    if len(set(hashes)) != len(SOURCE_GROUPS):
        raise ValueError("Original category panel contains duplicate source audio")
    clip = torch.stack(audio)
    en_tokens = torch.stack(english)
    hi_tokens = torch.stack(hindi)
    initial = model_state_sha256(model)
    rng = torch.random.get_rng_state().clone()
    mode = model.training
    model.eval()
    try:
        a, en, hi = _encode_in_batches(
            model, clip, en_tokens, hi_tokens,
        )
        en_scores = compositional_retrieval(a, en, SOURCE_GROUPS)
        hi_scores = compositional_retrieval(a, hi, SOURCE_GROUPS)
        wrong_labels = compositional_retrieval(
            a, en.roll(shifts=1, dims=0), SOURCE_GROUPS,
        )
        bilingual = (F.normalize(en, dim=-1) * F.normalize(
            hi, dim=-1)).sum(dim=-1)
        if (model_state_sha256(model) != initial
                or not torch.equal(rng, torch.random.get_rng_state())
                or not torch.isfinite(bilingual).all()):
            raise ValueError("Bilingual compositional inference mutated state")
        config = {
            "validation_seeds": list(validation_seeds),
            "source_data_fingerprint": source.fingerprint,
            "source_categories": len(SOURCE_GROUPS),
            "recording_sha256": hashes,
            "same_audio_recordings_for_both_languages": True,
        }
        return {
            "protocol": PROTOCOL,
            "configuration": config,
            "configuration_sha256": hashlib.sha256(json.dumps(
                config, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "english": en_scores,
            "hindi": hi_scores,
            "wrong_caption_category_control": wrong_labels,
            "same_category_english_hindi_caption_cosine_mean": float(
                bilingual.mean()
            ),
            "model_state_unchanged": True,
            "global_torch_rng_unchanged": True,
            "independent_source_recordings_in_candidate_panel":
                len(SOURCE_GROUPS),
            "two_languages_not_two_independent_recordings": True,
            "existing_model_weights_not_saved_or_released": True,
            "perceptual_or_open_vocabulary_prompt_alignment_claim": False,
            "warning": (
                "The source-composition lattice is a heuristic based on "
                "21 synthetic DSP category IDs. Partial semantic overlap "
                "is not human-reviewed natural-language equivalence. "
                "A random encoder is not a useful multilingual sound model."
            ),
        }
    finally:
        model.train(mode)
        if not torch.equal(rng, torch.random.get_rng_state()):
            torch.random.set_rng_state(rng)
