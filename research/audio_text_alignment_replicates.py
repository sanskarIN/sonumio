"""Independent-model-seed bilingual text/audio alignment validation.

The original from-scratch Sonumio audio + UTF-8 text contrastive model
can learn to rank only a small synthetic category vocabulary. A single
training initialization may produce attractive scores by chance.
Replication must use independent network initializations, matched
first-party acoustic recordings, source-group-disjoint InfoNCE batches,
and precisely the same held-out 21-category English/Hindi retrieval
panel. A false reversed-caption control is evaluated on each run.

A model initialization is the replicate. Two captions of one waveform
and several sources in one procedural category are NOT independent
model seeds. Descriptive rankings have no p-values, listener scores,
model-quality winner or Sonumio-1 identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

from research.audio_text_alignment_study import (
    PROTOCOL as SINGLE_PROTOCOL, CATEGORIES,
    _settings, run_study,
)
from research.multi_seed_study import summarize_values
from research.text_audio import SYNTHETIC_COMBINATIONS

PROTOCOL = "sonumio-replicated-bilingual-audio-text-alignment-v1"
MAX_REPLICATES = 4
ALIGNED_LANGUAGES = ("english", "hindi", "wrong_label_cyclic_shift")
METRICS = (
    "recall_at_1", "recall_at_5", "mean_reciprocal_rank",
    "mean_matched_minus_other_cosine",
)


def _sha(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _finite(value: object, *, minimum=None, maximum=None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Invalid nonfinite alignment or retrieval metric")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError("Alignment metric is below its valid range")
    if maximum is not None and number > maximum:
        raise ValueError("Alignment metric is above its valid range")
    return number


def _digest(data: dict) -> str:
    return hashlib.sha256(json.dumps(
        data, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _valid_seeds(seeds: tuple[int, ...]) -> None:
    if (type(seeds) is not tuple
            or not 2 <= len(seeds) <= MAX_REPLICATES
            or len(set(seeds)) != len(seeds)
            or any(type(x) is not int or not 0 <= x < 2**32 - 8192
                   for x in seeds)):
        raise ValueError("Require 2..4 distinct model initialization seeds")


def _model_config(
    *, seed: int, updates: int, batch_categories: int,
    learning_rate: float, temperature: float,
    training_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
) -> dict:
    return {
        "seed": seed, "updates": updates,
        "batch_categories": batch_categories,
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "training_seeds": list(training_seeds),
        "validation_seeds": list(validation_seeds),
        "audio_sample_rate": 8000,
        "audio_frames": 8192,
        "category_combinations": CATEGORIES,
        "audio_encoder": "original_strided_conv_16_channels",
        "text_encoder": "original_utf8_byte_transformer_16_channels",
        "embedding_dim": 32,
        "loss": "category_disjoint_symmetric_infonce_v1",
    }


def _expected_source_group(name: str, seed: int, language: str,
                           seeds: tuple[int, ...]) -> int:
    keys = {
        "-".join(str(value) for value in members): index
        for index, members in enumerate(SYNTHETIC_COMBINATIONS)
    }
    if (name not in keys or seed not in seeds
            or language not in ("en", "hi")):
        raise ValueError("Unknown semantic group, source seed or caption language")
    family = keys[name]
    slot = seeds.index(seed)
    return 2 * (family * len(seeds) + slot) + (0 if language == "en" else 1)


def _validate_retrieval(
    result: dict, *, expected_categories: int,
) -> dict[str, float]:
    if type(result) is not dict or result.get(
        "category_candidates"
    ) != expected_categories:
        raise ValueError("Incomplete heldout retrieval candidates")
    ranks = result.get("rank_per_category")
    if (type(ranks) is not list or len(ranks) != expected_categories
            or any(type(rank) is not int
                   or not 1 <= rank <= expected_categories
                   for rank in ranks)):
        raise ValueError("Missing or invalid per-category retrieval ranks")
    n1 = sum(rank == 1 for rank in ranks)
    n5 = sum(rank <= 5 for rank in ranks)
    if (result.get("rank1_matches") != n1
            or result.get("rank5_matches") != n5
            or result.get("all_candidates_from_synthetic_dsp_only") is not True
            or result.get("english_hindi_same_recording_not_independent") is not True
            or result.get("open_vocabulary_or_audio_quality_established") is not False):
        raise ValueError("Miscounted or falsely interpreted source ranking")
    expected = {
        "recall_at_1": n1 / expected_categories,
        "recall_at_5": n5 / expected_categories,
        "mean_reciprocal_rank":
            sum(1 / rank for rank in ranks) / expected_categories,
    }
    scores = {}
    for metric in METRICS:
        observed = _finite(
            result.get(metric),
            minimum=-2.001 if metric == "mean_matched_minus_other_cosine" else 0,
            maximum=2.001 if metric == "mean_matched_minus_other_cosine" else 1,
        )
        if metric in expected and not math.isclose(
            observed, expected[metric], rel_tol=1e-8, abs_tol=1e-10,
        ):
            raise ValueError("Reported bilingual retrieval ranking math was altered")
        scores[metric] = observed
    return scores


def _validate_run(
    run: dict, *, seed: int, expected: dict,
    previous_audio: tuple | None,
) -> dict:
    """Reject entire run if a source, state, metric or rights check is missing."""
    if type(run) is not dict or run.get("protocol") != SINGLE_PROTOCOL:
        raise ValueError("Wrong individual original bilingual alignment protocol")
    config = run.get("configuration")
    source = run.get("data")
    model = run.get("model")
    train = run.get("training")
    retrieval = run.get("heldout_retrieval")
    if not all(type(x) is dict for x in (
        config, source, model, train, retrieval,
    )):
        raise ValueError("Incomplete paired bilingual model evidence")
    if config != expected or run.get("configuration_sha256") != _digest(expected):
        raise ValueError("Alignment model hyperparameters or configuration drifted")
    if (run.get("original_model_protocol")
            != "sonumio-from-scratch-bilingual-audio-text-alignment-v1"
            or run.get("global_torch_rng_preserved") is not True
            or run.get("no_model_weights_or_audio_saved") is not True
            or run.get("model_publication_authorized") is not False
            or run.get("semantic_generalization_or_perceptual_quality_claim") is not False):
        raise ValueError("Forbidden model status or unknown training protocol")
    count_train = CATEGORIES * len(expected["training_seeds"])
    count_heldout = CATEGORIES * len(expected["validation_seeds"])
    if (source.get("unique_training_recordings") != count_train
            or source.get("unique_heldout_recordings") != count_heldout
            or source.get("training_bilingual_caption_examples") != 2 * count_train
            or source.get("heldout_bilingual_caption_examples") != 2 * count_heldout
            or source.get("train_heldout_exact_audio_overlap") is not False
            or source.get("first_party_procedural_sounds_only") is not True
            or source.get("independent_real_world_categories_proven") is not False
            or not _sha(source.get("train_fingerprint"))
            or not _sha(source.get("validation_fingerprint"))
            or source["train_fingerprint"] == source["validation_fingerprint"]):
        raise ValueError("Wrong bilingual acoustic recording-group coverage")
    hashes = retrieval.get("heldout_category_audio_sha256")
    if (type(hashes) is not list
            or len(hashes) != CATEGORIES or not all(_sha(x) for x in hashes)
            or len(set(hashes)) != CATEGORIES
            or retrieval.get("unique_source_categories") != CATEGORIES
            or retrieval.get("same_audio_candidates_for_both_languages") is not True
            or retrieval.get("category_labels_not_unique_recording_ids") is not True
            or retrieval.get("language_templates_share_sound_categories") is not True
            or retrieval.get("heldout_recordings_used_in_training") is not False
            or retrieval.get("independent_real_world_prompt_semantics_proven") is not False):
        raise ValueError("Bilingual retrieval source panel contains corrupted recordings")
    current_audio = (
        source["train_fingerprint"],
        source["validation_fingerprint"], tuple(hashes),
    )
    if previous_audio is not None and current_audio != previous_audio:
        raise ValueError("Source audio changed between independent model initializations")
    if (not _sha(model.get("initial_sha256"))
            or not _sha(model.get("trained_sha256"))
            or model["initial_sha256"] == model["trained_sha256"]
            or model.get("unchanged_during_evaluation") is not True
            or model.get("weights_published") is not False):
        raise ValueError("Inconsistent original audio/text embedding model state")
    if (train.get("actual_optimizer_steps") != expected["updates"]
            or train.get("same_category_false_negatives_used") is not False
            or train.get("bilingual_audio_recordings_counted_once") is not True):
        raise ValueError("Contrastive training steps or category controls missing")
    losses = [
        _finite(train.get(key), minimum=0) for key in (
            "loss_initial", "loss_final", "loss_min", "loss_max",
        )
    ]
    if (not losses[2] <= losses[0] <= losses[3]
            or not losses[2] <= losses[1] <= losses[3]):
        raise ValueError("Alignment optimizer loss bookkeeping does not match")
    batches = train.get("category_disjoint_minibatches")
    if type(batches) is not list or len(batches) != expected["updates"]:
        raise ValueError("Missing or duplicated semantic category minibatches")
    for batch in batches:
        if type(batch) is not dict:
            raise ValueError("Invalid category-disjoint training batch")
        names = batch.get("source_category_groups")
        recordings = batch.get("selected_unique_recordings")
        if (type(names) is not list
                or len(names) != expected["batch_categories"]
                or len(set(names)) != len(names)
                or type(recordings) is not list
                or len(recordings) != len(names)
                or batch.get("same_category_negative_pairs") != 0):
            raise ValueError("Different recordings of same category used as negatives")
        _finite(batch.get("positive_score"))
        _finite(batch.get("negative_score"))
        for name, row in zip(names, recordings):
            if (type(row) is not dict
                    or row.get("source_category") != name):
                raise ValueError("Category batch provenance changed")
            correct_index = _expected_source_group(
                name, row.get("recording_seed"),
                row.get("caption_language"),
                tuple(expected["training_seeds"]),
            )
            if row.get("recording_index") != correct_index:
                raise ValueError("Bilingual source example index was falsified")
    scores = {
        name: _validate_retrieval(
            retrieval.get(name), expected_categories=CATEGORIES,
        ) for name in ALIGNED_LANGUAGES
    }
    language_cosine = _finite(
        retrieval.get("english_hindi_same_recording_cosine_mean"),
        minimum=-1.0001, maximum=1.0001,
    )
    return {
        "source_identity": current_audio,
        "initial_model_sha256": model["initial_sha256"],
        "trained_model_sha256": model["trained_sha256"],
        "retrieval": scores,
        "bilingual_caption_cosine": language_cosine,
        "loss_first": losses[0], "loss_last": losses[1],
        "config_sha": run["configuration_sha256"],
    }


def replicate_studies(
    *, model_seeds: tuple[int, ...] = (17, 29, 41),
    updates: int = 4, batch_categories: int = 4,
    learning_rate: float = 0.0005, temperature: float = 0.1,
    training_seeds: tuple[int, ...] = (0, 1),
    validation_seeds: tuple[int, ...] = (101, 102),
    runner=run_study,
) -> dict:
    """Perform actual independent training and validate source-level evidence."""
    _valid_seeds(model_seeds)
    _settings(
        seed=model_seeds[0], updates=updates,
        batch_categories=batch_categories,
        learning_rate=learning_rate, temperature=temperature,
        training_seeds=training_seeds,
        validation_seeds=validation_seeds,
    )
    if not callable(runner):
        raise ValueError("A valid deterministic original-study runner is required")
    source_identity = None
    initial_seen = set()
    trained_seen = set()
    records = []
    for seed in model_seeds:
        run = runner(
            seed=seed, updates=updates,
            batch_categories=batch_categories,
            learning_rate=learning_rate,
            temperature=temperature,
            training_seeds=training_seeds,
            validation_seeds=validation_seeds,
        )
        checked = _validate_run(
            run, seed=seed,
            expected=_model_config(
                seed=seed, updates=updates,
                batch_categories=batch_categories,
                learning_rate=learning_rate,
                temperature=temperature,
                training_seeds=training_seeds,
                validation_seeds=validation_seeds,
            ), previous_audio=source_identity,
        )
        if source_identity is None:
            source_identity = checked["source_identity"]
        if checked["initial_model_sha256"] in initial_seen:
            raise ValueError("Independent model seeds reused the same initialization")
        if checked["trained_model_sha256"] in trained_seen:
            raise ValueError("Distinct model seeds reused the same trained parameters")
        initial_seen.add(checked["initial_model_sha256"])
        trained_seen.add(checked["trained_model_sha256"])
        records.append({
            "model_seed": seed,
            "initial_model_sha256": checked["initial_model_sha256"],
            "trained_model_sha256": checked["trained_model_sha256"],
            "configuration_sha256": checked["config_sha"],
            "training_loss_first": checked["loss_first"],
            "training_loss_last": checked["loss_last"],
            "per_language_retrieval": checked["retrieval"],
            "same_recording_english_hindi_text_cosine":
                checked["bilingual_caption_cosine"],
        })
    summaries = {
        language: {
            metric: summarize_values([
                record["per_language_retrieval"][language][metric]
                for record in records
            ])
            for metric in METRICS
        }
        for language in ALIGNED_LANGUAGES
    }
    summaries["cross_language_same_recording_cosine"] = summarize_values([
        record["same_recording_english_hindi_text_cosine"]
        for record in records
    ])
    cfg = {
        "model_seeds": list(model_seeds),
        "updates": updates, "batch_categories": batch_categories,
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "training_seeds": list(training_seeds),
        "validation_seeds": list(validation_seeds),
        "train_fingerprint": source_identity[0],
        "validation_fingerprint": source_identity[1],
        "single_training_protocol": SINGLE_PROTOCOL,
        "source_categories": CATEGORIES,
    }
    return {
        "protocol": PROTOCOL,
        "configuration": cfg,
        "configuration_sha256": _digest(cfg),
        "real_independent_model_training_runs": records,
        "independent_model_initializations": len(records),
        "source_sha256_and_heldout_candidate_panel_matched_across_runs": True,
        "unique_initial_and_final_model_sha256_validated": True,
        "category_disjoint_info_nce_false_negatives_rejected": True,
        "one_heldout_recording_per_category_and_language_verified": True,
        "same_audio_used_for_english_and_hindi_retrieval": True,
        "distinct_caption_tokens_not_extra_audio_recordings": True,
        "descriptive_between_initialization_retrieval": summaries,
        "statistical_significance_or_open_vocabulary_semantics_established": False,
        "human_listening_or_audio_generation_quality_established": False,
        "no_audio_embeddings_or_model_weights_written": True,
        "model_publication_authorized": False,
        "warning": (
            "Retrieval across 21 related synthetic first-party sound "
            "categories is not natural-language understanding or usable "
            "generative audio. Independently initialized model seeds are "
            "the only true neural replicates in this report."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seeds", default="17,29,41")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--batch-categories", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--training-seeds", default="0,1")
    parser.add_argument("--validation-seeds", default="101,102")
    opts = parser.parse_args()
    result = replicate_studies(
        model_seeds=tuple(int(x) for x in opts.model_seeds.split(",")),
        updates=opts.updates,
        batch_categories=opts.batch_categories,
        learning_rate=opts.learning_rate,
        temperature=opts.temperature,
        training_seeds=tuple(
            int(x) for x in opts.training_seeds.split(",")
        ),
        validation_seeds=tuple(
            int(x) for x in opts.validation_seeds.split(",")
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
