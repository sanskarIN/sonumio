"""Actual first-party bilingual audio/text alignment training and retrieval.

No CLAP, pretrained embeddings, external audio corpora, user recordings or
model artifacts. This original small contrastive model learns only on
Sonumio's six synthetic DSP categories and their 15 two-class mixtures.

Training batches have a maximum of one example PER SOUND-CATEGORY
COMBINATION. Different English/Hindi texts or recordings for the SAME
category never become false negatives. Evaluation tests one held-out
recording per category in both languages using one shared acoustic set.
The resulting ranking is narrow and NOT general semantic comprehension.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import torch
from torch import nn

from research.architecture_compare import _audio_hashes
from research.audio_text_alignment import (
    OriginalAudioTextAligner, PROTOCOL as MODEL_PROTOCOL,
    category_disjoint_info_nce, retrieval_metrics,
)
from research.latent_audio import (
    TRAIN_SEEDS, VALIDATION_SEEDS, synthetic_bilingual_waves,
)
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-first-party-bilingual-audio-text-training-study-v1"
CATEGORIES = len(SYNTHETIC_COMBINATIONS)
MAX_TRAIN_UPDATES = 64


def _settings(*, seed: int, updates: int, batch_categories: int,
              learning_rate: float, temperature: float,
              training_seeds: tuple[int, ...],
              validation_seeds: tuple[int, ...]) -> None:
    if (type(seed) is not int or not 0 <= seed < 2**32 - 8192
            or type(updates) is not int or not 1 <= updates <= MAX_TRAIN_UPDATES
            or type(batch_categories) is not int
            or not 2 <= batch_categories <= 16
            or type(learning_rate) not in (float, int)
            or not math.isfinite(learning_rate)
            or not 1e-6 <= learning_rate <= 0.01
            or type(temperature) not in (float, int)
            or not math.isfinite(temperature)
            or not 0.02 <= temperature <= 1):
        raise ValueError("Invalid bounded bilingual audio-text research settings")
    if (type(training_seeds) is not tuple
            or type(validation_seeds) is not tuple
            or not 1 <= len(training_seeds) <= 8
            or not 1 <= len(validation_seeds) <= 8
            or len(set(training_seeds)) != len(training_seeds)
            or len(set(validation_seeds)) != len(validation_seeds)
            or set(training_seeds) & set(validation_seeds)
            or any(type(s) is not int or not 0 <= s < 2**32
                   for s in training_seeds + validation_seeds)):
        raise ValueError("Training and held-out acoustic recording seeds must be disjoint")


def _group_key(members: tuple[int, ...]) -> str:
    if members not in SYNTHETIC_COMBINATIONS:
        raise ValueError("Unknown first-party sound category")
    return "-".join(str(x) for x in members)


def _training_indices(
    *, batch_categories: int, recording_seeds: tuple[int, ...],
    generator: torch.Generator,
) -> tuple[torch.Tensor, tuple[str, ...], list[dict]]:
    """Pick different category combinations; EN/HI count as same sound."""
    if (not isinstance(generator, torch.Generator)
            or type(batch_categories) is not int
            or not 2 <= batch_categories <= 16
            or type(recording_seeds) is not tuple
            or not recording_seeds):
        raise ValueError("Invalid category-disjoint text/audio minibatch")
    families = torch.randperm(
        CATEGORIES, generator=generator,
    )[:batch_categories].tolist()
    selected = []
    keys = []
    provenance = []
    for family in families:
        seed_slot = int(torch.randint(
            len(recording_seeds), (1,), generator=generator,
        ))
        language = int(torch.randint(2, (1,), generator=generator))
        # Synthetic DSP dataset loops combinations -> seed -> (en, hi)
        example_index = family * len(recording_seeds) * 2 + 2 * seed_slot + language
        selected.append(example_index)
        members = SYNTHETIC_COMBINATIONS[family]
        name = _group_key(members)
        keys.append(name)
        provenance.append({
            "source_category": name,
            "recording_seed": recording_seeds[seed_slot],
            "caption_language": "en" if language == 0 else "hi",
            "recording_index": example_index,
        })
    if len(set(keys)) != batch_categories or len(set(selected)) != batch_categories:
        raise ValueError("Duplicate category was chosen as an InfoNCE negative")
    return torch.tensor(selected, dtype=torch.long), tuple(keys), provenance


def _encode_in_batches(
    model: OriginalAudioTextAligner,
    audio: torch.Tensor, english: torch.Tensor,
    hindi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode all 21 held-out categories without relaxing forward batch cap."""
    if (len(audio) != CATEGORIES
            or english.shape[0] != CATEGORIES
            or hindi.shape[0] != CATEGORIES):
        raise ValueError("Incomplete bilingual held-out retrieval panel")
    a, e, h = [], [], []
    for offset in range(0, CATEGORIES, 16):
        a.append(model.encode_audio(audio[offset:offset + 16]))
        e.append(model.encode_text(english[offset:offset + 16]))
        h.append(model.encode_text(hindi[offset:offset + 16]))
    return torch.cat(a), torch.cat(e), torch.cat(h)


@torch.no_grad()
def _heldout(
    model: OriginalAudioTextAligner, validation,
    validation_seeds: tuple[int, ...],
) -> dict:
    """One actual 8-kHz recording per source category, two caption languages."""
    if validation.examples != CATEGORIES * len(validation_seeds) * 2:
        raise ValueError("Expected one bilingual pair for every sound family")
    model.eval()
    audio = []
    en = []
    hi = []
    hashes = []
    for family in range(CATEGORIES):
        # First held-out recording seed for each category.
        index = family * len(validation_seeds) * 2
        left = validation.waves[index]
        duplicate = validation.waves[index + 1]
        if not torch.equal(left, duplicate):
            raise ValueError("Bilingual captions must reference identical recorded audio")
        audio.append(left)
        en.append(validation.tokens[index])
        hi.append(validation.tokens[index + 1])
        hashes.append(hashlib.sha256(
            left.numpy().tobytes()
        ).hexdigest())
    if len(set(hashes)) != CATEGORIES:
        raise ValueError("Held-out category candidate audio is duplicated")
    waveforms = torch.stack(audio)
    english = torch.stack(en)
    hindi = torch.stack(hi)
    audio_embeddings, en_embeddings, hi_embeddings = _encode_in_batches(
        model, waveforms, english, hindi,
    )
    keys = tuple(_group_key(m) for m in SYNTHETIC_COMBINATIONS)
    metrics_en = retrieval_metrics(audio_embeddings, en_embeddings, keys)
    metrics_hi = retrieval_metrics(audio_embeddings, hi_embeddings, keys)
    # This sham diagnostic cyclically mismatches every text category.
    # It is a deliberately WRONG label control, not another trained model.
    wrong = retrieval_metrics(
        audio_embeddings, en_embeddings.roll(1, 0), keys,
    )
    # Cross-language paired caption distance, same audio source.
    language_cosine = (en_embeddings * hi_embeddings).sum(-1)
    if not torch.isfinite(language_cosine).all():
        raise ValueError("Nonfinite bilingual audio/text alignment embedding")
    return {
        "english": metrics_en,
        "hindi": metrics_hi,
        "wrong_label_cyclic_shift": wrong,
        "english_hindi_same_recording_cosine_mean":
            float(language_cosine.mean()),
        "unique_source_categories": CATEGORIES,
        "heldout_category_audio_sha256": hashes,
        "same_audio_candidates_for_both_languages": True,
        "category_labels_not_unique_recording_ids": True,
        "language_templates_share_sound_categories": True,
        "heldout_recordings_used_in_training": False,
        "independent_real_world_prompt_semantics_proven": False,
    }


def run_study(
    *, seed: int = 17, updates: int = 4,
    batch_categories: int = 4,
    learning_rate: float = 0.0005, temperature: float = 0.1,
    training_seeds: tuple[int, ...] = TRAIN_SEEDS,
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS,
) -> dict:
    _settings(
        seed=seed, updates=updates,
        batch_categories=batch_categories,
        learning_rate=learning_rate, temperature=temperature,
        training_seeds=training_seeds,
        validation_seeds=validation_seeds,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        training = synthetic_bilingual_waves(training_seeds)
        validation = synthetic_bilingual_waves(validation_seeds)
        train_hashes = _audio_hashes(training.waves)
        val_hashes = _audio_hashes(validation.waves)
        if (train_hashes & val_hashes
                or training.fingerprint == validation.fingerprint
                or len(train_hashes) != CATEGORIES * len(training_seeds)
                or len(val_hashes) != CATEGORIES * len(validation_seeds)):
            raise ValueError("Audio/text source split leakage or duplicate source")
        network = OriginalAudioTextAligner(
            channels=16, embedding_dim=32,
        )
        initial_sha = model_state_sha256(network)
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=learning_rate,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        losses = []
        train_diag = []
        for _ in range(updates):
            indices, categories, provenance = _training_indices(
                batch_categories=batch_categories,
                recording_seeds=training_seeds,
                generator=generator,
            )
            clip = training.waves[indices]
            tokens = training.tokens[indices]
            network.train()
            optimizer.zero_grad(set_to_none=True)
            audio_embedding, text_embedding = network(clip, tokens)
            loss, diagnostic = category_disjoint_info_nce(
                audio_embedding, text_embedding, categories,
                temperature=temperature,
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite first-party contrastive training loss")
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=1,
                error_if_nonfinite=True,
            )
            if not torch.isfinite(gradient):
                raise ValueError("Nonfinite audio/text alignment model gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            train_diag.append({
                "source_category_groups": list(categories),
                "selected_unique_recordings": provenance,
                "positive_score": diagnostic["positive_similarity_mean"],
                "negative_score": diagnostic["off_diagonal_similarity_mean"],
                "same_category_negative_pairs": 0,
            })
        trained_sha = model_state_sha256(network)
        if trained_sha == initial_sha:
            raise ValueError("Contrastive research model weights never updated")
        metrics = _heldout(network, validation, validation_seeds)
        if model_state_sha256(network) != trained_sha:
            raise ValueError("Retrieval evaluation modified trained parameters")
        settings = {
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
        digest = hashlib.sha256(
            json.dumps(settings, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        return {
            "protocol": PROTOCOL,
            "original_model_protocol": MODEL_PROTOCOL,
            "configuration": settings,
            "configuration_sha256": digest,
            "data": {
                "train_fingerprint": training.fingerprint,
                "validation_fingerprint": validation.fingerprint,
                "unique_training_recordings": len(train_hashes),
                "unique_heldout_recordings": len(val_hashes),
                "training_bilingual_caption_examples": training.examples,
                "heldout_bilingual_caption_examples": validation.examples,
                "train_heldout_exact_audio_overlap": False,
                "first_party_procedural_sounds_only": True,
                "independent_real_world_categories_proven": False,
            },
            "model": {
                "initial_sha256": initial_sha,
                "trained_sha256": trained_sha,
                "unchanged_during_evaluation": True,
                "weights_published": False,
            },
            "training": {
                "loss_initial": losses[0],
                "loss_final": losses[-1],
                "loss_min": float(min(losses)),
                "loss_max": float(max(losses)),
                "actual_optimizer_steps": len(losses),
                "category_disjoint_minibatches": train_diag,
                "same_category_false_negatives_used": False,
                "bilingual_audio_recordings_counted_once": True,
            },
            "heldout_retrieval": metrics,
            "global_torch_rng_preserved": True,
            "no_model_weights_or_audio_saved": True,
            "model_publication_authorized": False,
            "semantic_generalization_or_perceptual_quality_claim": False,
            "warning": (
                "Only template-category retrieval on synthetic DSP. "
                "A positive matched score or retrieval rank does not "
                "prove natural prompt understanding, real sound quality, "
                "speech, music, originality, or release readiness."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--batch-categories", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--training-seeds", default="0,1")
    parser.add_argument("--validation-seeds", default="101,102")
    args = parser.parse_args()
    report = run_study(
        seed=args.seed, updates=args.updates,
        batch_categories=args.batch_categories,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        training_seeds=tuple(
            int(x) for x in args.training_seeds.split(",")
        ),
        validation_seeds=tuple(
            int(x) for x in args.validation_seeds.split(",")
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
