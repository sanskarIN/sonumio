"""Matched old-vs-multiresolution codec training on first-party synthetic DSP.

Two original Sonumio tiny neural codecs begin with EXACTLY the same
parameters. Both use the same acoustic training windows, batches,
optimizer schedule and gradient clipping. The ONLY research difference
is original waveform/STFT loss versus explicit multiresolution transient
and highband losses. Evaluation uses separate seed-disjoint acoustic
recordings and the SAME numerical objective for both variants.

The report is numerical-only. Short training cannot establish that a
codec sounds better or that speech/music/fidelity is release quality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from statistics import fmean

import torch
from torch import nn

from research.codec_objectives import (
    CodecObjectiveWeights, CodecSpectralConfig,
    COMPONENT_NAMES, PROTOCOL as LOSS_PROTOCOL,
    describe_codec_error, weighted_codec_loss,
)
from research.latent_audio import (
    FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS,
    TinyAudioAutoencoder, reconstruction_loss,
    synthetic_bilingual_waves,
)
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-paired-neural-codec-loss-study-v1"
ARMS = ("original_loss", "multiresolution_loss")
MAX_STEPS = 64


def _settings(*, model_seed: int, updates: int,
              training_batch: int,
              learning_rate: float,
              training_seeds: tuple[int, ...],
              heldout_seeds: tuple[int, ...]) -> None:
    if (type(model_seed) is not int or not 0 <= model_seed < 2**32 - 500
            or type(updates) is not int or not 1 <= updates <= MAX_STEPS
            or type(training_batch) is not int or not 1 <= training_batch <= 8
            or type(learning_rate) not in (float, int)
            or not math.isfinite(learning_rate)
            or not 1e-6 <= learning_rate <= 0.01):
        raise ValueError("Invalid original neural codec objective study settings")
    if (type(training_seeds) is not tuple
            or type(heldout_seeds) is not tuple
            or not 1 <= len(training_seeds) <= 8
            or not 1 <= len(heldout_seeds) <= 8
            or len(set(training_seeds)) != len(training_seeds)
            or len(set(heldout_seeds)) != len(heldout_seeds)
            or set(training_seeds) & set(heldout_seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32
                   for seed in training_seeds + heldout_seeds)):
        raise ValueError("Require disjoint, unique first-party acoustic seeds")


def _audio_sha(audio: torch.Tensor) -> str:
    if (not isinstance(audio, torch.Tensor) or audio.shape != (1, FRAMES)
            or audio.device.type != "cpu"
            or audio.dtype != torch.float32
            or not torch.isfinite(audio).all()):
        raise ValueError("Cannot fingerprint invalid codec reference")
    return hashlib.sha256(audio.detach().numpy().tobytes()).hexdigest()


def _source_unique(sources, seeds: tuple[int, ...]):
    """Validate bilingual duplicates; return one recording per audio source."""
    if (sources.examples != 2 * len(SYNTHETIC_COMBINATIONS) * len(seeds)
            or sources.waves.ndim != 3 or sources.waves.shape[1:] != (1, FRAMES)
            or sources.tokens.shape[0] != sources.examples
            or not torch.equal(sources.waves[::2], sources.waves[1::2])):
        raise ValueError("Codec source does not preserve bilingual audio groups")
    unique = sources.waves[::2].contiguous()
    fingerprints = [_audio_sha(clip) for clip in unique]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Duplicate unique recordings in codec study")
    return unique, fingerprints


@torch.no_grad()
def _evaluate_models(models: dict, heldout: torch.Tensor,
                     recorded_hashes: list[str],
                     heldout_seeds: tuple[int, ...]) -> dict:
    """Inspect each independent recording, never duplicate bilingual audio."""
    if (set(models) != set(ARMS)
            or len(heldout) != len(recorded_hashes)
            or not len(heldout) == len(SYNTHETIC_COMBINATIONS) * len(heldout_seeds)):
        raise ValueError("Incomplete held-out codec model set or acoustic split")
    for model in models.values():
        model.eval()
    component_sums = {
        arm: {metric: 0.0 for metric in COMPONENT_NAMES}
        for arm in ARMS
    }
    mean_metrics = {
        arm: {"sample_aligned_mae": 0.0, "sample_aligned_rmse": 0.0,
              "snr_db": 0.0}
        for arm in ARMS
    }
    per_source = []
    source_group_delta = {
        "/".join(str(n) for n in members): {metric: [] for metric in COMPONENT_NAMES}
        for members in SYNTHETIC_COMBINATIONS
    }
    for index in range(len(heldout)):
        original = heldout[index:index + 1]
        measurements = {}
        group_index = index // len(heldout_seeds)
        group = SYNTHETIC_COMBINATIONS[group_index]
        name = "/".join(str(n) for n in group)
        for arm in ARMS:
            decoded = models[arm](original)
            if (decoded.shape != original.shape
                    or decoded.dtype != torch.float32
                    or not torch.isfinite(decoded).all()):
                raise ValueError("Nonfinite neural codec heldout output")
            report = describe_codec_error(original, decoded)
            measurements[arm] = report
            for metric in COMPONENT_NAMES:
                value = report["components"][metric]
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Nonfinite per-recording codec component")
                component_sums[arm][metric] += value
            for metric in mean_metrics[arm]:
                mean_metrics[arm][metric] += report[metric]
        deltas = {
            metric: measurements["multiresolution_loss"]["components"][metric]
                    - measurements["original_loss"]["components"][metric]
            for metric in COMPONENT_NAMES
        }
        for metric, delta in deltas.items():
            source_group_delta[name][metric].append(delta)
        per_source.append({
            "acoustic_group": name,
            "recording_seed": heldout_seeds[index % len(heldout_seeds)],
            "heldout_sha256": recorded_hashes[index],
            "sample_aligned_mae": {
                arm: measurements[arm]["sample_aligned_mae"]
                for arm in ARMS
            },
            "sample_aligned_rmse": {
                arm: measurements[arm]["sample_aligned_rmse"]
                for arm in ARMS
            },
            "sample_aligned_snr_db": {
                arm: measurements[arm]["snr_db"]
                for arm in ARMS
            },
            "multiresolution_minus_original_components": deltas,
            "independent_acoustic_group_within_this_seed": True,
        })
    for arm in ARMS:
        for metric in component_sums[arm]:
            component_sums[arm][metric] /= len(heldout)
        for metric in mean_metrics[arm]:
            mean_metrics[arm][metric] /= len(heldout)
    grouped = {
        metric: {
            "mean_over_acoustic_families": float(fmean(
                fmean(records[metric])
                for records in source_group_delta.values()
            )),
            "positive_family_deltas": sum(
                fmean(records[metric]) > 0
                for records in source_group_delta.values()
            ),
            "negative_family_deltas": sum(
                fmean(records[metric]) < 0
                for records in source_group_delta.values()
            ),
            "source_combination_families": len(source_group_delta),
            "caption_duplicates_counted_once": True,
        }
        for metric in COMPONENT_NAMES
    }
    return {
        "components_mean_per_independent_recording": component_sums,
        "aggregate_waveform_metrics": mean_metrics,
        "source_family_deltas_multires_minus_original": grouped,
        "per_recording": per_source,
        "independent_acoustic_recordings_scored": len(heldout),
        "bilingual_caption_duplicates_skipped": len(heldout),
        "source_combination_families": len(SYNTHETIC_COMBINATIONS),
        "matched_heldout_waveform_for_both_codecs": True,
        "no_perceptual_quality_or_statistical_significance_claim": True,
    }


def run_study(
    *, model_seed: int = 17, updates: int = 3,
    training_batch: int = 2, learning_rate: float = 0.001,
    training_seeds: tuple[int, ...] = TRAIN_SEEDS,
    heldout_seeds: tuple[int, ...] = VALIDATION_SEEDS,
    weights: CodecObjectiveWeights | None = None,
    spectral: CodecSpectralConfig | None = None,
) -> dict:
    """Run genuine paired first-party training with source-level heldout."""
    _settings(
        model_seed=model_seed, updates=updates,
        training_batch=training_batch, learning_rate=learning_rate,
        training_seeds=training_seeds, heldout_seeds=heldout_seeds,
    )
    opts = weights if weights is not None else CodecObjectiveWeights()
    config = spectral if spectral is not None else CodecSpectralConfig()
    if (not isinstance(opts, CodecObjectiveWeights)
            or not isinstance(config, CodecSpectralConfig)):
        raise ValueError("Original codec study requires validated numeric losses")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        train_dataset = synthetic_bilingual_waves(training_seeds)
        heldout_dataset = synthetic_bilingual_waves(heldout_seeds)
        training, training_hashes = _source_unique(train_dataset, training_seeds)
        heldout, heldout_hashes = _source_unique(heldout_dataset, heldout_seeds)
        if (set(training_hashes) & set(heldout_hashes)
                or train_dataset.fingerprint == heldout_dataset.fingerprint):
            raise ValueError("Training/heldout exact source-recording overlap")
        first = TinyAudioAutoencoder()
        other = copy.deepcopy(first)
        networks = {"original_loss": first, "multiresolution_loss": other}
        initial_sha = {
            name: model_state_sha256(model)
            for name, model in networks.items()
        }
        if initial_sha[ARMS[0]] != initial_sha[ARMS[1]]:
            raise ValueError("Codec variants did not start with identical parameters")
        optimizers = {
            arm: torch.optim.AdamW(model.parameters(), lr=learning_rate)
            for arm, model in networks.items()
        }
        generator = torch.Generator(device="cpu").manual_seed(model_seed + 1)
        histories = {arm: [] for arm in ARMS}
        for _ in range(updates):
            indices = torch.randint(
                len(training), (training_batch,),
                generator=generator, dtype=torch.long,
            )
            original = training[indices]
            for arm in ARMS:
                model = networks[arm]
                model.train()
                optimizers[arm].zero_grad(set_to_none=True)
                decoded = model(original)
                loss = (
                    reconstruction_loss(original, decoded)
                    if arm == "original_loss"
                    else weighted_codec_loss(
                        original, decoded, weights=opts, config=config
                    )
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite codec training objective")
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0,
                    error_if_nonfinite=True,
                )
                if not torch.isfinite(grad_norm):
                    raise ValueError("Nonfinite codec training gradient")
                optimizers[arm].step()
                histories[arm].append(float(loss.detach()))
        final_sha = {
            name: model_state_sha256(model)
            for name, model in networks.items()
        }
        if (final_sha["original_loss"] == final_sha["multiresolution_loss"]
                or any(final_sha[name] == initial_sha[name] for name in ARMS)):
            raise ValueError("Codec objective study did not change candidate weights")
        paired = _evaluate_models(
            networks, heldout, heldout_hashes, heldout_seeds,
        )
        if any(model_state_sha256(networks[name]) != final_sha[name]
               for name in ARMS):
            raise ValueError("Held-out evaluation mutated neural codec state")
        settings = {
            "model_seed": model_seed,
            "updates": updates,
            "training_batch": training_batch,
            "learning_rate": float(learning_rate),
            "training_seeds": list(training_seeds),
            "heldout_seeds": list(heldout_seeds),
            "source": "sonumio_first_party_synthetic_dsp",
            "sample_rate": 8000,
            "frames": FRAMES,
            "original_objective": "sonumio_waveform_l1_plus_3_log_stfts",
            "candidate_objective": LOSS_PROTOCOL,
            "candidate_weights": opts.as_dict(),
            "candidate_spectral_config": config.as_dict(),
        }
        digest = hashlib.sha256(json.dumps(
            settings, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        return {
            "protocol": PROTOCOL,
            "configuration": settings,
            "configuration_sha256": digest,
            "data": {
                "train_fingerprint": train_dataset.fingerprint,
                "heldout_fingerprint": heldout_dataset.fingerprint,
                "training_recordings": len(training),
                "heldout_recordings": len(heldout),
                "english_hindi_captions_per_split": train_dataset.examples,
                "exact_recording_overlap": False,
                "heldout_audio_used_in_training": False,
                "source_recording_sha256_heldout": heldout_hashes,
            },
            "models": {
                "identical_initial_sha256": initial_sha["original_loss"],
                "trained_model_sha256": final_sha,
                "trained_states_distinct": True,
                "unchanged_during_evaluation": True,
            },
            "paired_controls": {
                "same_initial_parameters": True,
                "same_training_minibatches": True,
                "same_optimizer_type_learning_rate_and_gradient_clipping": True,
                "only_reconstruction_training_loss_differs": True,
                "heldout_objective_components_identical_for_comparison": True,
                "english_hindi_caption_duplicates_counted_once": True,
            },
            "training_losses_not_directly_comparable": {
                arm: {"first": values[0], "last": values[-1], "count": len(values)}
                for arm, values in histories.items()
            },
            "heldout_reconstruction_diagnostics": paired,
            "global_torch_rng_isolated": True,
            "no_model_weights_or_audio_saved": True,
            "model_publication_authorized": False,
            "perceptual_fidelity_or_music_speech_quality_established": False,
            "warning": (
                "Only paired synthetic signal reconstruction metrics. "
                "Numerical spectral losses are not listener scores and "
                "this brief nonconverged training cannot establish "
                "codec quality, general music/speech or model readiness."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--training-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--training-seeds", default="0,1")
    parser.add_argument("--heldout-seeds", default="101,102")
    args = parser.parse_args()
    report = run_study(
        model_seed=args.model_seed,
        updates=args.updates,
        training_batch=args.training_batch,
        learning_rate=args.learning_rate,
        training_seeds=tuple(
            int(x) for x in args.training_seeds.split(",")
        ),
        heldout_seeds=tuple(
            int(x) for x in args.heldout_seeds.split(",")
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
