"""Independent training-seed replication for Sonumio's original codec study.

A reconstruction-loss experiment must not use duplicated language captions
or multiple generated noises as independent trained-model replicates. This
module repeats matched original-vs-multiresolution *actual neural training*
for separate initializations and audits source hash, model identity, whole
held-out coverage, per-recording numerical scores, and report integrity
BEFORE publishing descriptive distributions.

No model or audio artifacts are written. No perceptual or rights claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from statistics import fmean

from research.codec_loss_study import (
    ARMS, PROTOCOL as SINGLE_PROTOCOL, _settings, run_study,
)
from research.codec_objectives import (
    COMPONENT_NAMES, CodecObjectiveWeights, CodecSpectralConfig,
    PROTOCOL as LOSS_PROTOCOL,
)
from research.multi_seed_study import summarize_values
from research.text_audio import SYNTHETIC_COMBINATIONS

PROTOCOL = "sonumio-independent-neural-codec-loss-replicates-v1"
MAX_SEEDS = 4
ACOUSTIC_FAMILIES = len(SYNTHETIC_COMBINATIONS)
METRIC_KEYS = ("sample_aligned_mae", "sample_aligned_rmse", "snr_db")
ERROR_METRICS = ("sample_aligned_mae", "sample_aligned_rmse")
FIELD_AUDIO = "source_recording_sha256_heldout"


def _sha(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _num(value: object, *, positive: bool = False) -> float:
    if (type(value) not in (float, int) or not math.isfinite(value)
            or (positive and value < 0)):
        raise ValueError("Malformed, nonfinite or negative codec study measurement")
    return float(value)


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _seed_list(model_seeds: tuple[int, ...]) -> None:
    if (type(model_seeds) is not tuple
            or not 2 <= len(model_seeds) <= MAX_SEEDS
            or len(set(model_seeds)) != len(model_seeds)
            or any(type(x) is not int or not 0 <= x < 2**32 - 500
                   for x in model_seeds)):
        raise ValueError("Require two to four distinct bounded model seeds")


def _config(*, seed: int, updates: int, training_batch: int,
            learning_rate: float, training_seeds: tuple[int, ...],
            heldout_seeds: tuple[int, ...],
            weights: CodecObjectiveWeights,
            spectral: CodecSpectralConfig) -> dict:
    return {
        "model_seed": seed, "updates": updates,
        "training_batch": training_batch,
        "learning_rate": float(learning_rate),
        "training_seeds": list(training_seeds),
        "heldout_seeds": list(heldout_seeds),
        "source": "sonumio_first_party_synthetic_dsp",
        "sample_rate": 8000, "frames": 8192,
        "original_objective": "sonumio_waveform_l1_plus_3_log_stfts",
        "candidate_objective": LOSS_PROTOCOL,
        "candidate_weights": weights.as_dict(),
        "candidate_spectral_config": spectral.as_dict(),
    }


def _verify_single(
    report: dict, *, seed: int, expected: dict,
    previous_acoustics: tuple | None,
) -> dict:
    """Validate report content, numerical bookkeeping and source-level proof."""
    if type(report) is not dict or report.get("protocol") != SINGLE_PROTOCOL:
        raise ValueError("Unknown individual Sonumio codec study protocol")
    cfg = report.get("configuration")
    data = report.get("data")
    models = report.get("models")
    controls = report.get("paired_controls")
    evals = report.get("heldout_reconstruction_diagnostics")
    training = report.get("training_losses_not_directly_comparable")
    if not all(type(obj) is dict for obj in (
        cfg, data, models, controls, evals, training,
    )):
        raise ValueError("Incomplete matched neural-codec study fields")
    if cfg != expected or report.get("configuration_sha256") != _digest(expected):
        raise ValueError("Codec candidate configuration changed across replications")
    if (report.get("global_torch_rng_isolated") is not True
            or report.get("no_model_weights_or_audio_saved") is not True
            or report.get("model_publication_authorized") is not False
            or report.get("perceptual_fidelity_or_music_speech_quality_established") is not False):
        raise ValueError("Codec report violates experiment or publication boundaries")
    target_count = ACOUSTIC_FAMILIES * len(expected["heldout_seeds"])
    train_count = ACOUSTIC_FAMILIES * len(expected["training_seeds"])
    audio_hashes = data.get(FIELD_AUDIO)
    if (data.get("training_recordings") != train_count
            or data.get("heldout_recordings") != target_count
            or data.get("english_hindi_captions_per_split") != 2 * train_count
            or data.get("exact_recording_overlap") is not False
            or data.get("heldout_audio_used_in_training") is not False
            or not _sha(data.get("train_fingerprint"))
            or not _sha(data.get("heldout_fingerprint"))
            or data["train_fingerprint"] == data["heldout_fingerprint"]
            or type(audio_hashes) is not list
            or len(audio_hashes) != target_count
            or not all(_sha(x) for x in audio_hashes)
            or len(set(audio_hashes)) != target_count):
        raise ValueError("Incomplete or duplicated codec source split evidence")
    identity = (
        data["train_fingerprint"], data["heldout_fingerprint"],
        tuple(audio_hashes),
    )
    if previous_acoustics is not None and previous_acoustics != identity:
        raise ValueError("Actual held-out source hashes drifted between model seeds")
    state = models.get("trained_model_sha256")
    if (not _sha(models.get("identical_initial_sha256"))
            or type(state) is not dict or set(state) != set(ARMS)
            or any(not _sha(state[arm]) for arm in ARMS)
            or state[ARMS[0]] == state[ARMS[1]]
            or any(state[arm] == models["identical_initial_sha256"]
                   for arm in ARMS)
            or models.get("trained_states_distinct") is not True
            or models.get("unchanged_during_evaluation") is not True):
        raise ValueError("Neural codec model states are not validly independent")
    for key in (
        "same_initial_parameters", "same_training_minibatches",
        "same_optimizer_type_learning_rate_and_gradient_clipping",
        "only_reconstruction_training_loss_differs",
        "heldout_objective_components_identical_for_comparison",
        "english_hindi_caption_duplicates_counted_once",
    ):
        if controls.get(key) is not True:
            raise ValueError("Codec candidates lost paired training controls")
    if set(training) != set(ARMS):
        raise ValueError("Missing training losses for a codec variant")
    for arm in ARMS:
        trace = training[arm]
        if type(trace) is not dict or trace.get("count") != expected["updates"]:
            raise ValueError("Codec training update coverage drifted")
        _num(trace.get("first"), positive=True)
        _num(trace.get("last"), positive=True)
    if (evals.get("independent_acoustic_recordings_scored") != target_count
            or evals.get("bilingual_caption_duplicates_skipped") != target_count
            or evals.get("source_combination_families") != ACOUSTIC_FAMILIES
            or evals.get("matched_heldout_waveform_for_both_codecs") is not True
            or evals.get("no_perceptual_quality_or_statistical_significance_claim") is not True):
        raise ValueError("Incomplete independent held-out codec evaluation")
    metrics = evals.get("components_mean_per_independent_recording")
    wave = evals.get("aggregate_waveform_metrics")
    rows = evals.get("per_recording")
    groups = evals.get("source_family_deltas_multires_minus_original")
    if (type(metrics) is not dict or set(metrics) != set(ARMS)
            or type(wave) is not dict or set(wave) != set(ARMS)
            or type(rows) is not list or len(rows) != target_count
            or type(groups) is not dict or set(groups) != set(COMPONENT_NAMES)):
        raise ValueError("Missing held-out codec measurements")
    for arm in ARMS:
        if (type(metrics[arm]) is not dict
                or set(metrics[arm]) != set(COMPONENT_NAMES)
                or type(wave[arm]) is not dict
                or set(wave[arm]) != set(METRIC_KEYS)):
            raise ValueError("Codec metrics missing or changed schema")
        for value in metrics[arm].values():
            _num(value, positive=True)
        for key, value in wave[arm].items():
            _num(value, positive=key in ERROR_METRICS)
    seen_groups = set()
    seen_rows = set()
    delta_by_family = {
        "/".join(str(i) for i in members): {
            component: [] for component in COMPONENT_NAMES
        }
        for members in SYNTHETIC_COMBINATIONS
    }
    row_mae = {arm: [] for arm in ARMS}
    row_rmse = {arm: [] for arm in ARMS}
    row_snr = {arm: [] for arm in ARMS}
    deltas = {component: [] for component in COMPONENT_NAMES}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("Corrupt held-out recording diagnostic")
        source_id = row.get("heldout_sha256")
        family = row.get("acoustic_group")
        recording_seed = row.get("recording_seed")
        slot = (family, recording_seed)
        if (not _sha(source_id) or source_id not in audio_hashes
                or type(family) is not str or family not in delta_by_family
                or type(recording_seed) is not int
                or recording_seed not in expected["heldout_seeds"]
                or slot in seen_rows
                or row.get("independent_acoustic_group_within_this_seed") is not True):
            raise ValueError("Duplicate or unaudited held-out acoustic recording")
        seen_rows.add(slot)
        seen_groups.add(family)
        for name, bucket in (
            ("sample_aligned_mae", row_mae),
            ("sample_aligned_rmse", row_rmse),
            ("sample_aligned_snr_db", row_snr),
        ):
            collection = row.get(name)
            if type(collection) is not dict or set(collection) != set(ARMS):
                raise ValueError("Incomplete matched recording-level waveform metrics")
            for arm in ARMS:
                bucket[arm].append(_num(
                    collection[arm], positive=name != "sample_aligned_snr_db",
                ))
        difference = row.get("multiresolution_minus_original_components")
        if type(difference) is not dict or set(difference) != set(COMPONENT_NAMES):
            raise ValueError("Incomplete component-level paired source deltas")
        for metric, value in difference.items():
            clean = _num(value)
            delta_by_family[family][metric].append(clean)
            deltas[metric].append(clean)
    if (len(seen_rows) != target_count
            or len(seen_groups) != ACOUSTIC_FAMILIES
            or set(x["heldout_sha256"] for x in rows) != set(audio_hashes)):
        raise ValueError("Some held-out recording groups were skipped")
    for arm in ARMS:
        for kind, collected, stated in (
            ("sample_aligned_mae", row_mae, "sample_aligned_mae"),
            ("sample_aligned_rmse", row_rmse, "sample_aligned_rmse"),
            ("sample_aligned_snr_db", row_snr, "snr_db"),
        ):
            mean = fmean(collected[arm])
            if not math.isclose(
                mean, wave[arm][stated], rel_tol=1e-6, abs_tol=1e-8,
            ):
                raise ValueError("Held-out numerical codec summary is inconsistent")
    effects = {}
    for metric in COMPONENT_NAMES:
        family = groups[metric]
        if (type(family) is not dict
                or family.get("source_combination_families") != ACOUSTIC_FAMILIES
                or family.get("caption_duplicates_counted_once") is not True):
            raise ValueError("Codec category groups are incomplete")
        family_means = [
            fmean(delta_by_family[key][metric])
            for key in delta_by_family
        ]
        group_mean = fmean(family_means)
        if (not math.isclose(
                _num(family.get("mean_over_acoustic_families")),
                group_mean, rel_tol=1e-6, abs_tol=1e-8,
            ) or family.get("positive_family_deltas") !=
                sum(x > 0 for x in family_means)
                or family.get("negative_family_deltas") !=
                sum(x < 0 for x in family_means)):
            raise ValueError("Source-family matched numerical loss deltas were forged")
        effects[metric] = group_mean
    return {
        "source_identity": identity,
        "model_initial": models["identical_initial_sha256"],
        "model_trained": state,
        "component_effects": effects,
        "waveform_mae_effect": (
            wave[ARMS[1]]["sample_aligned_mae"]
            - wave[ARMS[0]]["sample_aligned_mae"]
        ),
        "waveform_rmse_effect": (
            wave[ARMS[1]]["sample_aligned_rmse"]
            - wave[ARMS[0]]["sample_aligned_rmse"]
        ),
        "waveform_snr_effect": (
            wave[ARMS[1]]["snr_db"] - wave[ARMS[0]]["snr_db"]
        ),
        "config_sha": report["configuration_sha256"],
    }


def replicate_studies(
    *, model_seeds: tuple[int, ...] = (17, 29, 41),
    updates: int = 4, training_batch: int = 2,
    learning_rate: float = 0.001,
    training_seeds: tuple[int, ...] = (0, 1),
    heldout_seeds: tuple[int, ...] = (101, 102),
    weights: CodecObjectiveWeights | None = None,
    spectral: CodecSpectralConfig | None = None,
    runner=run_study,
) -> dict:
    """Train 2..4 independent paired codec variants with common sources."""
    _seed_list(model_seeds)
    _settings(
        model_seed=model_seeds[0], updates=updates,
        training_batch=training_batch, learning_rate=learning_rate,
        training_seeds=training_seeds, heldout_seeds=heldout_seeds,
    )
    loss_weights = weights if weights is not None else CodecObjectiveWeights()
    spectral_config = spectral if spectral is not None else CodecSpectralConfig()
    if (not isinstance(loss_weights, CodecObjectiveWeights)
            or not isinstance(spectral_config, CodecSpectralConfig)
            or not callable(runner)):
        raise ValueError("Invalid codec objective or replication runner")
    source = None
    initial_seen = set()
    final_seen = {arm: set() for arm in ARMS}
    rows = []
    for seed in model_seeds:
        run = runner(
            model_seed=seed, updates=updates,
            training_batch=training_batch,
            learning_rate=learning_rate,
            training_seeds=training_seeds,
            heldout_seeds=heldout_seeds,
            weights=loss_weights, spectral=spectral_config,
        )
        validated = _verify_single(
            run, seed=seed,
            expected=_config(
                seed=seed, updates=updates,
                training_batch=training_batch,
                learning_rate=learning_rate,
                training_seeds=training_seeds,
                heldout_seeds=heldout_seeds,
                weights=loss_weights,
                spectral=spectral_config,
            ),
            previous_acoustics=source,
        )
        if source is None:
            source = validated["source_identity"]
        if validated["model_initial"] in initial_seen:
            raise ValueError("Independent model seeds reused identical initial weights")
        initial_seen.add(validated["model_initial"])
        for arm in ARMS:
            fingerprint = validated["model_trained"][arm]
            if fingerprint in final_seen[arm]:
                raise ValueError("Independent model seeds reused trained codec weights")
            final_seen[arm].add(fingerprint)
        rows.append({
            "model_seed": seed,
            "model_initial_sha256": validated["model_initial"],
            "trained_model_sha256": validated["model_trained"],
            "configuration_sha256": validated["config_sha"],
            "heldout_groupwise_multires_minus_original":
                validated["component_effects"],
            "heldout_waveform_mae_delta_multires_minus_original":
                validated["waveform_mae_effect"],
            "heldout_waveform_rmse_delta_multires_minus_original":
                validated["waveform_rmse_effect"],
            "heldout_waveform_snr_delta_multires_minus_original":
                validated["waveform_snr_effect"],
        })
    descriptors = {
        component: summarize_values([
            row["heldout_groupwise_multires_minus_original"][component]
            for row in rows
        ])
        for component in COMPONENT_NAMES
    }
    for label in (
        "heldout_waveform_mae_delta_multires_minus_original",
        "heldout_waveform_rmse_delta_multires_minus_original",
        "heldout_waveform_snr_delta_multires_minus_original",
    ):
        descriptors[label] = summarize_values([row[label] for row in rows])
    cfg = {
        "model_seeds": list(model_seeds),
        "updates": updates, "training_batch": training_batch,
        "learning_rate": float(learning_rate),
        "training_seeds": list(training_seeds),
        "heldout_seeds": list(heldout_seeds),
        "candidate_weights": loss_weights.as_dict(),
        "candidate_spectral_config": spectral_config.as_dict(),
        "train_source_sha256": source[0],
        "heldout_source_sha256": source[1],
        "original_single_run_protocol": SINGLE_PROTOCOL,
        "source": "first_party_synthetic_dsp_only",
    }
    return {
        "protocol": PROTOCOL,
        "configuration": cfg,
        "configuration_sha256": _digest(cfg),
        "replicated_model_training_runs": rows,
        "independent_model_seed_count": len(rows),
        "matched_source_recordings_across_seeds": True,
        "source_recording_groups_and_bilingual_duplicates_validated": True,
        "identical_model_architectures_and_initialization_per_pair": True,
        "independent_initial_model_weight_hashes_verified": True,
        "independent_final_model_weight_hashes_verified": True,
        "descriptive_between_model_seed_effects": descriptors,
        "model_seeds_not_caption_or_audio_sampling_seeds_are_replicates": True,
        "perceptual_or_open_domain_quality_claim": False,
        "statistical_significance_or_model_winner_declared": False,
        "no_weights_or_audio_written": True,
        "model_publication_authorized": False,
        "warning": (
            "Descriptive replicated synthetic reconstruction loss is not a "
            "listener preference, speech/music ability, rights assessment "
            "or converged model quality. Independent initialization counts "
            "do not make the correlated DSP families real-world samples."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seeds", default="17,29,41")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--training-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--training-seeds", default="0,1")
    parser.add_argument("--heldout-seeds", default="101,102")
    args = parser.parse_args()
    result = replicate_studies(
        model_seeds=tuple(int(x) for x in args.model_seeds.split(",")),
        updates=args.updates, training_batch=args.training_batch,
        learning_rate=args.learning_rate,
        training_seeds=tuple(int(x) for x in args.training_seeds.split(",")),
        heldout_seeds=tuple(int(x) for x in args.heldout_seeds.split(",")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
