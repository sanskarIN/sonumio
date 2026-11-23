"""Paired model-seed × EMA-decay ablation for Sonumio neural v-generation.

The online training trajectory is rerun identically for each EMA decay.
We REQUIRE byte-identical online weights, held-out scores and generated
waveforms within each initialization seed before comparing EMA outputs.

Only original rights-free first-party synthetic DSP is used. This is
numeric research, not audible quality evidence or Sonumio-1 publication.
No model parameters or audio files are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from statistics import mean

from research.multi_seed_study import summarize_values
from research.velocity_ema_study import (
    PROTOCOL as SINGLE_PROTOCOL, _settings, run_study,
)

PROTOCOL = "sonumio-velocity-ema-decay-paired-ablation-v1"
ARMS = ("raw", "latent")
VARIANTS = ("online", "ema")
PAIR_CATEGORIES = ("rain", "wind")
LANGUAGES = ("en", "hi")


def _digest(obj: dict) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _hex_digest(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _metric(value: object, *, fraction: bool = False) -> float:
    if (type(value) not in (float, int) or not math.isfinite(value)
            or (fraction and not 0 <= value <= 1)):
        raise ValueError("Missing, nonfinite or out-of-range EMA metric")
    return float(value)


def _validate_settings(*, seeds: tuple[int, ...], decays: tuple[float, ...],
                       steps: int, codec_steps: int, diffusion_steps: int,
                       channels: int, warmup: bool,
                       sampling_seeds: tuple[int, ...]) -> tuple[float, ...]:
    if (type(seeds) is not tuple or not 2 <= len(seeds) <= 4
            or any(type(x) is not int or not 0 <= x < 2**32 - 8192
                   for x in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Require 2..4 unique bounded model initialization seeds")
    if (type(decays) is not tuple or not 2 <= len(decays) <= 4
            or any(type(d) not in (float, int) or not math.isfinite(d)
                   or not 0 <= d < 1 for d in decays)
            or decays[0] != 0 or any(a >= b for a, b in zip(
                decays, decays[1:]))):
        raise ValueError("Require increasing EMA decays starting with zero")
    for seed in seeds:
        _settings(
            seed=seed, steps=steps, codec_steps=codec_steps,
            diffusion_steps=diffusion_steps, channels=channels,
            decay=decays[0], warmup=warmup, sampling_seeds=sampling_seeds,
        )
    return tuple(float(d) for d in decays)


def _verify_run(report: dict, *, seed: int, decay: float,
                steps: int, codec_steps: int, diffusion_steps: int,
                channels: int, warmup: bool,
                sampling_seeds: tuple[int, ...]) -> dict:
    """Refuse incomplete/corrupt/mixed protocols before aggregation."""
    if type(report) is not dict or report.get("protocol") != SINGLE_PROTOCOL:
        raise ValueError("Incompatible EMA research report protocol")
    config = report.get("configuration")
    train = report.get("training")
    heldout = report.get("heldout_velocity_objective")
    states = report.get("model_state_integrity")
    records = report.get("in_memory_generated_numerical_comparisons")
    aggregate = report.get("by_arm")
    if not all(type(x) is dict for x in (config, train, heldout, states, records, aggregate)):
        raise ValueError("Incomplete EMA research report fields")
    expected_config = {
        "seed": seed, "steps": steps, "codec_steps": codec_steps,
        "diffusion_steps": diffusion_steps, "channels": channels,
        "decay": float(decay), "warmup": warmup,
        "sampling_seeds": list(sampling_seeds),
        "prediction_objective": "velocity",
        "sampler": "deterministic_ddim_eta_zero",
        "source": "sonumio_first_party_synthetic_dsp",
    }
    if config != expected_config or report.get("configuration_sha256") != _digest(config):
        raise ValueError("EMA study configuration was changed or forged")
    if (report.get("online_and_ema_same_training_trajectory") is not True
            or report.get("heldout_audio_used_for_training") is not False
            or report.get("no_model_weights_or_audio_saved") is not True
            or report.get("model_publication_authorized") is not False
            or report.get("listening_or_semantic_audio_quality_evaluated") is not False):
        raise ValueError("EMA study violates source, pairing or release rules")
    if (train.get("independent_training_recordings") != 42
            or train.get("independent_heldout_recordings") != 42
            or train.get("bilingual_caption_examples_per_split") != 84
            or train.get("train_heldout_exact_overlap") is not False
            or not _hex_digest(train.get("train_fingerprint"))
            or not _hex_digest(train.get("validation_fingerprint"))
            or train["train_fingerprint"] == train["validation_fingerprint"]):
        raise ValueError("EMA study missing valid source-group evaluation")
    if (set(heldout) != set(ARMS) or set(states) != set(ARMS)
            or set(records) != set(ARMS) or set(aggregate) != set(ARMS)):
        raise ValueError("Missing or mixed raw/latent evaluation arms")

    online_rows: dict[str, dict] = {}
    outcomes: dict[str, dict] = {}
    for arm in ARMS:
        objective = heldout[arm]
        state = states[arm]
        rows = records[arm]
        if (type(objective) is not dict or type(state) is not dict
                or type(rows) is not list or type(aggregate[arm]) is not dict):
            raise ValueError("Missing per-arm EMA evaluation data")
        if (objective.get("shared_noise_and_timesteps") is not True
                or objective.get("bilingual_caption_examples") != 84
                or objective.get("distinct_audio_recordings") != 42
                or state.get("same_parameters_through_evaluation") is not True
                or type(state.get("ema_updates")) is not int
                or state["ema_updates"] != steps
                or not _hex_digest(state.get("online_model_sha256"))
                or not _hex_digest(state.get("ema_model_sha256"))):
            raise ValueError("EMA pairing or model-integrity evidence missing")
        online_mse = _metric(objective.get("online"))
        ema_mse = _metric(objective.get("ema"))
        difference = _metric(objective.get("ema_minus_online"))
        if (online_mse < 0 or ema_mse < 0
                or not math.isclose(difference, ema_mse - online_mse,
                                    rel_tol=1e-8, abs_tol=1e-10)):
            raise ValueError("EMA held-out comparison arithmetic is invalid")
        if len(rows) != len(sampling_seeds) * 4:
            raise ValueError("EMA report has missing waveform comparison rows")
        identified = {}
        for row in rows:
            if type(row) is not dict or row.get(
                "prompt_adherence_or_human_quality_measured"
            ) is not False:
                raise ValueError("EMA report falsely claims auditory or semantic quality")
            key = (row.get("seed"), row.get("category"), row.get("language"))
            if (type(key[0]) is not int or key[0] not in sampling_seeds
                    or key[1] not in PAIR_CATEGORIES or key[2] not in LANGUAGES
                    or key in identified):
                raise ValueError("Invalid/duplicated EMA waveform comparison keys")
            identified[key] = row
            for variant in VARIANTS:
                entry = row.get(variant)
                if type(entry) is not dict or type(entry.get("postclip")) is not dict:
                    raise ValueError("Missing online/EMA waveform health")
                health = entry["postclip"]
                if not _hex_digest(health.get("sha256")):
                    raise ValueError("Invalid generated waveform hash")
                for field in ("rms", "peak", "clipping_fraction"):
                    value = _metric(health.get(field), fraction=field == "clipping_fraction")
                    if field in ("rms", "peak") and not 0 <= value <= 1.0001:
                        raise ValueError("Out-of-range generated waveform")
                if entry.get("prelimiter_signal_source") != (
                    "raw_ddpm_before_final_clamp" if arm == "raw"
                    else "latent_codec_decoder_before_tanh"
                ):
                    raise ValueError("Unexplained waveform/activation provenance")
                _metric(entry.get("preclip_peak"))
                _metric(entry.get("preclip_out_of_range_fraction"), fraction=True)
            delta = row.get("online_ema_waveform_difference")
            if (type(delta) is not dict
                    or type(delta.get("exact_same_waveform")) is not bool
                    or not 0 <= _metric(delta.get("mean_absolute_difference")) <= 2
                    or not -1 <= _metric(delta.get("waveform_cosine")) <= 1):
                raise ValueError("Invalid EMA waveform comparison")
            if decay == 0 and (row["online"]["postclip"]["sha256"]
                               != row["ema"]["postclip"]["sha256"]
                               or delta["exact_same_waveform"] is not True
                               or delta["mean_absolute_difference"] != 0):
                raise ValueError("Zero-decay negative control changed audio")
        expected_keys = {
            (sample_seed, category, language)
            for sample_seed in sampling_seeds
            for category in PAIR_CATEGORIES for language in LANGUAGES
        }
        if set(identified) != expected_keys:
            raise ValueError("Incomplete bilingual EMA waveform evaluation grid")
        observed_online = {
            "|".join(map(str, key)): identified[key]["online"]
            for key in sorted(identified)
        }
        online_rows[arm] = observed_online
        declared = aggregate[arm]
        if declared.get("generated_pairs_compared") != len(rows):
            raise ValueError("Incorrect number of generated EMA sample pairs")
        online_clip = mean(
            row["online"]["postclip"]["clipping_fraction"] for row in rows
        )
        ema_clip = mean(
            row["ema"]["postclip"]["clipping_fraction"] for row in rows
        )
        waveform_difference = mean(
            row["online_ema_waveform_difference"]["mean_absolute_difference"]
            for row in rows
        )
        for name, observed in (
            ("mean_online_clipping_fraction", online_clip),
            ("mean_ema_clipping_fraction", ema_clip),
            ("mean_waveform_difference", waveform_difference),
            ("heldout_velocity_mse_ema_minus_online", difference),
        ):
            if not math.isclose(_metric(declared.get(name)), observed,
                                abs_tol=1e-10, rel_tol=1e-8):
                raise ValueError("EMA waveform/heldout summary is inconsistent")
        if decay == 0 and (state["online_model_sha256"]
                           != state["ema_model_sha256"] or difference != 0
                           or online_clip != ema_clip):
            raise ValueError("Zero-decay EMA weights or numerical score drifted")
        outcomes[arm] = {
            "online_sha": state["online_model_sha256"],
            "ema_sha": state["ema_model_sha256"],
            "online_mse": online_mse, "ema_mse": ema_mse,
            "mse_delta": difference,
            "online_clipping": online_clip, "ema_clipping": ema_clip,
            "clipping_delta": ema_clip - online_clip,
            "waveform_difference": waveform_difference,
        }
    return {
        "data_fingerprint": (
            train["train_fingerprint"], train["validation_fingerprint"],
        ),
        "online_rows": online_rows,
        "outcomes": outcomes,
        "config_sha": report["configuration_sha256"],
    }


def run_ablation(*, seeds: tuple[int, ...] = (17, 29, 41),
                 decays: tuple[float, ...] = (0.0, 0.5, 0.9),
                 steps: int = 8, codec_steps: int = 2,
                 diffusion_steps: int = 8, channels: int = 16,
                 warmup: bool = False,
                 sampling_seeds: tuple[int, ...] = (301, 302),
                 runner=run_study) -> dict:
    """Run strict matched EMA decay arms over independent model initializations."""
    decays = _validate_settings(
        seeds=seeds, decays=decays, steps=steps, codec_steps=codec_steps,
        diffusion_steps=diffusion_steps, channels=channels,
        warmup=warmup, sampling_seeds=sampling_seeds,
    )
    if not callable(runner):
        raise ValueError("EMA ablation runner must be callable")
    rows = []
    source_data = None
    online_state_per_seed: dict[int, dict] = {}
    distinct_online: dict[str, set[str]] = {arm: set() for arm in ARMS}
    for seed in seeds:
        reference = None
        for decay in decays:
            report = runner(
                seed=seed, steps=steps, codec_steps=codec_steps,
                diffusion_steps=diffusion_steps, channels=channels,
                decay=decay, warmup=warmup, sampling_seeds=sampling_seeds,
            )
            checked = _verify_run(
                report, seed=seed, decay=decay,
                steps=steps, codec_steps=codec_steps,
                diffusion_steps=diffusion_steps, channels=channels,
                warmup=warmup, sampling_seeds=sampling_seeds,
            )
            if source_data is None:
                source_data = checked["data_fingerprint"]
            elif source_data != checked["data_fingerprint"]:
                raise ValueError("Synthetic data identity drifted across EMA trials")
            if reference is None:
                reference = checked
                for arm in ARMS:
                    fingerprint = checked["outcomes"][arm]["online_sha"]
                    if fingerprint in distinct_online[arm]:
                        raise ValueError("Different model seeds produced duplicate online weights")
                    distinct_online[arm].add(fingerprint)
            for arm in ARMS:
                if (reference["outcomes"][arm]["online_sha"]
                        != checked["outcomes"][arm]["online_sha"]
                        or reference["outcomes"][arm]["online_mse"]
                        != checked["outcomes"][arm]["online_mse"]
                        or reference["online_rows"][arm]
                        != checked["online_rows"][arm]):
                    raise ValueError(
                        "Changing EMA decay changed the online optimization or inference"
                    )
            rows.append({
                "model_seed": seed, "ema_decay": decay,
                "configuration_sha256": checked["config_sha"],
                "arms": checked["outcomes"],
            })
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for decay in decays:
            subset = [
                row["arms"][arm] for row in rows
                if row["ema_decay"] == decay
            ]
            summary[arm][str(decay)] = {
                "heldout_mse_delta_ema_minus_online": summarize_values(
                    [item["mse_delta"] for item in subset]
                ),
                "generated_clipping_delta_ema_minus_online": summarize_values(
                    [item["clipping_delta"] for item in subset]
                ),
                "waveform_mae_between_online_and_ema": summarize_values(
                    [item["waveform_difference"] for item in subset]
                ),
                "ema_reduced_heldout_error_in_models": sum(
                    item["mse_delta"] < 0 for item in subset
                ),
                "ema_reduced_clipping_in_models": sum(
                    item["clipping_delta"] < 0 for item in subset
                ),
            }
    config = {
        "model_seeds": list(seeds), "ema_decays": list(decays),
        "online_steps_per_arm": steps, "codec_steps": codec_steps,
        "diffusion_steps": diffusion_steps, "channels": channels,
        "warmup": warmup, "sampling_seeds": list(sampling_seeds),
        "first_party_train_fingerprint": source_data[0],
        "first_party_validation_fingerprint": source_data[1],
        "source_protocol": SINGLE_PROTOCOL,
        "audio_sample_rate": 8000, "audio_frames": 8192,
    }
    return {
        "protocol": PROTOCOL,
        "configuration": config, "configuration_sha256": _digest(config),
        "per_model_seed_and_decay": rows,
        "within_arm_descriptive_summary": summary,
        "total_matched_runs": len(rows),
        "same_source_data_across_runs_verified": True,
        "identical_online_trajectory_within_model_seed_verified": True,
        "independent_online_model_fingerprints_across_seeds_verified": True,
        "zero_decay_null_control_passed": True,
        "no_weights_or_audio_saved": True,
        "model_release_authorized": False,
        "perceptual_quality_or_prompt_adherence_measured": False,
        "statistical_significance_or_quality_winner_declared": False,
        "warnings": [
            "EMA decay is the only experimental training difference within a model seed.",
            "Metrics describe short, synthetic, possibly untrained v-models; not audible quality.",
            "Duplicated English/Hindi captions share audio and are not independent recordings.",
            "Different initialization seeds are the only independent model replicates.",
            "Source byte fingerprints may differ across CPU hosts: no cross-runner identity claim.",
            "No third-party rights-protected recordings, WAV, model weights or release artifacts.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="17,29,41")
    parser.add_argument("--decays", default="0,0.5,0.9")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--codec-steps", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--sampling-seeds", default="301,302")
    parser.add_argument("--warmup", action="store_true")
    options = parser.parse_args()
    report = run_ablation(
        seeds=tuple(int(value) for value in options.seeds.split(",")),
        decays=tuple(float(value) for value in options.decays.split(",")),
        steps=options.steps, codec_steps=options.codec_steps,
        diffusion_steps=options.diffusion_steps,
        sampling_seeds=tuple(
            int(value) for value in options.sampling_seeds.split(",")
        ), warmup=options.warmup,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
