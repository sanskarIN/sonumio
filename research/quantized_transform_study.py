"""Select packed transform representations using validation only.

Fixed quantization ranges come from training recordings. Only the winning
configuration for each predeclared payload budget is scored on a fresh
confirmation panel. The fixed open-set probe is independent of those panels.
Rates exclude shared basis/scales and container overhead; no files are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import numpy as np

from research.codec_noise_controls import (
    NOISE_VALIDATION_SEEDS, NOISE_CONFIRMATION_SEEDS, build_noise_panel,
    compare_noise_predictions, noise_predictions,
)
from research.latent_audio import FRAMES, VALIDATION_SEEDS
from research.generated_source_grounding import REFERENCE_SEEDS
from research.open_set_source_probe import NOISE_SEEDS, evaluate_probe, fit_probe, reference_panels
from research.transform_audio_codec import METHODS, TransformFit
from research.transform_codec_study import FIT_SEEDS, _panel, score_decoded_panel
from research.transform_quantization import BIT_WIDTHS, CoefficientQuantizer, payload_size
from research.wave_diffusion import SAMPLE_RATE

PROTOCOL = "sonumio-quantized-transform-selection-v2"
CONFIRMATION_SEEDS = (1501, 1503, 1504, 1506)
COEFFICIENT_BUDGETS = (8, 16, 32)
PAYLOAD_CAPS_BPS = (8000, 16000, 32000, 64000)
PERCENTILE = 100.0


def _prepare_panels():
    fit_waves, fit_labels, fit_hashes = _panel(FIT_SEEDS)
    val_waves, val_labels, val_hashes = _panel(VALIDATION_SEEDS)
    confirmation_waves, confirmation_labels, confirmation_hashes = _panel(CONFIRMATION_SEEDS)
    sources, noises, reference_hashes = reference_panels()
    noise_panels = {
        "validation": build_noise_panel(NOISE_VALIDATION_SEEDS),
        "confirmation": build_noise_panel(NOISE_CONFIRMATION_SEEDS),
    }
    all_hashes = (fit_hashes + val_hashes + confirmation_hashes
                  + noise_panels["validation"].hashes + noise_panels["confirmation"].hashes) + tuple(
        digest for domain in reference_hashes.values() for split in domain.values() for digest in split
    )
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("Exact recording overlap across selection and confirmation panels")
    all_seeds = (FIT_SEEDS + VALIDATION_SEEDS + CONFIRMATION_SEEDS
                 + NOISE_VALIDATION_SEEDS + NOISE_CONFIRMATION_SEEDS) + tuple(
        seed for domain in (REFERENCE_SEEDS, NOISE_SEEDS) for group in domain.values() for seed in group
    )
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Selection or confirmation seed overlap")
    probe = fit_probe(sources["fit"], sources["calibration"], noises["fit"], noises["calibration"])
    gate = evaluate_probe(probe, sources["holdout"], noises["holdout"])
    return {
        "fit": (fit_waves, fit_labels, fit_hashes),
        "validation": (val_waves, val_labels, val_hashes),
        "confirmation": (confirmation_waves, confirmation_labels, confirmation_hashes),
    }, noise_panels, probe, gate


def _pack_decode(fitted, quantizer, method, budget, waves):
    coefficients = fitted.encode(waves, method, budget)
    # Independently pack every recording; no savings by sharing partial bytes
    # across files. Compare actual serialized payload size with the formula.
    restored, sizes = [], []
    for index in range(len(coefficients)):
        single = coefficients[index:index + 1]
        payload = quantizer.encode(single)
        sizes.append(len(payload))
        if len(payload) != payload_size(single.shape, quantizer.bits):
            raise ValueError("Packed coefficient payload length mismatch")
        restored.append(quantizer.decode(payload, single.shape))
    decoded_coefficients = np.concatenate(restored, axis=0)
    reconstructed = fitted.decode(decoded_coefficients, method, budget)
    if len(set(sizes)) != 1:
        raise ValueError("Fixed shape payload lengths differ")
    return reconstructed, quantizer.diagnose(coefficients, decoded_coefficients), sizes[0]


def _evaluate(fitted, quantizer, method, budget, waves, labels, probe):
    reconstructed, diagnostics, size = _pack_decode(fitted, quantizer, method, budget, waves)
    result = score_decoded_panel(waves, reconstructed, labels, probe)
    result["quantization"] = diagnostics
    result["payload_bytes_per_recording"] = size
    result["payload_bits_per_second"] = size * 8 * SAMPLE_RATE / FRAMES
    return result


def _noise_evaluation(fitted, quantizer, method, budget, panel, original, probe):
    if quantizer is None:
        decoded = fitted.decode(fitted.encode(panel.waves, method, budget), method, budget)
    else:
        decoded, _, _ = _pack_decode(fitted, quantizer, method, budget, panel.waves)
    return compare_noise_predictions(panel, original, noise_predictions(probe, decoded))


def choose_candidate(rows, cap_bps, *, source_gate_passed):
    """Predeclared validation-only ordering; no confirmation fields consulted."""
    if type(cap_bps) is not int or cap_bps <= 0 or type(source_gate_passed) is not bool:
        raise ValueError("Invalid payload selection policy")
    feasible = [row for row in rows if row["validation"]["payload_bits_per_second"] <= cap_bps]
    if not feasible:
        return None

    def key(row):
        validation = row["validation"]
        # If the independent source probe fails, its scores cannot rank candidates.
        categories = validation["by_category"]
        false_acceptances = validation["noise_controls"]["reconstructed_accepted"] if source_gate_passed else 0
        correct = validation["reconstructed_correct_accepted"] if source_gate_passed else 0
        represented = sum(group["reconstructed_correct_accepted"] > 0 for group in categories.values()) if source_gate_passed else 0
        error = validation["numerical"]["relative_squared_error"]
        if error is None or not math.isfinite(error):
            raise ValueError("Invalid validation reconstruction error")
        return (false_acceptances, -correct, -represented, error, validation["payload_bits_per_second"], row["id"])

    return min(feasible, key=key)["id"]


def run_study(*, methods=METHODS, budgets=COEFFICIENT_BUDGETS, widths=BIT_WIDTHS, caps=PAYLOAD_CAPS_BPS):
    for requested, supported in ((methods, METHODS), (budgets, COEFFICIENT_BUDGETS),
                                 (widths, BIT_WIDTHS), (caps, PAYLOAD_CAPS_BPS)):
        if (type(requested) is not tuple or not requested
                or len(set(requested)) != len(requested)
                or any(type(item) is not type(supported[0]) or item not in supported for item in requested)):
            raise ValueError("Invalid quantized transform study grid")
    panels, noise_panels, probe, gate = _prepare_panels()
    validation_noise = noise_panels["validation"]
    validation_original = noise_predictions(probe, validation_noise.waves)
    selection_probe_usable = gate["gate"]["passed"] and all(x is None for x in validation_original)
    fit_waves, fit_labels, _ = panels["fit"]
    val_waves, val_labels, _ = panels["validation"]
    confirmation_waves, confirmation_labels, _ = panels["confirmation"]
    fitted = TransformFit.fit(fit_waves, fit_labels)
    candidates, quantizers = [], {}
    for method in methods:
        for budget in budgets:
            fit_coefficients = fitted.encode(fit_waves, method, budget)
            for width in widths:
                name = f"{method}:{budget}:{width}"
                quantizer = CoefficientQuantizer.fit(fit_coefficients, width, percentile=PERCENTILE)
                quantizers[name] = (method, budget, quantizer)
                validation = _evaluate(fitted, quantizer, method, budget, val_waves, val_labels, probe)
                validation["noise_controls"] = _noise_evaluation(
                    fitted, quantizer, method, budget, validation_noise, validation_original, probe,
                )
                candidates.append({
                    "id": name, **fitted.describe(method, budget),
                    "quantizer": quantizer.describe(), "validation": validation,
                })
    # Freeze every choice before evaluating any candidate on confirmation.
    selection = [
        {"payload_cap_bps": cap, "candidate_id": choose_candidate(
            candidates, cap, source_gate_passed=selection_probe_usable,
        )}
        for cap in caps
    ]
    selected_ids = sorted({row["candidate_id"] for row in selection if row["candidate_id"] is not None})
    confirmation = {}
    confirmation_noise = noise_panels["confirmation"]
    confirmation_original = noise_predictions(probe, confirmation_noise.waves)
    for name in selected_ids:
        method, budget, quantizer = quantizers[name]
        quantized = _evaluate(fitted, quantizer, method, budget, confirmation_waves, confirmation_labels, probe)
        floating = fitted.decode(fitted.encode(confirmation_waves, method, budget), method, budget)
        quantized["noise_controls"] = _noise_evaluation(
            fitted, quantizer, method, budget, confirmation_noise, confirmation_original, probe,
        )
        floating_scores = score_decoded_panel(confirmation_waves, floating, confirmation_labels, probe)
        floating_scores["noise_controls"] = _noise_evaluation(
            fitted, None, method, budget, confirmation_noise, confirmation_original, probe,
        )
        confirmation[name] = {
            "quantized": quantized,
            "same_transform_unquantized": floating_scores,
            "confirmation_noise_screen_passed": (
                selection_probe_usable and quantized["noise_controls"]["zero_original_and_reconstruction_acceptances"]
            ),
        }
    return {
        "protocol": PROTOCOL,
        "configuration": {"methods": list(methods), "budgets": list(budgets),
                          "bit_widths": list(widths), "caps_bps": list(caps),
                          "sample_rate": SAMPLE_RATE, "frames": FRAMES},
        "panel_fingerprints": {
            name: {"recordings": len(waves), "sha256": hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()}
            for name, (waves, _, hashes) in panels.items()
        },
        "reference_gate_passed": gate["gate"]["passed"],
        "selection_uses_source_scores": selection_probe_usable,
        "selection_uses_noise_scores": selection_probe_usable,
        "validation_original_noise_accepted": sum(x is not None for x in validation_original),
        "confirmation_original_noise_accepted": sum(x is not None for x in confirmation_original),
        "noise_panel_fingerprints": {
            name: {"recordings": len(panel.waves), "sha256": panel.fingerprint}
            for name, panel in noise_panels.items()
        },
        "selection_rule": "validation false noise acceptance, correct accepted sources, category coverage, relative squared error, payload rate, stable ID",
        "confirmation_noise_screen_is_not_model_readiness": True,
        "ranges_and_transform_fit_on_training_only": True,
        "confirmation_used_for_selection": False, "exact_split_overlap": False,
        "payload_rate_excludes_shared_scales_basis_and_container": True,
        "production_bitrate_established": False,
        "audio_weights_or_payload_files_saved": False,
        "model_readiness_established": False,
        "validation_candidates": candidates,
        "selection": selection, "confirmation": confirmation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--budgets", nargs="+", type=int, choices=COEFFICIENT_BUDGETS, default=list(COEFFICIENT_BUDGETS))
    parser.add_argument("--widths", nargs="+", type=int, choices=BIT_WIDTHS, default=list(BIT_WIDTHS))
    parser.add_argument("--caps", nargs="+", type=int, choices=PAYLOAD_CAPS_BPS, default=list(PAYLOAD_CAPS_BPS))
    args = parser.parse_args()
    print(json.dumps(run_study(methods=tuple(args.methods), budgets=tuple(args.budgets),
                               widths=tuple(args.widths), caps=tuple(args.caps)), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
