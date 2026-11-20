"""Golden SHA256 fingerprints for fully verified forest-aware DSP experiment.

These are independent MEASURED 378-case float32 exact-audio fingerprints,
not guesses and NOT hashes of WAV files or trained model weights.

Five successful jobs: https://github.com/sanskarIN/sonumio/actions/runs/44095705429
Four NumPy 2.4.6 runner CPUs produced identical waveforms and crops.
One Intel CPU ran the full study under NumPy 1.26.4 and then 2.4.6
in two fresh processes, obtaining per-recording byte equality for both
ambient/forest sine methods across ALL 189 source/seed cases per method.

These are FIRST-PARTY NUMERICAL DSP RESEARCH cases only. Seed 3 appears
in Sonumio's separate text-audio neural training, so the additional
numerical stress grid is NOT globally untouched model evaluation data.
"""
from __future__ import annotations

from research.ambient_sine_candidate import METHODS
from research.portable_forest_candidate import validate_report, PROTOCOL

REFERENCE_PROTOCOL = "sonumio-forest-aware-complete-dsp-reference-v1"
REFERENCE_NUMPY = ("1.26.4", "2.4.6")
REFERENCE_WORKFLOW = "https://github.com/sanskarIN/sonumio/actions/runs/44095705429"
# SHA256 of concatenated per-recording digest bytes in the research protocol's
# fixed member/seed order, not SHA256 of a downloadable source dataset.
REFERENCE_SPLITS = {
    "train": {
        "recordings": 42,
        "candidate_24k_sha256": "05b1891546041d895c15be6125c511f0563fb29c331fe70bdabcd395ebe0b18d",
        "candidate_8k_crop_sha256": "5e8b09ca8de0918728d7c8bd4b00d21afd97f13689697e0a9651146567ae9fe8",
    },
    "validation": {
        "recordings": 42,
        "candidate_24k_sha256": "e487d27a49294f50537e19f9d5b78d969607581d88dfad566f1ac58c577d42c5",
        "candidate_8k_crop_sha256": "67f7e5dadd51a429f770b8c41638667cde4f688fc51a9094a66fb8df0d3b22fb",
    },
    "heldout": {
        "recordings": 105,
        "candidate_24k_sha256": "47be2e54b572f55b9970364e3167338aa1969e55f147f3d3373420337408d9cb",
        "candidate_8k_crop_sha256": "cc27a25309724f54582ceddc0170b73423ecd19da629fb5f1961e96506502b97",
    },
}


def verify_reference(report: dict) -> dict:
    """Fail on any byte drift while preserving historic data and version scope."""
    validate_report(report)
    if report.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected forest-aware research protocol")
    environment = report.get("environment")
    if (not isinstance(environment, dict)
            or environment.get("numpy") not in REFERENCE_NUMPY):
        raise ValueError("Forest-aware goldens verified only on NumPy 1.26.4 / 2.4.6")
    for method in METHODS:
        for split, expected in REFERENCE_SPLITS.items():
            actual = report["splits"][method].get(split)
            if not isinstance(actual, dict):
                raise ValueError("Missing forest-aware golden source split")
            for key, value in expected.items():
                if actual.get(key) != value:
                    raise ValueError(
                        "Verified forest-aware SHA drift at "
                        + method + "/" + split + "/" + key
                    )
    return {
        "protocol": REFERENCE_PROTOCOL,
        "reference_workflow": REFERENCE_WORKFLOW,
        "verified_numpy_versions": list(REFERENCE_NUMPY),
        "cases_per_method": 189,
        "methods": list(METHODS),
        "total_candidate_recordings": 378,
        "matched_verified_full_raw_and_crop_reference": True,
        "num_independent_numpy_2_4_cpu_jobs": 4,
        "num_same_cpu_cross_numpy_jobs": 1,
        "original_dsp_hash_pinned": False,
        "original_data_modified": False,
        "model_weights_saved": False,
        "audio_files_saved": False,
        "neural_training_validity_claimed": False,
    }
