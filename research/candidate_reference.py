"""Known-good full-corpus SHA reference for the *isolated* DSP candidate.

Hashes originate from independent CPU workflow #44063872705 on the
separately versioned, non-training candidate recipe. This is a controlled
numerical regression reference, NOT an audio-quality or licensing certificate.

Each aggregate SHA256 hashes the CONCATENATED PER-RECORDING SHA256 digests
in the canonical split/source/seed order. These are NOT file hashes, not
hashes of concatenated raw audio bytes, and not model weights.
"""
from __future__ import annotations

from research.complete_dsp_candidate import (
    CANDIDATE_RECIPE, PROTOCOL, validate_report,
)

REFERENCE_PROTOCOL = "sonumio-84-recording-dsp-candidate-reference-v1"
REFERENCE_WORKFLOW = "https://github.com/sanskarIN/sonumio/actions/runs/44063872705"
REFERENCE_SOURCE_COMMIT = "742d71d077a0d770a8b73bb171b6aa995ff8fcdb"
REFERENCE_SPLITS = {
    "train": {
        "recordings": 42,
        "candidate_raw_sha256": (
            "ee2c324eb111cd60ae4451fea8e769ae2fc6888eeac5e00124c5b64a910129af"
        ),
        "candidate_8k_dataset_sha256": (
            "3b0f97711eed11348c8fde76c60aef73ba549d1ee434d9a2c257b9f612dd4de3"
        ),
    },
    "validation": {
        "recordings": 42,
        "candidate_raw_sha256": (
            "3dcc50e5949675b03c3cd3edeb8f0614727a2a98dacc1c405f9cd4ed8cd3d5e3"
        ),
        "candidate_8k_dataset_sha256": (
            "8a5676f2de55435952b481df218e5ea9d872fcee4729e2adda3fd0a309984cb3"
        ),
    },
}


def verify_reference(report: dict) -> dict:
    """Require an already validated complete dataset to match the 4-CPU golden.

    The *original* NumPy DSP is not pinned to one expected SHA: its
    measured host-dependent float32 convolution is the problem being studied.
    This never promotes the candidate to model training.
    """
    validate_report(report)
    if report.get("protocol") != PROTOCOL or report.get("candidate_recipe") != CANDIDATE_RECIPE:
        raise ValueError("Incompatible source and numerical recipe")
    if report.get("scope") != "full" or len(report["recordings"]) != 84:
        raise ValueError("Reference is only for all 84 distinct recordings")
    for split, expected in REFERENCE_SPLITS.items():
        actual = report["splits"].get(split)
        if not isinstance(actual, dict):
            raise ValueError("Missing complete source split")
        for field, value in expected.items():
            if actual.get(field) != value:
                raise ValueError(f"Candidate reference drift: {split}.{field}")
    return {
        "protocol": REFERENCE_PROTOCOL,
        "recordings_checked": 84,
        "candidate_full_corpus_matches_reference": True,
        "original_numpy_hash_pinned": False,
        "no_audio_or_weights_published": True,
        "source_workflow": REFERENCE_WORKFLOW,
        "interpretation": (
            "This is a byte-fingerprint regression over synthetic 24k audio "
            "and 8192-frame 8k crop data; not model quality or a universal "
            "hardware/kernel guarantee."
        ),
    }
