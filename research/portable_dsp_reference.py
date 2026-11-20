"""Frozen NumPy 2.4.6 four-CPU portable DSP candidate SHA256 golden gates.

These values were *measured* in independent 4/4 successful hosted CPU jobs,
NOT guessed and NOT used to replace the original neural training dataset.
Source: https://github.com/sanskarIN/sonumio/actions/runs/44080000639

Each SHA256 is over concatenated per-recording digest bytes in canonical
source/seed order, NOT a hash of any published WAV or model checkpoint.
The original NumPy float32 legacy generator can be host/version-sensitive;
its SHA values are deliberately NOT pinned.
"""
from __future__ import annotations

from research.ambient_sine_candidate import METHODS
from research.portable_dsp_candidate import (
    PROTOCOL as PORTABLE_PROTOCOL, validate_report,
)

REFERENCE_PROTOCOL = "sonumio-portable-dsp-four-cpu-reference-v1"
REFERENCE_NUMPY = "2.4.6"
REFERENCE_RUN = "https://github.com/sanskarIN/sonumio/actions/runs/44080000639"
REFERENCE_SPLITS = {
    "train": {
        "recordings": 42,
        "portable_24k_sha256": "25482fb0bd0415c3db39787e835088c4af5b29b16ea993e79b1ae936d9422f34",
        "portable_8k_crop_sha256": "ee0fd7edd68e2fc5bc3bd6ceaa4472b875290d82f5ae5bde3e3249e055b40f79",
    },
    "validation": {
        "recordings": 42,
        "portable_24k_sha256": "df1da429546460fcf5faee0f76f1742cf324160e10a0565ff52a34fcf06bf48e",
        "portable_8k_crop_sha256": "d16ac7558a1f4f5ccec2490ca5746ada995cebb27fbf12d98e06c94a2aaebd01",
    },
    "heldout": {
        "recordings": 105,
        "portable_24k_sha256": "afe0e669d9577f284eaa50bf7d85d1e2722140aa8f3abb1f9fd8f88cd4ed8079",
        "portable_8k_crop_sha256": "30eaf47c085fec411fe5861037a73934bb149176fd553dbbeceb875a5dd3c8ca",
    },
}


def verify_reference(report: dict) -> dict:
    """Require all 378 research-only candidate waveforms to match goldens."""
    validate_report(report)
    if report["protocol"] != PORTABLE_PROTOCOL:
        raise ValueError("Unexpected original portable DSP research protocol")
    environment = report.get("environment")
    if not isinstance(environment, dict) or environment.get("numpy") != REFERENCE_NUMPY:
        raise ValueError("Frozen candidate hashes apply ONLY to NumPy 2.4.6")
    for method in METHODS:
        for split, expected in REFERENCE_SPLITS.items():
            actual = report["splits"][method].get(split)
            if not isinstance(actual, dict):
                raise ValueError("Missing portable DSP reference split")
            for field, value in expected.items():
                if actual.get(field) != value:
                    raise ValueError(
                        "Portable source reference drift for "
                        + method + "/" + split + "/" + field
                    )
    return {
        "protocol": REFERENCE_PROTOCOL,
        "reference_numpy": REFERENCE_NUMPY,
        "reference_workflow": REFERENCE_RUN,
        "recordings_per_method": 189,
        "methods_verified": list(METHODS),
        "total_candidate_recordings": 378,
        "all_candidate_hashes_match_four_cpu_reference": True,
        "original_generator_hash_pinned": False,
        "original_training_data_modified": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
    }
