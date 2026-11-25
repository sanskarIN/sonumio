"""Digest-pinned private codec stage handoff for crop latent experiments.

The file contains a canonical metadata envelope and a bounded numeric NPZ
payload. It represents a COMPLETED codec stage, not an optimizer checkpoint.
It is not safe to load untrusted provenance merely because a hash matches.
The caller supplies a trusted digest and a freshly audited archive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import platform
import struct
import tempfile

import numpy as np
import torch

from research.archive_latent_access import AuditedCropAccess
from research.archive_latent_calibration import fit_archive_calibration
from research.archive_sampler import CounterGroupSampler
from research.checkpoint_archive import validate_numeric_archive
from research.checkpoint_snapshot import read_verified_weights, verified_npz_bytes
from research.corpus_archive_format import ArchiveError, canonical_json, parse_canonical, require_sha256
from research.crop_codec import CropAudioCodec
from research.wave_diffusion import CLIP_FRAMES
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-private-codec-stage-handoff-v1"
MAGIC = b"SLHAND01"
HEADER = struct.Struct("<8sII")
MAX_HEADER = 128 * 1024
MAX_FILE = 8 * 1024 * 1024
STAGE_FIELDS = ("seed", "codec_steps", "batch_size", "bits", "calibration_mode")


def runtime_pin():
    return {
        "python": platform.python_version(), "machine": platform.machine(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
    }


def stage_identity(access, plan):
    if not isinstance(access, AuditedCropAccess):
        raise ArchiveError("Expected audited archive access")
    return {
        "protocol": PROTOCOL, "archive_sha256": access.ready.archive_sha256,
        "eligible_training_sha256": access.ready.eligible_fingerprints["train"],
        "codec_configuration": {key: getattr(plan, key) for key in STAGE_FIELDS},
        "frames": CLIP_FRAMES, "runtime": runtime_pin(),
        "optimizer": "adamw-lr.001-foreach-false-fused-false-clip1",
    }


def verify_training_report(access, plan, report):
    fields = {"steps","first_loss","last_loss","mean_loss","sampler","training_trace_sha256"}
    if type(report) is not dict or set(report) != fields:
        raise ArchiveError("Unexpected codec stage training report fields")
    if type(report["steps"]) is not int or report["steps"] != plan.codec_steps:
        raise ArchiveError("Codec stage completed step count differs")
    for key in ("first_loss","last_loss","mean_loss"):
        value = report[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise ArchiveError("Invalid codec stage training loss")
    sampler = CounterGroupSampler(access.ready, seed=plan.seed)
    trace = hashlib.sha256(b"sonumio-private-latent-codec-trace-v1\0")
    for _ in range(plan.codec_steps):
        indices = sampler.draw(plan.batch_size)
        trace.update(indices.astype("<i8",copy=False).tobytes())
    if canonical_json(report["sampler"]) != canonical_json(sampler.describe()):
        raise ArchiveError("Codec stage sampler replay differs")
    if require_sha256(report["training_trace_sha256"]) != trace.hexdigest():
        raise ArchiveError("Codec stage draw trace differs")


def _state_arrays(codec):
    if not isinstance(codec, CropAudioCodec) or codec.frames != CLIP_FRAMES:
        raise ArchiveError("Expected matching crop codec model")
    model_state_sha256(codec)
    state = codec.state_dict()
    if any(t.dtype != torch.float32 or t.device.type != "cpu" for t in state.values()):
        raise ArchiveError("Codec handoff requires CPU float32 state")
    return {name: t.detach().contiguous().numpy().copy() for name,t in state.items()}


def encode_handoff(codec, metadata):
    raw_header = canonical_json(metadata)
    output = io.BytesIO()
    np.savez(output, **_state_arrays(codec))
    payload = output.getvalue()
    blob = HEADER.pack(MAGIC,len(raw_header),len(payload)) + raw_header + payload
    if not 0 < len(raw_header) <= MAX_HEADER or len(blob) > MAX_FILE:
        raise ArchiveError("Private codec handoff exceeds byte budget")
    return blob


def decode_handoff(blob, codec):
    if type(blob) is not bytes or not HEADER.size < len(blob) <= MAX_FILE:
        raise ArchiveError("Invalid bounded private codec handoff")
    magic, head_size, payload_size = HEADER.unpack_from(blob)
    if (magic != MAGIC or not 0 < head_size <= MAX_HEADER or payload_size <= 0
            or HEADER.size+head_size+payload_size != len(blob)):
        raise ArchiveError("Invalid private codec handoff envelope")
    metadata = parse_canonical(blob[HEADER.size:HEADER.size+head_size])
    if type(metadata) is not dict:
        raise ArchiveError("Invalid codec handoff metadata")
    payload = blob[HEADER.size+head_size:]
    expected = codec.state_dict()
    validate_numeric_archive(payload,expected)
    replacements = {}
    with np.load(verified_npz_bytes(payload),allow_pickle=False) as arrays:
        for name,tensor in expected.items():
            value = arrays[name]
            if (value.dtype != np.float32 or value.shape != tuple(tensor.shape)
                    or not np.isfinite(value).all()):
                raise ArchiveError("Invalid private codec numeric tensor")
            replacements[name] = torch.from_numpy(value.copy())
    return metadata,replacements


@dataclass(frozen=True, slots=True)
class CodecStageReceipt:
    sha256: str
    bytes_written: int
    codec_sha256: str
    calibration_sha256: str

    def report(self):
        return asdict(self)


def _destination(path):
    path = Path(path)
    if path.suffix != ".ckpt":
        raise ArchiveError("Private codec handoff requires .ckpt suffix")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ArchiveError("Expected existing non-symlink checkpoint directory")
    return path


def _publish(path, blob):
    # No overwrite, and no half-written final path. The caller must select a
    # trusted local directory on a filesystem supporting hard links.
    path = _destination(path)
    if os.path.lexists(path):
        raise ArchiveError("Private codec handoff already exists")
    temp = None
    try:
        fd,name = tempfile.mkstemp(prefix=".sonumio-codec-",suffix=".ckpt",dir=path.parent)
        temp = Path(name)
        with os.fdopen(fd,"wb") as output:
            output.write(blob)
            output.flush()
            os.fsync(output.fileno())
        os.link(temp,path)
    except OSError as exc:
        raise ArchiveError("Private codec handoff publication failed") from exc
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def save_codec_stage(path, *, access, plan, codec, initial_sha256,
                     training, calibration, weighting):
    if codec.training or any(p.requires_grad for p in codec.parameters()):
        raise ArchiveError("Codec stage must be frozen before handoff")
    verify_training_report(access,plan,training)
    trained = model_state_sha256(codec)
    if trained == require_sha256(initial_sha256):
        raise ArchiveError("Cannot hand off unchanged codec")
    # Derive calibration again from the source, not caller metadata.
    fresh, fresh_weight = fit_archive_calibration(
        access,codec,bits=plan.bits,mode=plan.calibration_mode,
    )
    if (fresh.fingerprint() != calibration.fingerprint()
            or canonical_json(fresh_weight) != canonical_json(weighting)):
        raise ArchiveError("Codec handoff calibration does not match audited training")
    metadata = {
        "identity":stage_identity(access,plan),
        "initial_codec_sha256":initial_sha256,"trained_codec_sha256":trained,
        "calibration_sha256":fresh.fingerprint(),"weight_trace_sha256":fresh_weight["weight_trace_sha256"],
        "training":training,"publication_authorized":False,
    }
    blob = encode_handoff(codec,metadata)
    pin = hashlib.sha256(blob).hexdigest()
    _publish(path,blob)
    return CodecStageReceipt(pin,len(blob),trained,fresh.fingerprint())


def load_codec_stage(path, *, expected_sha256, access, plan, codec):
    """Validate on a disposable candidate, then return the frozen codec.

    The caller's initialized codec is never modified even if loading fails.
    Numerical equivalence is scoped to the pinned runtime. This validates
    consistency and integrity, not proof that the claimed training happened.
    """
    require_sha256(expected_sha256)
    path = _destination(path)
    blob = read_verified_weights(path,expected_sha256,max_bytes=MAX_FILE)
    initial_sha256 = model_state_sha256(codec)
    metadata,replacements = decode_handoff(blob,codec)
    fields = {"identity","initial_codec_sha256","trained_codec_sha256",
              "calibration_sha256","weight_trace_sha256","training","publication_authorized"}
    if set(metadata) != fields or metadata["publication_authorized"] is not False:
        raise ArchiveError("Unexpected private codec handoff metadata")
    if canonical_json(metadata["identity"]) != canonical_json(stage_identity(access,plan)):
        raise ArchiveError("Private codec stage identity differs")
    if require_sha256(metadata["initial_codec_sha256"]) != initial_sha256:
        raise ArchiveError("Private codec initialization differs")
    verify_training_report(access,plan,metadata["training"])
    # Construct under a local RNG fork; failures cannot mutate the original.
    with torch.random.fork_rng(devices=[]):
        with torch.device("cpu"):
            candidate = CropAudioCodec(CLIP_FRAMES)
    candidate.load_state_dict(replacements,strict=True)
    candidate.eval().requires_grad_(False)
    trained = model_state_sha256(candidate)
    if trained != require_sha256(metadata["trained_codec_sha256"]) or trained == initial_sha256:
        raise ArchiveError("Private codec trained state digest differs")
    calibration,weighting = fit_archive_calibration(
        access,candidate,bits=plan.bits,mode=plan.calibration_mode,
    )
    if (calibration.fingerprint() != require_sha256(metadata["calibration_sha256"])
            or weighting["weight_trace_sha256"] != require_sha256(metadata["weight_trace_sha256"])):
        raise ArchiveError("Private codec calibration replay differs")
    receipt = CodecStageReceipt(expected_sha256,len(blob),trained,calibration.fingerprint())
    return candidate,metadata["training"],calibration,weighting,receipt
