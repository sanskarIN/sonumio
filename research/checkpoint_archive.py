"""Bounded, numeric-only NPZ archive preflight for local research checkpoints.

SHA-256 is checked by callers before this preflight. This module additionally
rejects ZIP members and NumPy headers that could cause excessive allocations
before NumPy's array loader sees the expected tensor shapes.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_ARRAY_HEADER_BYTES = 4096


def validate_numeric_archive(path: Path, expected: dict) -> None:
    """Reject extra, oversized, encrypted or shape-spoofed arrays before loading.

    ``expected`` maps state-dict names to torch tensors. No arrays are loaded
    by this validator and it does not execute code from the checkpoint.
    """
    path = Path(path)
    if path.stat().st_size > MAX_COMPRESSED_BYTES:
        raise ValueError('Checkpoint exceeds research size limit')
    wanted = {name + '.npy': tensor for name, tensor in expected.items()}
    try:
        with ZipFile(path, 'r') as archive:
            infos = archive.infolist()
            names = [entry.filename for entry in infos]
            if len(names) != len(wanted) or set(names) != set(wanted):
                raise ValueError('Unexpected checkpoint tensor names')
            for info in infos:
                tensor = wanted[info.filename]
                limit = tensor.numel() * tensor.element_size() + MAX_ARRAY_HEADER_BYTES
                if info.file_size > limit:
                    raise ValueError('Checkpoint tensor exceeds expected byte budget: ' + info.filename)
                if info.is_dir() or info.flag_bits & 1 or info.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
                    raise ValueError('Unsupported checkpoint ZIP member: ' + info.filename)
                with archive.open(info, 'r') as fp:
                    version = np.lib.format.read_magic(fp)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                            fp, max_header_size=MAX_ARRAY_HEADER_BYTES)
                    elif version == (2, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                            fp, max_header_size=MAX_ARRAY_HEADER_BYTES)
                    else:
                        raise ValueError('Unsupported checkpoint NumPy header: ' + info.filename)
                    if tuple(shape) != tuple(tensor.shape) or dtype != np.dtype('float32') or fortran_order:
                        raise ValueError('Invalid checkpoint tensor header: ' + info.filename)
    except (BadZipFile, EOFError, OSError) as exc:
        raise ValueError('Malformed checkpoint NPZ archive') from exc
