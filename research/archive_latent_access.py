"""Audited bounded waveform access for private latent experiments.

Only training crops enter the optimizer and calibration. Validation scans
eligible rows separately. The test split is inspected by readiness but is
never supplied by these accessors. No provenance identifiers leave reports.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import torch

from research.archive_readiness import ArchiveReadinessResult, eligible_fingerprint
from research.archive_sampler import CounterGroupSampler
from research.audio_dataset import _cropped_clip_digest
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError
from research.wave_diffusion import CLIP_FRAMES

MAX_BATCH = 16
MAX_SCAN_CLIPS = 4096


class AuditedCropAccess:
    def __init__(self, reader: CorpusArchiveReader, ready: ArchiveReadinessResult):
        if not isinstance(reader, CorpusArchiveReader) or not isinstance(ready, ArchiveReadinessResult):
            raise ArchiveError("Expected a verified archive and readiness result")
        if (ready.report.get("readiness_gate_passed") is not True
                or reader.report()["archive_sha256"] != ready.archive_sha256):
            raise ArchiveError("Private latent experiment requires matching ready archive")
        self.reader = reader
        self.ready = ready
        self._eligible = {}
        for split in ("train", "validation"):
            rows = ready.eligible[split]
            if not 1 <= len(rows) <= MAX_SCAN_CLIPS:
                raise ArchiveError("Private latent eligible split exceeds scan budget")
            actual = eligible_fingerprint(rows)
            if (actual != ready.eligible_fingerprints[split]
                    or actual != ready.report["eligible_split_fingerprints"][split]):
                raise ArchiveError("Private latent eligibility fingerprint changed")
            mapping = {item.row.index: item for item in rows}
            if len(mapping) != len(rows):
                raise ArchiveError("Duplicate eligible private latent row")
            self._eligible[split] = mapping

    def count(self, split):
        if split not in self._eligible:
            raise ArchiveError("Private latent access only allows train and validation")
        return len(self._eligible[split])

    def _verify(self, split, index, row, wave):
        expected = self._eligible[split].get(index)
        if (expected is None or expected.row != row or row.index != index
                or wave.shape != (1, CLIP_FRAMES) or wave.dtype != np.float32
                or _cropped_clip_digest(wave[0]) != expected.pcm_sha256):
            raise ArchiveError("Private latent waveform differs from audited eligibility")

    def sample(self, sampler, count):
        if (not isinstance(sampler, CounterGroupSampler)
                or sampler.eligible_sha256 != self.ready.eligible_fingerprints["train"]):
            raise ArchiveError("Expected sampler bound to eligible training data")
        if type(count) is not int or not 1 <= count <= MAX_BATCH:
            raise ArchiveError("Invalid private latent batch count")
        indices = sampler.draw(count)
        batch = self.reader.read_train_batch(indices.tolist())
        for index, row, wave, label in zip(indices.tolist(), batch.rows, batch.clips,
                                            batch.labels.tolist(), strict=True):
            self._verify("train", index, row, wave)
            if label != row.label:
                raise ArchiveError("Private latent sampled label changed")
        return indices, torch.from_numpy(batch.clips), torch.from_numpy(batch.labels)

    def scan(self, split, batch_size):
        self.count(split)
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH:
            raise ArchiveError("Invalid private latent scan batch size")
        expected = self._eligible[split]
        seen = set()
        rows, waves = [], []
        for shard in self.reader.iter_shards(split):
            for row, wave in zip(shard.rows, shard.clips, strict=True):
                if row.index not in expected:
                    continue
                if row.index in seen:
                    raise ArchiveError("Duplicate private latent scan row")
                self._verify(split, row.index, row, wave)
                seen.add(row.index)
                rows.append(row)
                waves.append(wave.copy())
                if len(rows) == batch_size:
                    yield tuple(rows), torch.from_numpy(np.stack(waves))
                    rows.clear()
                    waves.clear()
        if rows:
            yield tuple(rows), torch.from_numpy(np.stack(waves))
        if seen != set(expected):
            raise ArchiveError("Private latent scan missed eligible rows")


def counter_generator(seed, domain, counter):
    """Separate CPU RNG streams; no global RNG or implicit data-order seeds."""
    if (type(seed) is not int or not 0 <= seed < 2**32
            or type(counter) is not int or not 0 <= counter < 2**32
            or type(domain) is not bytes or not 1 <= len(domain) <= 64):
        raise ArchiveError("Invalid private latent counter RNG coordinates")
    digest = hashlib.sha256(
        b"sonumio-private-archive-latent-rng-v1\0"
        + struct.pack("<I", len(domain)) + domain + struct.pack("<II", seed, counter)
    ).digest()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") & ((1<<63)-1))


def hash_tensor(trace, value, dtype):
    raw = np.asarray(value.detach().numpy(), dtype=dtype, order="C").tobytes()
    trace.update(len(raw).to_bytes(8, "little"))
    trace.update(raw)
