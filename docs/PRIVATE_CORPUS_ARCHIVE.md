# Private reviewed-corpus archive (research-only)

This feature is for a researcher's own local, rights-reviewed recordings.
The archive contains actual model-input PCM, captions, recording-group
identifiers and provenance. Do **not** commit, upload or distribute archive
files just because the Sonumio source has an open-source license. The
archive does not establish legal rights, voice absence or model quality.

## Build and inspect

```bash
python -m research.corpus_archive build manifest.jsonl recordings private.scorpus --clips-per-asset 2 --shard-clips 64
python -m research.corpus_archive inspect private.scorpus --expected-sha256 SHA256_FROM_BUILD
```

Build uses the existing reviewed local ingestion pipeline and all of its
rights-declaration and leakage checks. It then writes deterministic bounded
shards to a local temporary file, verifies the actual file and publishes
with an atomic hard link that fails if the destination already exists.
No overwrite is supported. The archive directory must exist.

The build phase still uses the in-memory corpus loader; it is not a
large-corpus out-of-core ingestion engine. The reader subsequently verifies
all bytes and accesses requested shards or batches without retaining all PCM.

```python
from research.corpus_archive import CorpusArchiveReader

with CorpusArchiveReader("private.scorpus", expected_sha256=approved_digest) as reader:
    print(reader.report())
    batch = reader.read_train_batch([4, 0, 2, 4])
    # batch.clips: [4, 1, 2048] float32, independent writable storage.
    # batch.labels: int64, and batch.rows contains private provenance.
```

Only the training split is addressable through read_train_batch. Repeated
indices are intentional. The reader groups requests by shard so it decodes
each touched shard at most once per batch. Random inspection of validation
and test clips is explicit through read_clip or iter_shards.

## Format version 1

All integer lengths are unsigned little-endian. The file layout is:

1. Eight-byte magic SCORPUS1.
2. Consecutive shard payloads, ordered by train, validation, test, and row.
3. Canonical UTF-8 JSON index.
4. A 16-byte footer: index length (uint64) and magic SCIDX001.

A shard payload contains its metadata length (uint32), a canonical JSON
array of provenance rows, and contiguous little-endian float32 mono PCM.
Each descriptor binds split, start, count, absolute offset, byte length
and SHA256. The index records the source manifest and split fingerprints,
exact split counts, sample rate and crop geometry. Offsets must cover the
payload exactly; gaps, overlaps, trailing data and reordered shards fail.
All counts and byte lengths are checked before bounded reads.

The reader verifies every shard, finite samples, exact crop hashes across
splits, recording-group isolation and consistent asset provenance. A
trusted full-file SHA256 remains necessary to bind a cache to an approved
build: a self-consistent archive by itself does not prove authenticity or
rights. Earlier approximate leakage checks are performed by ingestion, not
repeated by the reader. The reader rechecks shard digests and ordinary-file
metadata on access; this is not a hostile filesystem sandbox.

Caps: 512 MiB total file size, 8 MiB index, 4 MiB metadata per shard, 256
clips per shard, 32,768 total clips and 1,024 training clips per batch.
Python metadata and temporary arrays use additional memory. The format
is uncompressed; it uses no pickle, model loading or executable payloads.
No automatic cache discovery, external data download or publication is
provided. The CLI emits aggregate JSON only, never captions or PCM.

## Remaining work

Out-of-core source decoding and leakage indexes for larger corpora,
sampler state checkpointing, trusted independent dataset review, and
training integration are separate engineering tasks. No Sonumio-1 weights
or model quality are established by this archive.
