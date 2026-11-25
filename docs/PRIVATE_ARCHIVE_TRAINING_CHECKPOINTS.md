# Private archive training checkpoints

Experimental CPU archive training supports optional local complete-step
checkpoints. This is not a published Sonumio model or a supported inference
format. The default training command remains in-memory only; no model
weights, recordings, or private archive files are committed or published.

## Run and resume

Use the archive digest obtained during the approved corpus build. Example:

```bash
python -m research.archive_wave_training private.scorpus \
  --expected-sha256 ARCHIVE_SHA256 \
  --steps 20 --batch-size 4 --channels 8 --diffusion-steps 8 \
  --checkpoint-out prefix.stcheckpoint
```

Retain the checkpoint SHA256 from the JSON receipt independently of the
checkpoint file. Resume by specifying the same configuration, archive and
checkpoint, but a new total step target:

```bash
python -m research.archive_wave_training private.scorpus \
  --expected-sha256 ARCHIVE_SHA256 \
  --steps 40 --batch-size 4 --channels 8 --diffusion-steps 8 \
  --resume prefix.stcheckpoint --resume-sha256 CHECKPOINT_SHA256 \
  --checkpoint-out continued.stcheckpoint
```

For periodic recovery points, use an existing ordinary local directory and
`--checkpoint-dir checkpoints --checkpoint-every 10`. Files are named
`step-00000010.stcheckpoint`, etc. Existing destinations are rejected before
optimization begins, and each file is committed with atomic no-clobber
publication. If a subsequent step fails, previously completed checkpoints
remain available. A failed run is not a successful training report.
The maximum is 64 periodic files and a conservative 512 MiB total output
budget per invocation. Individual files are limited to 64 MiB.

## State and exact continuation

Each checkpoint binds:

- Source archive SHA256 and all three freshly audited eligibility fingerprints.
- Training configuration excluding only the target total step count.
- Initial model fingerprint and trained model fingerprint.
- CPU Float32 model parameters, including all state tensors.
- Single-group AdamW first and second moments and per-parameter step counts.
- CPU random-generator state, checked against a separate SHA256.
- Balanced sampler position and replay-verified index trace.
- Completed optimizer-step count and accumulated scalar loss summary.
- Runtime versions, CPU capability, thread counts and deterministic settings.

The restore procedure reconstructs the known model and optimizer first,
verifies the externally retained whole-file digest, then verifies every tensor
against the expected shape, type, byte count, order, offset and SHA256.
The restored optimizer is populated only with validated numerical tensors.
No pickle, torch.load, arbitrary object deserialization or checkpoint-controlled
optimizer options are used.

The supported continuation scope is the same CPU runtime and training settings.
Exact equivalence is a regression-test goal, not a guarantee for different
machines, PyTorch versions or device backends. The training loop preserves
the caller's CPU RNG state and validation uses independent random generators.
A completed checkpoint can also be evaluated without additional optimizer steps.

## Binary format v1

The byte layout is eight ASCII magic bytes STCP0001, a little-endian unsigned
32-bit JSON header length, canonical UTF-8 JSON, then contiguous tensor
payloads in lexicographic name order. The header is at most 256 KiB and
contains metadata and descriptors. Tensor payloads have only little-endian
Float32 or unsigned 8-bit representations. Scalars have an empty shape.
Every descriptor includes its exact name, dtype, shape, relative offset,
length and SHA256. No padding, overlapping ranges or trailing bytes are
accepted. A schema derived from trusted model code sets allocation bounds.

Whole-file SHA256 is an integrity pin, not a signature or proof of ownership.
A malicious producer could create a consistent but untrustworthy checkpoint.
Only use checkpoints obtained from your own approved local experiment.
Do not publish them, even if the code is publicly licensed: the model state
may reflect private training data.

## Verification

The CPU regression suite covers deterministic format round trips, malformed
headers and tensor descriptors, corruption even after internal rehashing,
optimizer and RNG checks, exact interrupted/resumed comparison, output
preflight, periodic saves and source-release exclusion. The source-only
release gate refuses private archive, training checkpoint and scratch
suffixes even if somebody force-adds them to Git.

```bash
python -m pytest -q tests/test_archive_checkpoint.py tests/test_archive_training.py
```

No claim of perceptual model quality, commercial training readiness, or
Sonumio-1 completion follows from these tests.


## Read-only checkpoint audit

Validate a recovery file without running training:

```bash
python -m research.archive_checkpoint_audit private.scorpus \
  --archive-sha256 ARCHIVE_SHA256 prefix.stcheckpoint \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --steps 20 --batch-size 4 --channels 8 --diffusion-steps 8
```

The audit performs a fresh corpus readiness pass, reconstructs trusted
model and AdamW state, verifies the sampler replay and returns only
aggregate fingerprints and progress. It does not write anything.

To compare interrupted and uninterrupted runs at the same completed
step, add `--compare second.stcheckpoint --compare-sha256 SECOND_SHA256`.
An optional `--require-identical` returns a nonzero exit code unless all
same-step numerical state matches. The comparison checks model, optimizer,
CPU RNG, sampler, progress, initialization and dataset/configuration pins
separately, rather than mistaking an equal model fingerprint for equal
optimizer state. Different-step snapshots return a null equivalence
result. A successful check does not prove ancestry, rights, or quality.

All options except the target total steps must match the original run.
The specified steps must be at least the largest completed step among
the audited files. This tool never imports pickle or executes an optimizer
update; caller CPU RNG state is preserved. Do not use the aggregate report
to publish the underlying private artifacts.
