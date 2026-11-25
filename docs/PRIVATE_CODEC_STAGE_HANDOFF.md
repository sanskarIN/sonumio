# Completed codec stage handoff (private research)

The codec stage can be completed in a separate local invocation. It writes
a bounded numeric-only private .ckpt file only on explicit request:

```bash
python -m research.archive_latent_study private.scorpus --expected-sha256 ARCHIVE_SHA256 \
  --codec-steps 32 --batch-size 4 --bits 8 --codec-out checkpoints/codec.ckpt --codec-only
```

Keep the returned codec_handoff.receipt.sha256 independently. Use an existing
trusted local output directory. The writer uses a private temporary file and
a no-overwrite hard-link publication; filesystems without hard links fail
closed. Directory fsync/power-loss durability is not guaranteed.

```bash
python -m research.archive_latent_study private.scorpus --expected-sha256 ARCHIVE_SHA256 \
  --codec-steps 32 --batch-size 4 --bits 8 --codec-in checkpoints/codec.ckpt \
  --codec-sha256 CODEC_SHA256 --diffusion-updates 64
```

The loader verifies the same immutable file snapshot and strict bounded
numeric NPZ layout without pickle. The canonical envelope binds the archive
and training eligibility, codec-stage settings, runtime, initial and trained
state hashes, sampler draw trace and calibration identity. Calibration is
recomputed using the audited training split. No supplied metadata is used
to allocate model architecture. A disposable candidate is validated before
use. No original initialization is modified on error.

The handoff allows different denoiser budgets but rejects changes to the
codec stage or runtime. Regression tests compare fresh two-stage training
with restored-stage training in the same CPU runtime. This is NOT mid-stage
optimizer resume, cross-platform bitwise reproducibility, a cryptographic
signature, or proof of the claimed training history. Only use trusted
private artifacts. No audio files are saved, and the source release gate
already rejects .ckpt files. The default run writes nothing.
