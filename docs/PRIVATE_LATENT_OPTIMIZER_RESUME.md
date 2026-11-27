# Private latent optimizer continuation

Experimental CPU research. Reuse the completed-codec handoff to avoid codec
retraining, then write a latent denoiser checkpoint at a completed optimizer
update. Existing directory required. Example (replace digest placeholders):

```bash
python -m research.archive_latent_study private.scorpus --expected-sha256 ARCHIVE_SHA \
  --codec-in checkpoints/codec.ckpt --codec-sha256 CODEC_SHA \
  --diffusion-updates 64 --latent-out checkpoints/latent64.stcheckpoint
```

Retain the returned latent_checkpoint.receipt.sha256 independently. Extend
the trajectory to 128 total updates:

```bash
python -m research.archive_latent_study private.scorpus --expected-sha256 ARCHIVE_SHA \
  --codec-in checkpoints/codec.ckpt --codec-sha256 CODEC_SHA \
  --diffusion-updates 128 --latent-in checkpoints/latent64.stcheckpoint \
  --latent-sha256 LATENT_SHA --latent-out checkpoints/latent128.stcheckpoint
```

The target is total updates, not additional updates. The same audited
corpus, codec and calibration fingerprints, architecture, seed, batch
size, schedule, runtime and AdamW contract are required. The completed
step count, loss history, sampler and counter RNG trace are validated
against a deterministic replay of training indices and condition/noise
draws. The replay needs no held-out waveform or old optimizer execution.
All model and optimizer tensors use bounded numeric raw checkpoints, with
a trusted external SHA256 and per-tensor digest checks. A disposable
candidate is validated before loading. Output publication is no-clobber.

Tests compare model, optimizer-driven continuation, loss history,
validation and generated-signal results in one CPU runtime. This is not
a proof of historical training, a signature, or a guarantee of
cross-hardware bitwise reproducibility. There is no automatic periodic
saving or interrupted partial-update recovery yet. Default invocations
write nothing. Never commit private artifacts or publish model weights
without owner authorization.
