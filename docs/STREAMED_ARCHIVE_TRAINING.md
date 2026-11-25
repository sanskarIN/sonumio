# Streamed private-archive waveform research

This follow-on to the private corpus archive performs numerical readiness,
group-balanced sampling and experimental CPU waveform DDPM training without
materializing the entire PCM corpus. The archive must come from approved
local recordings and be bound to an independently retained SHA256.

```bash
python -m research.archive_readiness private.scorpus --expected-sha256 APPROVED_DIGEST --require-ready
python -m research.archive_wave_training private.scorpus --expected-sha256 APPROVED_DIGEST --steps 2 --channels 8 --diffusion-steps 8
```

The readiness audit applies the same existing per-clip thresholds, exact
within-split duplicate exclusion, recording-group counts, and eligible
fingerprint protocol as the in-memory readiness audit. It retains metadata
and scalar summaries, processes one PCM shard at a time and emits aggregate
JSON only. It does not independently certify rights, voice absence or
recording independence. The test split is inspected for readiness and
integrity but never used for optimization or validation.

## Deterministic training indices

The counter sampler uses domain-separated SHA256 draws, with rejection
sampling rather than biased modulo reduction. Each six-category cycle is a
deterministic shuffled permutation; group, asset and crop are selected
uniformly in sequence within that category. The absolute draw position
allows random-access replay independent of batch partition. The plan is
bound to the eligible-training fingerprint and seed, not the archive's
shard packing. The trace binds the ordered int64 indices.

```python
sampler = CounterGroupSampler(readiness, seed=17)
first = sampler.draw(8)
state = sampler.describe()
restored = CounterGroupSampler.restore(readiness, state)
next_batch = restored.draw(8)
```

Restore verifies the plan and replays the bounded draw prefix. It is only
sampler state, **not** a model, optimizer, PyTorch RNG or full training
checkpoint. The maximum trace length is 400,000 draws.

## Paired validation

The trainer checks each fetched training row against audited provenance and
its PCM SHA256 before optimization. Validation uses eligible validation
rows only; per-example timesteps and noise are seeded from the validation
fingerprint, explicit seed, row index and repeat. Therefore paired models
receive the same corruption inputs regardless of batch partition. The
diagnostic compares model noise MSE with a zero-predictor control, reports
clip-weighted and recording-group macro averages, and per-category values.
The trace records actual generated noise bytes and timesteps. Floating
kernel results can vary slightly with batch geometry and platform.
Model state is fingerprinted before and after validation.

The command uses the original experimental single-class TinyWaveDenoiser,
CPU AdamW, bounded minibatches and gradient clipping. It preserves caller
CPU RNG state and writes no model checkpoints, audio or external data.
The default readiness gate must pass before optimization. This is a
reproducible training experiment, not a release-ready model, perceptual
quality evaluation, distributed trainer or original Sonumio-1.
