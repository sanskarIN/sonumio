# Sonumio research: matched EMA-decay ablation

**Status: experimental first-party synthetic audio; no model publication.**

This study expands the in-memory neural EMA training experiment with
independent model-initialization seeds and a prespecified choice of EMA
decay. It asks a narrower question than whether Sonumio can generate
convincing audio: **how do EMA decay choices change numerical v-prediction
error and generated-waveform health, when the online training trajectory is
demonstrably unchanged?**

## Matched design

- Use original velocity-prediction research architectures (raw waveform and
  neural latent with fixed trained codec), the same first-party procedural
  DSP training/held-out data, the same chosen model seeds, the same number
  of AdamW updates, and the same DDIM sampling seeds across decay settings.
- Use decay `0.0` as a **null control**: its EMA must equal the online
  weights. Held-out velocity MSE, waveform hashes, and clipping must match.
- For a given model seed, varying EMA decay must **not change online
  weights** (SHA256), **online held-out v-MSE**, or **every generated online
  waveform** (SHA256, RMS, clipping and prelimit diagnostics).
- Require a different online model state hash for each independent model
  initialization seed. Fail rather than counting duplicate model states
  as independent replications.
- Recompute each trial's configuration digest; verify source-group counts,
  42 independent recordings in training, 42 held out, 84 bilingual caption
  examples per split, and train/held-out exact separation.
- Validate all sample keys and numerical summary values; prohibit a false
  perceptual-quality, publication, or sample-authority claim.
- Aggregate **within each original model architecture only**; never pool
  raw- and latent-domain v-MSE.

There are two independent variations: initial model seed and EMA decay.
Audio-generation seeds, Hindi/English captions and shared audio sources
are **repeated measurements**, not independent trained models. The code
reports descriptive effect distributions across training seeds without
attaching unearned significance intervals.

## Experiments

For a 2 × 2 CI smoke with genuine synthetic training and inference:

```bash
python -m research.velocity_ema_decay_ablation \
  --seeds 17,29 --decays 0,0.5 \
  --steps 2 --codec-steps 1 --diffusion-steps 8 \
  --sampling-seeds 301
```

For a more substantive but still nonconverged local pilot:

```bash
python -m research.velocity_ema_decay_ablation \
  --seeds 17,29,41 --decays 0,0.5,0.9 \
  --steps 32 --codec-steps 8 --diffusion-steps 8 \
  --sampling-seeds 301,302
```

All training, neural weights, EMA shadow copies, and samples stay
in process memory. The CLI emits numerical JSON to standard output.
It does not save model weights, generated audio, or checkpoints.

```bash
python -m pytest -q tests/test_velocity_ema_decay_ablation.py
```

The key fields are `within_arm_descriptive_summary`,
`identical_online_trajectory_within_model_seed_verified`,
`zero_decay_null_control_passed`, and the original data/model state
fingerprints. A negative `heldout_mse_delta_ema_minus_online` means the
EMA shadow scored lower velocity-prediction MSE on that finite held-out
corruption draw, **not** that its audio sounds better. Clipping reductions
may accompany a muted or distorted output. This study does not test actual
English/Hindi audio semantics, independent human perception, speech,
musical structure, model originality or real-world licensed datasets.

Until long training, model-seed replication, independent listener
evaluation, rights-review and external validation are completed, these
pilot results do not justify publishing a Sonumio-1 weight or other model.
