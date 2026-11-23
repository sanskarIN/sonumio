# Sonumio clean-prefix velocity continuation research

**Experimental. Source-only, no weights, no WAV, no Sonumio-1 model identity.**

## Why this differs from the earlier overlap/crossfade experiment

`research/longform_ddim.py` joins independently sampled one-second audio
clips using a deterministic crossfade. It does not make the next clip
depend on actual prior audio. A crossfade is not a trained continuation model.

`research/velocity_prefix_continuation.py` instead treats the **first
128–4096 samples** of the next 8192-frame clip as a known, exact clean
prefix from the previous clip, and performs velocity-prediction DDIM
only for the **remaining future frames**. The clean prefix is placed in
the denoiser input at every training and inference timestep. This is
a distinct distribution from the existing unconditioned full-clip
velocity models, which must NOT be represented as trained continuation
models. It is 8-kHz mono raw waveform research, not latent generation.

## Matching training objective

For a clean clip `x0`, Gaussian noise `epsilon`, schedule `alpha_bar_t`
and prefix width `P`, build the regular DDPM noisy clip

```text
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
```

Replace `x_t[:P]` with **the original clean** `x0[:P]`, not noise or
a predicted reconstruction. Let the v-target be

```text
v_t = sqrt(alpha_bar_t) * epsilon - sqrt(1-alpha_bar_t) * x0
loss = MSE(model(x_t_with_clean_prefix, t, tokens)[P:], v_t[P:])
```

The target exists for all frames, but known prefix frames receive no
supervision. At inference each reverse-DDIM evaluation starts from the
same fixed known prefix; only the future section is updated by velocity
DDIM transitions. The final `x0[:P]` is copied exactly. The training
and sampling distributions match at the **clean prefix**.

This method guarantees **sample-level identity** of the retained overlap.
It does not guarantee that the first *new* sample connects smoothly, nor
that a short denoiser understands a longer sound scene. A boundary-jump
diagnostic reports the first unknown sample transition rather than
claiming it is improved.

## First real controlled training experiment

`research/velocity_prefix_study.py` trains two small original neural
networks from exactly equal initial weights using the same first-party
synthetic DSP recordings, bilingual caption minibatches, optimizer,
noise, timestep and known-prefix construction. Only the loss region
changes: unknown-future-only versus all 8192 frames.

Both models are evaluated on **all 84 held-out English/Hindi caption
examples representing 42 distinct audio recordings**; training has
42 disjoint audio recordings (84 captions). The held-out target is
future velocity prediction on the *same noise and timestep draws*.
A two-example same-seed DDIM pilot records output hashes, first unknown
sample discontinuity, signal peak and out-of-range fraction.
No WAV/weights/checkpoints are written, and trained model state hashes
are checked again after all evaluation.

```bash
python -m research.velocity_prefix_study \
  --seed 17 --updates 2 --prefix-frames 512 \
  --diffusion-steps 8 --evaluations 2
python -m pytest -q tests/test_velocity_prefix_continuation.py \
  tests/test_velocity_prefix_study.py
```

This is a protocol pilot, not converged training. A lower future-frame
v-MSE may be scientifically useful, but is **not a listening score,
naturalness, quality, speaker coherence, music structure, or prompt
adherence**. The raw denoiser supports only 8192-frame windows; the
sampler cannot confer long-range memory or speech understanding.

## Sequence extension and provenance

`sample_continuation_sequence` requires a **caller-supplied initial
clip** (finite CPU float32, 8192 frames, rights to use it). This is
intentional: a model trained only for masked-future prediction should
not be asked to generate an initial clip from scratch as if it had
been trained for that. The next clip's known prefix comes directly from
the tail of the caller's clip, and each later clip retains the preceding
tail exactly. The method appends only new frames; unlike crossfading,
it does not blend or overwrite the original source.

The final buffer has

```text
frames = 8192 + (segments - 1) * (8192 - prefix_frames)
```

with a maximum of eight segments. The returned report includes exact
known-prefix hashes, segment seeds, model-state and global RNG
preservation, per-join first unknown sample jump, output SHA256, and
explicit research/nonpublication flags.

The tests cover an independent mathematical exact-v oracle using two
noise schedules and three timestep budgets, actual original denoiser
training gradients, malformed inputs, UTF-8 captions, exact prefix
identity, deterministic seeds, RNG isolation and real multi-segment
shape/content controls.

## Remaining research prerequisites

A useful continuation model needs substantially longer training, more
diverse legally cleared audio, held-out source and speaker groups,
independent model-seed replication, long-context conditioning architecture,
real perceptual/listener testing and semantic or musical alignment metrics.
It is **not** enough that a continuity boundary has an identical prefix.
No Sonumio software pre-release, weight distribution or model generation
is authorized by this study.
