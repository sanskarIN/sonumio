# Sonumio: actual global-history neural conditioning on adjacent audio

**Experimental original research, first-party synthetic DSP only; no weights, WAV files, model identities or releases.**

## Combining two verified foundations

The exact-adjacent source dataset provides two 8192-frame windows taken
from the same longer first-party recording, with an exact shared prefix.
The new full-history v-denoiser separately encodes the entire previous
8192-frame clip into a global context vector that can reach all future
residual blocks. This study combines them in **actual optimizer training**.

A local-prefix-only model cannot distinguish two histories that share
exactly the same final P samples; the explicit full-window history
encoder can. But the existence of that path is not evidence that it
will *learn* a useful coherent continuation.

## Model variants and isolation

`research/global_history_study.py` creates two copies of the same
original neural model, initialized with bit-identical weights:

- `full_history`: receives the real preceding 8192-frame recording.
- `hidden_history`: receives an all-zero history signal.

The two models receive the **same** actual future waveform target,
same encoded English/Hindi caption batch, same Gaussian corruption,
same sampled diffusion timesteps, same optimizer and learning rate,
same gradient norm clipping, and the **same future-only v-MSE**.
Only access to earlier audio differs. All known overlap samples are
clean and fixed; hidden-history input uses zero prefix during training
and held-out inference, a deliberate different conditioning condition.

### The strict wrong-remote control

A third **evaluation-only** condition asks the trained full-history
model to predict the same future with an unrelated preceding recording.
Crucially, this wrong remote history has its final P samples replaced
with the **correct known prefix**.

```text
Correct history: [TRUE distant past ... TRUE final P samples]
Wrong history:   [WRONG distant past ... TRUE final P samples]
Future input:    [TRUE final P samples ... identical noisy future]
Timesteps:       exactly identical for both
Text:            exactly identical for both
Model:           exactly the same frozen trained full-history model
```

Thus any different output cannot be blamed on replacing the known
prefix, changing the generated-noise seed, moving the clip boundary,
different text, or different network weights. It is a clean numerical
test of sensitivity to **remote preceding context**. Sensitivity is
not the same as correct *use* of history.

The held-out experiment evaluates *all* bilingual examples, while
per-recording grouped effects count English and Hindi variants of
the same waveform as one acoustic group. It reports:
- full-history and hidden-history held-out future v-MSE,
- wrong-remote-history future v-MSE on the same model,
- descriptive hidden-minus-full and wrong-minus-correct effects,
- group-level positive/negative counts,
- generated-vs-true future sample-aligned error,
- first-unknown-sample discontinuity, and
- generated-output differences under the remote-only perturbation.

These are **nonperceptual** numeric controls. A small MSE is not proof
of natural sound, speech/music continuity, prompt meaning, or
a good long-form model. The project still uses 8-kHz 8192-frame
recordings and a tiny v-denoiser.

## Independent initialization-seed replication

`research/global_history_replicates.py` trains separately from
2 to 4 independent model seeds using exactly the same first-party
source recordings and all non-seed hyperparameters.

Before aggregating any numbers, it validates:
- exact canonical configuration SHA256,
- same source recordings and train/heldout fingerprints,
- complete source grouping and 42/42 independent recording counts
  for the standard two-seed split,
- exact known prefix in wrong remote context,
- matched noisy future and timesteps for all held-out conditions,
- finite and complete neural training history,
- distinct initial and trained model weight fingerprints per seed,
- complete generated future sample grid with numeric metrics,
- no unearned perceptual or publication flags.

It reports model-seed-level descriptive means, medians, standard
deviations and observed min/max ranges. These are **not** confidence
intervals, p-values or model-quality preferences. Multiple generated
noise seeds and two language captions from one audio are not extra
independent trained-model replicates.

## CPU research commands

```bash
python -m pytest -q tests/test_global_history_study.py \
  tests/test_global_history_replicates.py

python -m research.global_history_study \
  --model-seed 17 --updates 2 --prefix-frames 512 \
  --diffusion-steps 8 --evaluations 2 \
  --training-seeds 0 --heldout-seeds 101

python -m research.global_history_replicates \
  --model-seeds 17,29 --updates 2 --prefix-frames 512 \
  --diffusion-steps 8 --evaluations 2 \
  --training-seeds 0 --heldout-seeds 101
```

Each synthetic recording seed covers 21 source combinations. One
training seed and one held-out seed mean 21 unique recordings each,
with 42 English/Hindi caption examples per split. Two seeds of
recordings per split cover 42 independent recordings, with 84 caption
examples each.

The **model seed** is separate from recording seeds and is the
independent neural replicate. The model's original learned history
encoder gets real backpropagated gradients and is never silently
replaced by a pretrained representation.

## Why this is not ready for publishing trained audio

The original synthetic DSP source families are still correlated and
simplistic, including seed-dependent amplitude-envelope augmentation
for otherwise deterministic oscillators. The neural denoiser sees only
one previous 8192-frame clip and predicts a similarly short future
window. A model trained for 2–64 gradient steps is not converged.

A future credible Sonumio-1 needs substantially more varied,
rights-cleared source- and speaker-disjoint data, longer temporal
receptive fields, independent listener evaluation, prompt-semantic
benchmarks, 24–48 kHz output quality, real generation stability,
inference efficiency and robust model provenance.

Passing this research does not authorize any new software release,
pre-release tag, weight publication, trained model identity, WAV
asset, or declaration of speech/music readiness.
