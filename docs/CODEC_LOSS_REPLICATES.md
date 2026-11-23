# Sonumio codec objectives: independent model-seed replication

**Source-only, original small CPU neural codec research. No trained weights, generated audio or release assets.**

## Why a single paired codec study is insufficient

The Sonumio codec objective pilot trains two copies of the same original
tiny 8-kHz neural audio autoencoder from matching initial weights. The
original codec uses waveform L1 plus three log-magnitude STFT losses;
the candidate includes separate waveform, transient, envelope, spectral,
highband and spectral-flux components.

One trained model can appear numerically better because of a particular
initialization or one synthetic sampling minibatch. Two English/Hindi
captions of one waveform are certainly not two independent acoustic
observations. A few seed repetitions do not justify significance claims,
but they can reveal obvious instability and enable honest comparison.

## What this replication actually verifies

`research/codec_loss_replicates.py` repeats the genuine original
CPU neural training experiment for **2–4 independent model-initialization
seeds**, with all source recordings and other hyperparameters matched.

Before combining numerical effects, it audits each report:
- exact single-study protocol and canonical configuration SHA256;
- exactly the requested learning rate, minibatch size, optimizer steps,
  source recording seeds, objective weights and FFT configuration;
- identical first-party training/heldout source fingerprints and exact
  heldout recording hashes across model seeds;
- two different training initializations per model seed, with **both
  compared codecs starting from exactly the same weights**;
- different model initializations across independently seeded runs,
  and different actual trained codec state SHA256 fingerprints;
- one heldout audio recording per source-family/recording-seed pair:
  six individual categories plus 15 pairs, without language duplicates;
- all per-recording sample MAE/RMSE/SNR, original-vs-candidate component
  deltas and source-family grouping, with finite values and matching
  report arithmetic;
- explicit source family and recording-seed identity on every sample;
- no forbidden trained weights, WAV, model identity or perceptual claims.

A missing row, duplicated category group, changed recording seed, forged
aggregate, NaN, model-state collision or source-fingerprint drift **fails**
the entire replication. It does not silently proceed with a smaller or
incomparable set of recordings.

## Evidence reported

The canonical JSON report contains one record per independent model seed.
For each training pair it reports:
- starting and final SHA256 model identities (not weight tensors),
- trained candidate-minus-original heldout waveform MAE/RMSE,
- candidate-minus-original heldout SNR,
- candidate-minus-original **source-family grouped** deviations on each
  of the eight exact differentiable codec reconstruction components,
- exact configuration identity.

Across model initializations it uses only descriptive mean, median,
sample standard deviation, minimum, maximum and sign counts. No
confidence interval, p-value or winner is attached.

The sign of a reconstruction component delta means only that the
candidate trained with a different objective produces different
numerical error on the fixed synthetic evaluation recordings.
It is **not** proof that its waveform sounds natural or intelligible.

## Reproduce

```bash
python -m pytest -q tests/test_codec_loss_replicates.py

python -m research.codec_loss_replicates \
  --model-seeds 17,29 --updates 2 \
  --training-batch 2 --training-seeds 0 \
  --heldout-seeds 101
```

The smoke evaluates 21 unique heldout recordings per independent
model initialization, representing 21 DSP combination families. It
does not count English/Hindi caption duplicates or the original and
candidate neural arms as extra model-seed replicates.

For a somewhat broader, still nonconverged synthetic diagnostic:

```bash
python -m research.codec_loss_replicates \
  --model-seeds 17,29,41 \
  --updates 32 --training-batch 4 \
  --training-seeds 0,1 --heldout-seeds 101,102
```

The latter uses 42 separate acoustic recordings in each split,
although its 21 DSP sound-combination families remain related.

All training and validation are in memory. The CLI prints bounded
numeric JSON to standard output without audio, weights, embeddings,
external data downloads, checkpoint files or model-release artifacts.

## Prerequisites for a quality claim

A public Sonumio model should eventually be judged on rights-reviewed
real-world music/speech/sound-effect data; source/speaker-disjoint
validation; meaningfully longer convergence training; independent
retrainings; higher sample rates; reconstruction versus *generated*
audio distinctions; blind human ratings; and robust alignment metrics.

No part of this codec study authorizes an original Sonumio-1 model,
software release, public trained weights or a perceptual-quality claim.
