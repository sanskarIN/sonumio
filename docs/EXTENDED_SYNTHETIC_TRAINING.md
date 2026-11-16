# Extended training: Sonumio-owned synthetic sound recordings

**Status:** experimental training, not a Sonumio model-generation release.

This workflow actually trains Sonumio's original raw-waveform and
convolutional-latent text-conditioned neural diffusion prototypes, rather than
only checking that a model can run a forward pass. It uses **only Sonumio's
original procedural DSP** as source data, without downloading data, voices,
music, third-party model weights, or unverified recordings.

The dedicated GitHub Actions workflow:
[Private synthetic audio training (CPU)](../.github/workflows/synthetic-training.yml).

## Fixed experiment plan (chosen before measuring results)

| Parameter | Configuration |
| --- | --- |
| Dataset provenance | Sonumio procedural sound renderer |
| Sound categories | rain, thunder, wind, forest birds, ocean, ambient |
| Combinations | 6 single categories + 15 two-source combinations |
| Train recording seeds | 0 and 1 |
| Validation recording seeds | 101 and 102 |
| Train recording count | 42 distinct WAV waveforms, 84 English/Hindi caption pairs |
| Validation count | 42 distinct WAV waveforms, 84 bilingual caption pairs |
| Temporal representation | 1.024 s at 8 kHz mono; raw 1×8192, latent 16×128 |
| Initialization seeds | 17, 29, 41 |
| Training steps per seed | 128 codec updates, 256 per denoiser |
| Diffusion schedule | 8 timesteps, separate from training optimizer steps |
| Model width and batch size | 16 channels, batch of two |
| Held-out evaluation | all 21 source combinations, 84 validation captions |
| Language controls | correct, category-disjoint wrong, blank |
| Zero-noise-prediction baseline | yes, within each representation |
| Model-weight output | **none** |
| Final output | ephemeral JSON report and GitHub Actions diagnostic log |

The run uses a CPU-only Linux worker with two PyTorch threads. Workflow
timeout is 20 minutes. If the job fails or times out, do not claim it trained
successfully; inspect the run and fix the failure before using any metrics.

**Why synthetic first?** The owner has not supplied a rights-reviewed
real-world dataset. Procedurally generated sounds provide a verifiable
first-party training substrate and regression baseline. They do not prove
perceptual fidelity, consent validity for any future recordings, nor
generalization to real acoustic environments.

## Evaluation controls and interpretation

All training updates are applied in memory. No checkpoint is serialized.
Each arm sees the same waveforms and minibatch indices, but with an independently
initialized text encoder, different noise representation and different
network sizes. The latent arm incurs extra codec training cost. The workflow
explicitly avoids comparing their raw versus latent MSE as if they were
equivalent audio-quality measurements.

The results include per-seed and aggregated denoising MSE, differences from
**category-disjoint captions**, differences from unconditional/blank text,
noise-prediction improvement against a zero predictor, and codec reconstruction
MAE. Results are reported as descriptive means, medians, standard deviations
and observed ranges within each arm. Three initialization seeds on one DSP
dataset are not a population-level quality study or an independent listening
panel.

The program asserts that:

- Exactly three designated initialization seeds ran with the specified counts.
- Every held-out sound combination was evaluated.
- All correct/incorrect captions come from a reproducible checked protocol.
- There is no exact training/evaluation waveform overlap.
- Only a metrics JSON file exists under the runner's ignored checkpoint folder.
- No model generation, release, model weight, archive or artifact is produced.

Run this same experiment manually from the repo root with a private report:

~~~bash
python -m pip install -e '.[research,test]'
python -m research.multi_seed_study \
  --seeds 17,29,41 --codec-steps 128 --denoiser-steps 256 \
  --schedule-steps 8 --channels 16 --evaluation-combinations 21 \
  --output checkpoints/private-research/extended-synthetic-training.json
~~~

On Windows PowerShell use single-line commands or backticks instead of
backslash line continuations. Running this experiment yourself recomputes
weights in RAM and discards them after metrics. **No trained checkpoint is
provided**. You would need separate explicit consent, provenance review,
security checks and authorization before any later distribution decision.

## Remaining prerequisites before Sonumio-1

A human-reviewed, rights-cleared, diverse real-world corpus; independent
recording-session splits; higher sample rates and longer clip lengths;
trainable codecs with demonstrable reconstruction fidelity; multi-seed
convergence studies under matching compute budgets; neural waveform
sampling and blind listening; memorization, attribution and privacy reviews;
production resource limits; and explicit release approval.

Passing this synthetic training workflow is **not** evidence that the model
can create realistic SFX, speech or music for creators.
