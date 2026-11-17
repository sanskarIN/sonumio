# Sonumio

**Imagine it. Hear it.** Open-source generative audio platform for creating speech, music, and sound effects from text prompts, powered by evolving AI models.

> Experimental development: the default procedural DSP synthesizer generates actual WAV audio, but is NOT an AI model. Original Sonumio-1 does NOT exist yet.

## Setup
Python 3.11+:
1. `python -m venv .venv`
2. Activate the environment; on Windows run `.venv\Scripts\activate`
3. `python -m pip install -e '.[test]'`
4. `sonumio generate "Rain and thunder in a forest" --seconds 3 --output outputs/rain.wav`
5. `python -m pytest -q`
6. `sonumio serve` then open http://127.0.0.1:8000/docs

The procedural baseline supports rain, thunder, wind, birds, water and ambient tones in English/Hindi. Unsupported prompts raise an error. Sample format: 24 kHz mono PCM WAV.

The optional Meta MusicGen-small adapter is a THIRD-PARTY pretrained text-to-music model, research-only, disabled by default. Meta weights are CC-BY-NC-4.0 and not covered by Sonumio's Apache-2.0 source license; you must read and respect the upstream terms and separately install research dependencies. Set `SONUMIO_ENABLE_MUSICGEN=1` to enable. This adapter has NOT undergone real model weight/GPU validation in this prototype.

API: `GET /health`, `GET /v1/models`, `POST /v1/generations` and `GET /v1/generations/{id}/audio`. Do not expose the local server publicly without authentication, quotas, rate limits and cleanup.

Original Sonumio-1 and later model generations are research goals, NOT software version numbers or existing trained weights. See `docs/` for architecture, evaluation and roadmap. All releases/model weight publications need owner approval.

Source license: Apache-2.0. External weights and datasets retain their own terms.

## Original-model research scaffolding (development only)

From the repository root, after installing the optional PyTorch research dependencies:

- `python -m research.spectral_smoke --steps 40 --seed 7`: trains a tiny synthetic-spectrum regression network, records held-out error and saves a local NPZ checkpoint. This is **not** a neural waveform generator and not Sonumio-1.
- `python -m research.dataset_manifest path/to/manifest.jsonl`: checks data-rights declarations, voice exclusions, recording/checksum duplicates and group-isolated evaluation splits. This cannot verify legal rights without human review.

The MusicGen adapter uses a pinned upstream revision with safetensors loading and remote code disabled. Actual neural checkpoint execution is still unverified.

## Development-only performance measurements

Each successful generation reports `generation_wall_seconds`, `real_time_factor`, peak/RMS dBFS and the hard-clipping fraction. Run `python benchmarks/latency.py` after installing the package. The benchmark measures **procedural CPU synthesis**, not pretrained or original neural model latency.

## Sound scene generation (experimental procedural mixing)

The `scene` command and `POST /v1/scenes` can position up to eight DSP sound
layers on a 30-second timeline with per-layer fades, start time, seed and gain.

```bash
sonumio scene examples/forest_storm_scene.json --output outputs/forest_storm.wav
```

See [scene documentation](docs/SCENES.md). This is an audio mixing feature, not
neural sound generation.

## Original generative-model research experiment

An initial **trainable waveform diffusion** experiment now synthesizes very short,
8 kHz waveforms from six category conditions. Its training examples are created
by Sonumio's procedural engine. This is an experimental neural generator, **not
Sonumio-1** and not production-quality prompt-driven audio.

```bash
python -m pip install -e '.[research,test]'
python -m research.wave_diffusion train --steps 50 --output checkpoints/wave-diffusion
python -m research.wave_diffusion sample --checkpoint checkpoints/wave-diffusion --prompt rain --output outputs/research-wave.wav
```

See [diffusion research limitations](docs/DIFFUSION_EXPERIMENT.md). Locally trained
weights are excluded from Git, and no weights, releases or pre-releases are authorized.

## Rights-gated local WAV training (experimental)

The research loader verifies local WAV SHA256, PCM headers, declared permissions,
and recording-group-disjoint train/validation splits. Anti-aliased 8 kHz crops
can replace synthetic DSP examples for the tiny six-category diffusion experiment.
The loader is NOT independent copyright verification or voice detection.

~~~bash
python -m pip install -e '.[research,test]'
python -m research.wave_diffusion train --steps 50 --manifest /path/to/manifest.jsonl --audio-root /path/to/rights-reviewed-wavs --output checkpoints/reviewed-private
~~~

See [dataset ingestion requirements](docs/DATASET_INGESTION.md).
No GitHub release, pre-release, or model weight publication is authorized.

## Experimental two-sound neural research (not Sonumio-1)

The synthetic-only diffusion experiment now supports optional multi-label
conditioning for one or two fixed sound-effect categories in a limited
English/Hindi keyword vocabulary. It is learned class conditioning, not
open-vocabulary language understanding.

~~~bash
python -m research.wave_diffusion train --steps 50 --mixed-synthetic --output checkpoints/mixed-private
python -m research.wave_diffusion sample --checkpoint checkpoints/mixed-private --prompt "rain and thunder" --output outputs/mixed.wav
~~~

See [multi-sound research limitations](docs/MULTI_CONDITION_EXPERIMENT.md).
Keep checkpoints private; releases and pre-releases remain prohibited.

## React audio studio (local development only)

A new [React + TypeScript web interface](web/README.md) provides a sound-prompt
workspace, a bounded scene-layer editor, API status, model provenance,
playback and WAV download for the procedural baseline. It **does not**
claim to run Sonumio-1 or a trained neural text-to-audio model.

Start `sonumio serve` from the repository root. In a second terminal, run
`cd web && npm install && npm run dev`, then open
`http://127.0.0.1:5173`. The Vite dev proxy uses loopback `/v1` and
`/health` endpoints. For checks, run `npm run typecheck`, `npm test` and
`npm run build` from `web/`. This is not hardened for public deployment.

No GitHub releases or pre-releases are authorized.

## Bounded background audio jobs (experimental)

The local API now includes an in-process single-worker job queue for procedural
sound generation and scene composition. Submit `POST /v1/jobs/generations`
or `POST /v1/jobs/scenes` and poll `GET /v1/jobs/{id}`.
Cancel with `DELETE /v1/jobs/{id}`. The queue admits at most four active or
waiting jobs, expires local job outputs after 30 minutes while running, and
rejects unsupported models before dispatch. Running generation cannot be
forcibly interrupted, and jobs do not survive a server restart.

See [development job limitations](docs/LOCAL_JOBS.md). This local-only
prototype has **no production authentication or durable model scheduling**.
No releases, pre-releases or weight distribution are authorized.

## New learned-text neural audio research (NOT Sonumio-1)

The isolated `research/text_audio.py` experiment combines a **trainable
UTF-8 text Transformer** with an original conditional waveform diffusion
denoiser. It accepts English and Hindi text without hard-coded category IDs
inside the neural forward/sampling path. However, its **synthetic training**
still covers only six DSP sound categories and 15 two-sound pairs, with
template-based bilingual captions and 0.256-second 8 kHz outputs.

Local smoke test (not a usable creator model):

```bash
python -m pip install -e '.[research,test]'
python -m research.text_audio train --steps 50 --channels 16 --diffusion-steps 8 --output checkpoints/private-text-experiment
python -m research.text_audio sample --checkpoint checkpoints/private-text-experiment --prompt 'बारिश और हवा' --output outputs/text-experiment.wav
python -m research.text_alignment --checkpoint checkpoints/private-text-experiment --compare-initialization
python -m pytest -q tests/test_text_audio.py
```

The experiment also supports **rights-declared local WAV captions**, preserving
recording-group-isolated train/validation splits through the existing validator.
Passing metadata gates is NOT independent permission or voice verification.
Numerical/JSON checkpoints are SHA256-checked and remain private. The
caption-retrieval benchmark measures whether a matched English or Hindi
caption helps predict noise on held-out synthetic clips. Audio quality
and natural-language prompt adherence require separate evaluation.

See [limitations and methodology](docs/TEXT_CONDITIONING_EXPERIMENT.md).
No model weights, tags, GitHub pre-releases or releases are authorized.


## One-second latent diffusion research (development branch only)

A new **independent trainable waveform codec + bilingual text-conditioned latent
DDPM** prototype explores 1.024-second, 8 kHz mono sound effects. It has a
64× temporal compression bottleneck, multiresolution spectral reconstruction
loss and hard-negative English/Hindi caption retrieval checks. It uses only
synthetic DSP examples and **does not** demonstrate creator-quality output,
long-form music, speech synthesis, or a finished original model.

~~~bash
python -m research.latent_audio train --ae-steps 10 --diffusion-steps 10 --schedule-steps 8 --output checkpoints/private-latent-experiment
python -m research.latent_audio sample --checkpoint checkpoints/private-latent-experiment --prompt 'rain and thunder' --output outputs/latent-experiment.wav
python -m research.latent_alignment --checkpoint checkpoints/private-latent-experiment
~~~

See [isolated latent audio research methodology](docs/LATENT_AUDIO_EXPERIMENT.md).
All checkpoints remain local. No release, pre-release or weight publication.


### Research evaluation controls

The draft latent-audio work includes a separate
[objective reconstruction benchmark](docs/LATENT_AUDIO_EXPERIMENT.md#objective-reconstruction-and-model-health-evaluation)
that checks all unique held-out synthetic recordings against silence,
identity and moving-average DSP controls. These numerical diagnostics do not
show that generated sound is natural, prompt-aligned or release-ready.


### Shared-data raw-vs-latent architecture study

The draft research branch also includes an isolated
[architecture comparison](docs/ARCHITECTURE_COMPARISON.md) that runs the
raw-waveform and latent text-conditioned DDPM prototypes on the same
1.024-second, 8 kHz synthetic recordings. It records caption-conditioned
noise diagnostics, parameter counts, separate codec cost and CPU timings.
The two branches are **not** matched on FLOPs or parameter count, and their
absolute noise MSE values **must not** be ranked against each other.

See the [project development status and remaining gates](docs/PROJECT_STATUS.md).
No audio model weights or release artifacts are created by this benchmark.


### Replicated raw/latent audio research

The draft branch adds a [multi-seed research protocol](docs/MULTI_SEED_STUDY.md)
to repeat controlled raw-vs-latent DDPM experiments with three or more
prespecified initialization seeds. It reports **within-model** prompt/blank/
zero-predictor comparisons, observed seed variability and exact data
fingerprints; it never compares raw and latent noise MSE directly to pick a
winner. It cannot certify generated-audio quality or model release readiness.

The [recorded three-seed CPU pilot](docs/MULTI_SEED_PILOT_RESULTS.md)
documents measured training diagnostics and the remaining evidence gaps.
It does not demonstrate creator-quality generation or justify a release.


### Extended training on the currently permitted synthetic dataset

An isolated [128-codec / 256-denoiser-step multi-seed training protocol](docs/EXTENDED_SYNTHETIC_TRAINING.md)
runs on Sonumio-owned sound samples using all 21 source combinations
in held-out evaluation. It is research only, not high-quality audio generation,
and produces no published model weights. Do not confuse synthetic audio with
a rights-cleared, independently validated real-world recording corpus.


**Measured training evidence:** the [extended synthetic training report](docs/EXTENDED_SYNTHETIC_RESULTS.md)
documents a verified three-seed run (128 codec and 256 denoiser updates,
full 21-combination validation) and honest limitations. All four CI
workflows passed; no trained neural weights or model releases were published.


### Actual generated-waveform diagnostics (experimental)

The development branch now uses an [in-memory generation audit](docs/GENERATED_AUDIO_AUDIT.md)
to inspect waveforms produced after neural training, not only DDPM
noise-prediction loss. This checks silence/clipping/spectral health and
whether different bilingual prompts yield different generated samples.
It **does not** prove realistic sound effects or reliable prompt meaning,
and saves neither WAV samples nor model weights.


**Generated waveform audit findings:** the
[actual in-memory sampling pilot](docs/GENERATED_AUDIO_PILOT_RESULTS.md)
found substantial clipping in raw DDPM samples and weak numerical
prompt-change sensitivity in latent samples. The eight-step linear
schedule also has a serious terminal-noise/initial-prior mismatch risk.
These are *research blockers*, not finished Sonumio capabilities.


### Controlled diffusion-prior schedule experiment

The draft branch now has a [matched linear-vs-cosine schedule ablation](docs/SCHEDULE_ABLATION.md).
It holds Sonumio-owned training data, model random seeds, optimizer budgets,
source combinations and actual generated-waveform diagnostics fixed while
testing whether near-zero terminal noise reduces the previously measured
clipping. Results must be checked before claiming any improvement. Neither
noise-MSE reduction nor waveform prompt sensitivity by itself proves quality.


**Measured schedule result:** the
[paired three-seed ablation](docs/SCHEDULE_ABLATION_RESULTS.md)
found that the experimental cosine schedule made generated waveforms
dramatically **more clipped** (raw 13.7% → 94.6%; latent 0% → 63.9%).
The cosine schedule is **not** recommended or selected. Numerical CI
success and lower terminal SNR do not certify generated audio.


### Objective-matched velocity-prediction follow-up

The draft research branch now includes a
[velocity-versus-epsilon objective experiment](docs/VELOCITY_OBJECTIVE_STUDY.md).
Both networks begin from identical parameter tensors, share the same
synthetic training examples, Gaussian noise, timesteps, text dropout masks
and cosine schedule, and differ only in target and corresponding reverse
DDPM sampler. The experiment measures unclipped generated-waveform
amplitudes and numerical clipping risk. **No model quality improvement
is claimed before the actual training evidence is inspected.**


**Measured neural-generation improvement:** the
[paired velocity-objective results](docs/VELOCITY_OBJECTIVE_RESULTS.md)
report raw-waveform clipping falling from **94.64% to 0.004%**
and latent internal-Tanh saturation from **71.13% to 0%**
under matched cosine-schedule training. This is *numerical
signal-health improvement*, not proof of realistic audible sound,
prompt fidelity, or Sonumio-1 readiness. No weights were published.


### Synthetic research data reproducibility gate

The development branch includes a [recording-level data reproducibility audit](docs/DATA_REPRODUCIBILITY.md)
to inspect exact SHA256 hashes of all **84 unique** first-party DSP
train/validation recordings, plus separate coarse quantized diagnostics.
CI checks independently regenerated data across Python processes and
compares actual model-training source hashes with an independent preflight
manifest. This addresses unresolved cross-run fingerprint drift without
altering model input audio or releasing recordings/weights.


**Observed reproducibility evidence:** the
[CPU-run comparison](docs/DATA_REPRODUCIBILITY_RESULTS.md)
confirmed bit-identical DSP data across independent processes on
one runner and exact preflight/model-training agreement, while
different AMD EPYC runner models produced **different exact dataset
hashes**. Coarse 0.01 amplitude-bucket hashes matched, but that does
not prove byte identity or a maximum numerical-error bound. The model
and data recipe remain research-only and unchanged.


**Stage-level DSP reproducibility:** a [first-party pipeline trace](docs/DSP_PIPELINE_TRACE.md)
now fingerprints each source layer, mixture, fade, peak normalization,
FIR kernel, filtered signal, decimation, and final training crop.
It checks all stages against the unmodified procedural renderer and
records differences between CPU runners without changing the original
dataset or publishing audio/weights. Results must be interpreted as
numerical provenance, not audible quality.


A [measured multi-CPU DSP study](docs/DSP_PIPELINE_TRACE_RESULTS.md)
verified exact original-audio stage identities across one AMD EPYC
9V45 and three Intel Xeon hosts, with independent-process checks
on all 84 source recordings. The previously differing AMD EPYC 7763
environment was not represented; the root cause remains open.
A mixed Ubuntu 22.04/24.04 matrix now broadens the CPU sampling.


**CPU source-level numerical results:** the
[reproduced EPYC 7763 divergence](docs/DSP_PIPELINE_TRACE_RESULTS.md)
is isolated to rain, thunder and wind before mixing; forest,
water and ambient source hashes matched. A
[targeted convolution study](docs/DSP_CONVOLUTION_ISOLATION.md)
now checks the original float32 random inputs, 12/60/140/180-tap
kernels and outputs without changing model-training audio.


**Verified original DSP numerical root:** the
[matched convolution investigation](docs/DSP_CONVOLUTION_RESULTS.md)
showed identical original float32 input/kernel bytes but differing
NumPy convolution output bytes on AMD EPYC 7763 for rain (180 taps),
thunder (140) and wind (60). The 12-tap water control and the
independent selected \`math.fsum\` references matched. No original
audio recipe or neural training outputs were modified; full-audio
perceptual relevance of ~1e-8 sampled differences is untested.


**Research-only scalar convolution candidate (17 November):** the
[isolated reproducibility study](docs/SCALAR_CONVOLUTION_CANDIDATE.md)
computes selected output positions using Python scalar binary64 products,
compensated summation and a single float32 conversion. It checks input/kernel
hashes against the unchanged original DSP and adds independent CPU-matrix
fingerprints; selected-sample agreement cannot establish full-audio identity
or creator-quality generation. No existing data recipe or training code
switches to this candidate.
