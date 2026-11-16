# Sonumio-1 model research — planned, not available

## Architecture comparison
| Approach | Advantage | Major cost | Decision |
| --- | --- | --- | --- |
| Procedural DSP | Fast CPU smoke checks | No learned semantics | Working baseline only |
| Neural codec + autoregressive Transformer | Effective discrete audio tokens, promptable | Codec dependencies, slow sequence decoding | Evaluate for music |
| Conditional spectrogram + vocoder | Modest research scale | Vocoder artifacts | **First original training experiment** |
| Latent diffusion or flow-matching DiT | General soundscapes, scalable | Larger training and iterative sampling | Primary longer-term sound research candidate |
| Raw waveform diffusion | No codec artifacts | Expensive high-rate waveforms | Experimental only |

MusicGen-small weights are CC-BY-NC-4.0; Stable Audio Open weights use Stability AI Community License. Neither is an original Sonumio model. Check distribution terms, model cards and licensing before use.

## Stage R0: licensed dataset manifest
Start with original/owned recordings and verifiable consent/permissions, or carefully audited CC0 recordings. Manifest fields: source URI, contributor, recording ID, caption and language, license and attribution requirements, training and redistribution permissions, consent, duration, sample rate, checksum, split, collection date, provenance verification. No personal voices in initial dataset.

Group splits by recording session/source rather than random chopped clips. Exclude duplicates and suspicious copyrighted music. Build a curated evaluation set (~100 caption-audio pairs). Never publish model weights without a full provenance review.

## Stage R1: original small-scale experiment
Train a 16-24 kHz, 1-4 second conditional spectrogram model on a small curated sound-effects dataset; test overfitting on 32 clips before scaling. Compare a deterministic Griffin-Lim decoder versus an appropriately licensed, documented vocoder. Log seed, config hash, dataset fingerprint, loss, time, VRAM, checkpoint hash, and nearest-neighbor/memorization audit.

## Stage R2: Sonumio-1 candidate
Evaluate a compact latent diffusion/flow-matching network against codec-token Transformer under matched data. Start with ~2-8 seconds at ~22-32 kHz mono. Only expand to stereo/long-form once objective and listening tests show improvement. Hardware/cost estimates must be based on measured training throughput, GPU memory and planned dataset size, not invented forecasts.

## Benchmark gates
- Signal integrity: correct duration, sample rate, no clipping, valid WAV on every successful request.
- Performance: cold and warm generation latency, peak CPU memory / GPU VRAM, real-time factor (elapsed generation time / audio duration).
- Quality: FAD on fixed reference distribution and frozen embedding setup, CLAP prompt correspondence, category accuracy and blind listening tests with multiple raters.
- Safety and ethics: consent coverage, dataset leakage, memorization, unauthorized voice likeness, copyrighted melody similarity.

## Not currently implemented
Original neural training or weights, realistic open-vocabulary sound effects, speech, multi-track editing, audio extension, streaming and audio-to-audio.

## Implemented research scaffolding
- `research/spectral_smoke.py`: optional PyTorch experiment using Sonumio-owned synthetic DSP examples. Measures train and held-out seed MSE and stores local NPZ weights (not published). **Does not generate audio.**
- `research/dataset_manifest.py`: JSONL validation of source metadata and rights declarations, split group integrity, duplicates and voice-data exclusion. Human rights verification is still mandatory.


## Longer-clip latent audio experiment — in draft, not Sonumio-1

The isolated [latent audio research experiment](LATENT_AUDIO_EXPERIMENT.md)
adds an original convolutional autoencoder and English/Hindi text-conditioned
latent DDPM for **1.024-second 8 kHz** synthetic clips. Unlike the earlier
short raw-waveform DDPM, the noisy diffusion tensor is a 16×128 latent instead
of an 8,192-frame waveform. Reconstruction uses waveform plus multiresolution
STFT losses. A matched/noisy-latent caption-retrieval benchmark includes
one-source-replacement hard negatives and a caption-blind chance control.

This is *not* an effective compression ratio for shipping audio files, not
long-form audio generation, and not evidence of subjective audio quality.
Independent listening/evaluation and diverse rights-reviewed datasets remain
unimplemented. No weights, tags or releases will be published without approval.


## Objective reconstruction protocol (draft research)

The [synthetic latent quality benchmark](LATENT_AUDIO_EXPERIMENT.md#objective-reconstruction-and-model-health-evaluation)
now uses the full held-out source-combination/seed grid (42 unique recordings)
rather than evaluating only the first bilingual pair. It reports aligned
waveform error, multi-resolution log-spectrum, spectral convergence and SNR;
silence, exact identity and moving-average controls; an exact-byte
train/evaluation leakage check; and group-level exploratory bootstrap
intervals. A separate generator health scan records finite/clipped/silent
waveform statistics **without labeling them subjective sound quality**.


## Same-data raw waveform vs latent DDPM benchmark

An isolated [controlled architecture study](ARCHITECTURE_COMPARISON.md)
trains both original English/Hindi-conditioned denoisers on the same
1.024-second synthetic clips, schedule, minibatch indices and update count.
It reports per-arm paired/wrong/blank caption errors against a zero-noise
predictor control, separate codec overhead, parameter counts and raw/latent
CPU timings. **It is not parameter/FLOP matched and does not allow
cross-domain numerical MSE ranking or claims about perceptual quality.**


## Replicated architecture diagnostics

The [multi-seed study](MULTI_SEED_STUDY.md) repeats the shared-data raw/
latent experiments with independent model-initialization seeds. It enforces
matching source-data fingerprints, selected captions, model configurations,
no weights and valid numerical diagnostics before reporting descriptive
mean/median/standard deviation/observed range and directional sign counts
**within each representation**. The same data sources and captions appear
in every run, so this measures initialization sensitivity **only**.


## First-party synthetic training expansion

A [prespecified extended-training study](EXTENDED_SYNTHETIC_TRAINING.md) now
trains raw and latent English/Hindi text-conditioned audio DDPMs with
three initialization seeds, 128 codec updates, 256 denoiser updates,
and all 21 sound-combination holdouts, without any neural weight export.
Its results must be inspected before drawing conclusions; synthetic
denoising improvements alone do not certify actual generation fidelity.


## Generated waveform health after neural training

The [numerical generation audit](GENERATED_AUDIO_AUDIT.md) samples actual
raw-waveform and latent-DDPM outputs after training and reports finite/range
validation, RMS, clipping, silence, spectral characteristics, and
same-noise-seed prompt-change sensitivity for English/Hindi rain/wind
prompts. This does not replace independent listening or rights-reviewed
reference audio. No waveform or model-weight artifacts are published.


## Empirical generation failures

The [in-memory generation pilot](GENERATED_AUDIO_PILOT_RESULTS.md)
reported substantial clipping in raw output and weak measured prompt
variation for the latent output. The research pipeline also identifies
a likely terminal DDPM prior mismatch with the eight-step linear beta
schedule. The next controlled study must address these issues without
conflating denoising error reduction with audible output quality.


## Schedule-prior experiment

A [paired linear-versus-cosine schedule study](SCHEDULE_ABLATION.md)
uses matched initialization seeds and generated-audio health metrics to
test whether the eight-step terminal-prior mismatch identified in
the generation pilot contributes to clipping. The default linear
schedule and previously recorded research results remain unchanged.
The experiment makes no release, listener-quality or model-selection claim.


## Negative cosine-schedule result

The [actual paired-seed schedule ablation results](SCHEDULE_ABLATION_RESULTS.md)
show the near-zero-terminal-SNR cosine profile substantially **increased
hard-clipping** in both raw and latent diffusion arms, despite fixing the
terminal-noise diagnostic. This falsifies a simplistic assumption that
reducing terminal SNR alone solves Sonumio's short-model generation
problems. The model remains research-only; v-prediction/posterior
sampling and pre-clamp saturation diagnostics are the next testable
engineering hypothesis.


## Coupled velocity-vs-epsilon objective research

The [velocity-prediction experiment](VELOCITY_OBJECTIVE_STUDY.md)
compares bit-identically initialized v and epsilon models on the same
cosine schedule, same examples/timesteps/noise and same optimizer budgets.
Its stable x0-based posterior avoids division by the extremely small
terminal alpha factor used by the failed epsilon sampler. It records
both pre-clamp out-of-range statistics and final clipping. Even if
these improve, rights-cleared data and actual listening evaluation
remain unfulfilled prerequisites.


## Verified velocity-objective experiment

The [measured three-seed v-versus-epsilon results](VELOCITY_OBJECTIVE_RESULTS.md)
confirm substantially reduced waveform saturation under the near-zero
terminal-SNR cosine schedule. The corrected codec audit additionally
exposes decoder **pre-Tanh** activations (rather than falsely treating
a Tanh-bounded waveform as an unclipped decoder output).
The experiment is synthetic-only; human listening, semantic prompting,
rights-audited real audio and extended training still remain.


## Synthetic-data provenance and cross-process reproducibility

The [dataset reproducibility gate](DATA_REPRODUCIBILITY.md) records
per-recording exact and bucketed hashes on all 42 training and 42 held-out
original DSP sounds. New research CI regenerates the dataset in separate
Python processes, requires exact agreement within the same runtime
configuration and reports thread-profile differences separately.
The private training CI also checks its actual model input fingerprints
against an independent preflight. Coarse bucketed hash agreement is not
a byte-identity claim or proof of real-world independent test coverage.


## Stage-level DSP CPU reproducibility

The [pipeline tracer](DSP_PIPELINE_TRACE.md) regenerates original
first-party DSP and validates exact 24-kHz generator output and the
final 8-kHz training crop, with hashes for every layer and processing
stage. Research CI requires two independent process traces to agree
exactly on the same host; the private training workflow checks all
84 final trace digests against independently generated training
input provenance. Per-stage and single-source SHA aggregates help
narrow host-dependent DSP float differences without altering audio.


## Original source-convolution numerical investigation

The latest [source-level trace findings](DSP_PIPELINE_TRACE_RESULTS.md)
reproduced exact hash drift in rain, thunder and wind on
AMD EPYC 7763, not in the other three sound classes.
The [convolution isolation protocol](DSP_CONVOLUTION_ISOLATION.md)
compares actual seeded inputs, kernels, convolution outputs and
scalar \`math.fsum\` references. This is a diagnostic of the
unchanged original DSP only, not proof of end-user audio quality.
