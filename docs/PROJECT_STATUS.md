# Sonumio development status and research gates

**Date:** 16 November 2026. **Branch:** `feat/initial-audio-platform`
(PR #1, **DRAFT, not merged**). Public `main` contains the Apache-2.0
license and the project README, while the much larger prototype remains in
development. **No stable release, pre-release, tag or model weights have been
authorized or published.** Research results are not production quality.

## Delivered source and infrastructure (draft PR unless noted)

| Area | Implemented and testable | Not yet complete |
| --- | --- | --- |
| Public identity | Permanent name Sonumio, tagline “Imagine it. Hear it.”, Apache-2.0; concise README and license on `main` | A polished public documentation/deployment site and governance |
| Local generation | Procedural DSP baseline: rain, thunder, wind, forest, ocean, ambient; short offline WAV generation | Realistic open-domain neural sound generation |
| Product scaffolding | Python CLI, FastAPI service, React/TypeScript studio prototype | Product-level UX, accessibility, usability study and distribution |
| Async workflow | In-process queued jobs, task state and cancellation tests | Persistent queue, quotas, recovery, lifecycle and storage hardening |
| Original neural experiments | Spectral regression, class-conditional short waveform DDPM, English/Hindi byte-conditioned waveform DDPM, two-source conditioning, trainable codec + 1.024 s latent DDPM | Independently proven realistic quality, high-rate/long-form architecture, original Sonumio-1 |
| Data handling | Rights-declared local WAV manifests, bounded snapshot parsing, SHA256 file verification, session-group split validation and no declared voice data | Independent source/license review, consent auditing, verified diverse long-term dataset and non-speech detection |
| Checkpoints | Private numeric NPZ+JSON, SHA checks, archive shape/type/size preflight | Audited persistence, retention, distribution controls and model cards |
| Evaluation | Synthetic seed-isolated data, caption hard negatives, codec identity/silence/DSP controls, waveform/spectral metrics, cluster bootstrap, research CI | Real-world held-out tests, FAD/CLAP, blind listening, listening demographics, cross-source generalization and memorization audit |
| Architecture study | Same 8-kHz 1.024-second recordings for raw-vs-latent text-conditioned DDPM diagnostic, per-arm controls and parameter/timing reports | Fair matched-FLOP/parameter trials, meaningful convergence runs, multi-seed comparisons, independent sampling preference tests |
| Tests | Python core, web UI, CPU research GitHub Actions | Large-scale integration, GPU training, packaging, security review, release qualification |

The architecture comparison is **controlled on data, clip length,
minibatch indices, schedule and step count**, but it is **not
compute- or capacity-matched**. Its raw and latent noise MSE values are not
directly comparable quality statistics. No winner is claimed.

## Remaining engineering gates (prioritized)

1. **Real corpus with evidence.** Build a small, independently human-reviewed
   non-voice sound-effects dataset, with verified ownership, permitted training
   use, source records, attribution, deduplication and session-level holdouts.
   Current declarations alone cannot prove legal rights.
2. **Realistic model quality.** Establish 16–32 kHz and multi-second audio
   reconstruction and neural generation benchmarks. Run multi-seed training
   with known resource usage; improve the codec/latent architecture based on
   measured failures, not just reduced loss.
3. **Scientific comparison.** Compare raw, spectral and latent approaches under
   honest matched compute/parameter budgets, separate codec overhead, and
   independently record quality, latency and VRAM.
4. **Listening and safety studies.** Run blinded perceptual evaluation; probe
   long-tail prompts, English/Hindi natural wording, copyright/memorization
   and safety/privacy failure modes. Restrict voice data until consent policy
   and safety gates are met.
5. **Production backend and storage.** Harden queues, request quotas,
   cancellability, output retention, failure recovery, per-user isolation,
   logging privacy and offline model provenance.
6. **UI/UX and creator workflow.** Polish studio accessibility, progress,
   editing, audio previews, error handling and install experience.
   Music/TTS/long-form SFX remain research goals, not ready features.
7. **Reproducible packaging.** Introduce security audit, reproducible
   dependency locks and isolated builds, versioned API/docs, license/NOTICE
   review, compatibility testing and external contributor onboarding.
8. **Explicit release authorization.** The project owner must separately
   approve a release, tag, published weights or draft PR merge when justified.
   Passing CI alone is not authorization.

## Current acceptance rule

Keep PR #1 as a draft and develop/tests there. Merge only after explicit
approval and release-readiness gates; never publish model weights, tags,
pre-releases or stable artifacts by implication. Synthetic procedural audio is
a development test substrate, not proof of Sonumio's final capability.


## Multi-initialization follow-up (development branch)

A [replicable multi-seed experiment](MULTI_SEED_STUDY.md) now runs both raw
and latent audio DDPM benchmarks for multiple prespecified initialization
seeds, preserving per-seed outputs and checking schema, dataset fingerprints,
numeric metric consistency and absence of published weights. It reports
observed across-seed spread **within each model**, never a cross-domain
MSE-based winner. The remaining major research dependencies still include
independent rights-cleared audio, long training, generation/listening
assessment, compute-matched comparisons and a human safety review.


The replicated research study now also checks that **wrong-caption controls
do not accidentally name a true source category**. Its updated protocol
fingerprints those category-disjoint bilingual hard negatives before combining
results across model-initialization seeds. It remains synthetic-only.

The [first executed three-seed pilot result](MULTI_SEED_PILOT_RESULTS.md)
records the actual reproducible CI configuration, observed within-model
zero-noise prediction improvements, and small inconsistent bilingual
caption-alignment effects. This is still a synthetic pilot, not a model or
release readiness result.


## Extended synthetic training (research-only)

A [dedicated CPU workflow](EXTENDED_SYNTHETIC_TRAINING.md) runs three model
initializations with 128 codec updates and 256 denoiser updates per
architecture, using the full 21-combination synthetic evaluation grid.
It writes only an ephemeral private metrics JSON on its runner, with no
weight artifacts or publication. This is substantially longer training than
the first experimental pilot, but still **not** a release or quality gate.


### Extended training executed and verified

A [successful three-seed CPU experiment](EXTENDED_SYNTHETIC_RESULTS.md)
completed **128 codec and 256 denoiser updates per model per seed** with all
21 synthetic sound combinations evaluated. Every model exceeded its own
zero-predictor control, but cross-domain results cannot select a winner.
No real-world recordings, subjective listening tests or weight files were
included. Four GitHub workflows passed on the experiment commit.


A numerical [in-memory generative audio audit](EXTENDED_SYNTHETIC_TRAINING.md#in-memory-neural-generation-health-audit)
now runs after model training, before weights are discarded. It checks
actually sampled 8-kHz one-second waveform duration, finite samples,
clipping, silence, simple spectrum and prompt-change sensitivity. These
do not replace real listening or independent audio-class relevance tests.


### Generated audio findings and prioritized corrective research

The [actual generated-waveform pilot](GENERATED_AUDIO_PILOT_RESULTS.md) ran
after full synthetic training. Raw output clipped around 13.7% of audio
samples, while latent output had small same-noise-seed prompt-change
differences. A new numerical diagnostic identifies the eight-step
linear-beta terminal-prior mismatch (alpha_bar ≈ 0.922), which must
be studied under controlled alternative diffusion schedules before
making quality claims. Current audio generation is **not production-ready**.


### Controlled next experiment — two diffusion schedules

The [diffusion-prior schedule ablation](SCHEDULE_ABLATION.md) now
trains and samples both the original eight-step linear-beta DDPM and a
cosine near-zero-terminal-SNR variant under the same three model seeds,
train/evaluation recordings, optimizer updates, model widths and
audio-generation numerical controls. Prior metrics cannot be interpreted
as proof the alternative improves audible quality. This study is explicitly
for diagnosing raw output clipping and weak latent prompt variation.


### Experiment outcome — cosine does not improve audio health

The [measured schedule-ablation report](SCHEDULE_ABLATION_RESULTS.md)
shows original linear-schedule generated audio clipping at **13.71% raw /
0% latent**, versus **94.62% raw / 63.86% latent** under the eight-step
cosine schedule after matched training. All three matched seeds had
higher clipping for cosine. The original linear schedule remains default
only for research compatibility, not because it is production quality.
A stable velocity-prediction objective and pre-clamp diagnostics must be
evaluated before any schedule promotion.
