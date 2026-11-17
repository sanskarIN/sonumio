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


### New controlled objective test, without release promotion

The [coupled v-prediction experiment](VELOCITY_OBJECTIVE_STUDY.md)
tests a mathematically paired velocity target with a corresponding stable
reverse posterior, while keeping the severely clipped cosine+epsilon
configuration as the negative baseline. It additionally measures
**pre-clamp** waveform peaks and out-of-range fractions so clipping is
not hidden. Results remain synthetic-only, research-only and private.


### Experimental velocity prediction reduced severe saturation

The [verified paired objective report](VELOCITY_OBJECTIVE_RESULTS.md)
shows numerical saturation improvements for 8-kHz 1.024-second
synthetic samples on all three initializations: raw cosine+epsilon
hard-clipping **94.64%**, raw cosine+velocity **0.004%**;
latent decoder internal-Tanh saturation **71.13%** versus **0%**.
The measured velocity result is promising for numerical stability
but does not establish actual sound fidelity or prompt meaning.
A mismatch in exact synthetic data fingerprints across separate
CI executions is recorded as an unresolved reproducibility issue.


### Training dataset exactness and provenance

A [source-level reproducibility audit](DATA_REPRODUCIBILITY.md) now guards
training integrity. It enumerates all 84 distinct source/seed records,
validates bilingual pairing and split uniqueness, logs exact little-endian
float32 recording hashes separately from coarse numerical bucket hashes,
and checks independently regenerated datasets across Python processes.
The training workflow refuses a mismatch between the separately
regenerated dataset and the corpus actually seen by the paired trainer.
This is a research integrity gate, not a sound-quality improvement.


### Recording-level reproducibility evidence

The [verified two-runner research comparison](DATA_REPRODUCIBILITY_RESULTS.md)
showed that deterministic source code and seeds are insufficient to
promise identical DSP float32 bytes on different GitHub-hosted CPU
models. Independent processes agreed exactly **within** one runner,
and the private training process agreed exactly with its independent
preflight, but cross-host overall hashes differed. Neither source
recordings nor trained weights were uploaded; model-quality research
must continue using strict within-run data pairing.


### Original DSP stage provenance

The [numerical stage-level trace](DSP_PIPELINE_TRACE.md) checks all
84 original first-party recordings and records exact hashes for each
individual sound component, layer mix, edge fade, normalization,
49-tap resampling kernel, filtered float64 samples, 8-kHz downsampling
and final 8192-sample crop. It enforces exact agreement with existing
Sonumio DSP source output, not a newly substituted recording recipe.
The research and training CI jobs report per-stage signatures
from their separate CPU hosts.


### DSP pipeline cross-CPU observations

The [four-runner stage-tracing study](DSP_PIPELINE_TRACE_RESULTS.md)
confirmed that all 20 original DSP stage aggregates and all
12 single-source layer/crop hashes match exactly across AMD EPYC
9V45 and Intel Xeon 6975P-C hosts. The historical AMD EPYC 7763
mismatch remains unresolved, so a controlled older-host experiment
is warranted before changing the original audio recipe.


### EPYC 7763 stage-level mismatch reproduced

The [four-host investigation](DSP_PIPELINE_TRACE_RESULTS.md)
reproduced different original audio bytes specifically in
rain/thunder/wind source layers on an EPYC 7763 machine,
while a same-image Intel CPU and Ubuntu 24.04 AMD machines
matched each other. A [targeted source-convolution diagnostic](DSP_CONVOLUTION_ISOLATION.md)
now tests RNG inputs, float32 kernel construction and
NumPy convolution outputs for 60/140/180 versus 12 taps,
without modifying the training corpus or neural models.


### Convolution arithmetic is the first observed host-specific divergence

The [cross-host convolution report](DSP_CONVOLUTION_RESULTS.md)
isolated the earlier EPYC 7763 source-fingerprint difference:
original rain, thunder and wind input arrays and kernels had
identical exact hashes across tested hosts, but NumPy
60/140/180-tap **convolution output** SHA256s differed.
Water's 12-tap control and scalar \`math.fsum\` references matched.
This is a numerical-reproducibility finding, not proof of audible
differences or an exact low-level CPU instruction. Do not rewrite
the original corpus; if bitwise cross-host data is required,
first design and independently evaluate an explicitly versioned
reproducible recipe.


### 17 November 2026 — separate scalar-convolution candidate

Added a [non-invasive scalar convolution candidate](SCALAR_CONVOLUTION_CANDIDATE.md)
with strict input validation, original DSP input/kernel SHA256 gates,
selected-output sample error metrics, report consistency tests, and an
independent four-runner SHA comparison in research CI. The source
renderer and all previously measured training data remain exactly
unchanged. Until independent host results and complete waveform
evaluations are verified, cross-CPU reproducibility and audio quality
are **not** claimed. PR #1 remains draft, unmerged and unreleased.


### 17 November 2026 — verified full source-convolution CPU experiment

The [measured full-frame matrix](FULL_CONVOLUTION_RESULTS.md)
now reports identical scalar-candidate source-convolution bytes
across four hosted CPU jobs (2 × EPYC 7763, Xeon 6975P-C and
EPYC 9V45), across **16 source/seed recordings** and all
36,000+ output positions per case. Original NumPy rain/thunder/wind
hashes differed on EPYC 7763. All eight fixed golden-byte tests
passed on each runner and the max candidate/NumPy absolute
difference over the complete output was <= 1.20e-7.
**This does not establish full DSP pipeline/crop equality or
model-training reproducibility on all CPU environments.**
The historical source generator and synthetic training data
remain unchanged, and this PR remains development-only.


### 17 November 2026 — isolated DSP-to-8-kHz-crop candidate pilot

A [separate numerical candidate](COMPLETE_DSP_CANDIDATE.md) carries
the previously tested scalar source-convolution primitive through
24 kHz first-party sound mixing/normalization, the **unchanged**
FIR resampler and a seed-specific 8192-frame 8-kHz crop. The
historical renderer is checked byte-for-byte against the neural
research audio-data oracle; the candidate is not passed to training.
A 40-recording train/held-out pilot on **four CPU runners** produced
identical candidate crop hashes across tested hosts even where the
original NumPy data differed. The pilot passed 4/4 matrix jobs;
the research-suite implementation head also had 275 passing tests.
Source-level anti-leakage checks now reject both duplicate candidate
and original waveform hashes within a split and exact split overlap.
The full **84-recording** corpus remains a separate CI gate, and
this is not proof of model quality or release readiness.


### 17 November 2026 — isolated DSP throughput measurement added

The [in-memory paired benchmark](DSP_CANDIDATE_BENCHMARK.md)
now compares original 24-kHz DSP rendering with the separately
specified scalar source-convolution candidate. Same source/seed,
bounded 1–8 repetitions, warm-ups and alternating test order
produce local median/p95 timing, real-time factor, exact within-host
SHA fingerprints and maximum audio-sample difference. It records
no WAV or model artifacts and does **not** assert a speed target.
Prior-head Python 3.12 standard CI had an **installation BrokenPipeError**,
not a test assertion failure; a 3.13 job passed and the matrix job
was marked failed. Final-head CI must be checked separately.


### 17 November 2026 — all 84 first-party candidate recordings verified

The [full-corpus multi-CPU results](COMPLETE_DSP_RESULTS.md) now show
**4/4 successful independent CPU jobs**, with two EPYC 7763,
one EPYC 9V45 and one Intel Xeon 6975P-C runner. All **84** synthetic
source/seed recordings (42 train, 42 validation) yielded exactly
matching candidate *complete 24-kHz* and final 8-kHz crop SHA256
aggregates across the tested hosts, where original NumPy results
differed. No external audio rights, perceptual quality, neural
generalization or production throughput claim follows from this.
The new `research/candidate_reference.py` gates future full-corpus
test reports against the observed candidate SHA256s; tests and
benchmark measurements on later commits need separate CI validation.


### 17 November 2026 — performance bottleneck and new fast hypothesis

[CPU research run #44064895590](https://github.com/sanskarIN/sonumio/actions/runs/44064895590)
passed **281 research tests** including the new paired
`benchmarks/dsp_candidate.py` study. On one EPYC 7763 runner,
a 1.5-second *rain* generator run had median original DSP
0.001956907s versus median scalar reference 1.328028s
(~678.64× slower, two repetitions, no warmup). This is
a diagnostic, not a general speed guarantee.

Added the separately [specified double-precision vectorized
hypothesis](VECTORIZED_CONVOLUTION_CANDIDATE.md), which casts
float32 source inputs/kernels to float64 before NumPy convolution
and rounds final output once to float32. It tests complete hashes
against the four-host scalar fsum reference, with numerical
regressions and a CPU matrix. It is **NOT** promoted into
the original DSP, neural training or dataset pipeline.
Cross-host results and quality/performance implications are
unverified until the new workflow completes.


### 17 November 2026 — fast complete-DSP hypothesis now on separate CI path

Created [`research/fast_dsp_candidate.py`](../research/fast_dsp_candidate.py)
and its [experiment protocol](FAST_DSP_CANDIDATE.md). This new
research-only source recipe uses NumPy vectorized float64
source convolution and otherwise preserves original sound
randomness, mix/fades, peak normalization, FIR resampling
and seed-disjoint 8192-frame training crop. It validates the
original waveform against the unchanged neural research
oracle, checks 24-kHz and 8-kHz numerical error, and rejects
duplicated exact waveforms and train/validation SHA overlap.
The CPU matrix compares full-corpus fingerprints from two
separate processes per host and reports whether they equal
the independently verified scalar reference. No existing
generator, model trainer, weights or dataset fingerprints
were modified. Fast candidate exact reproducibility is
**not yet confirmed** until the new CI finishes.
