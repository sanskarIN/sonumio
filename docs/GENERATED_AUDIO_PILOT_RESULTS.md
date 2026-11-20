# Generated waveform pilot: trained raw and latent DDPMs

**16 November 2026 — measured research failure analysis, NOT a model release.**

Evidence: [Sonumio training and generation workflow 43850593239](https://github.com/sanskarIN/sonumio/actions/runs/43850593239),
code commit [f97d8790](https://github.com/sanskarIN/sonumio/commit/f97d87909384dfc0fb9052aecc5399be3f58bd03).
This run **completed successfully**. All four CI workflows passed and the
CPU research suite reported **138 passed**.

The models were **actually trained** on first-party Sonumio procedural
8-kHz mono audio, then sampled in memory for a fixed bilingual numerical
audit. **No WAV files, pretrained models or neural weights were saved or
published.** No Sonumio-1 model exists.

## Exact research configuration

| Item | Value |
| --- | --- |
| Initializations | 17, 29, 41 |
| Unique train recordings | 42 synthetic waveforms |
| Train captions | 84 English/Hindi pairs |
| Unique validation recordings | 42 synthetic waveforms |
| Evaluated source mixtures | All 21 (6 singles + 15 pairs) |
| Codec optimizer updates | 128 per seed |
| Raw denoiser optimizer updates | 256 per seed |
| Latent denoiser optimizer updates | 256 per seed |
| Noise schedule | 8-step linear beta |
| Sampled waveform duration | 1.024 seconds, 8-kHz mono |
| Generation-audit prompts | rain / बारिश; wind / हवा |
| Generation seed per initialization | 1517, 1529, 1541 |
| Shared data/protocol | raw-vs-latent-multi-seed-diagnostics-v3 |
| Configuration SHA256 | d9867be02a397682ac533ee54a7cfde86357b30d10c8eb835d5606bf4b41aedb |

The same random initial field is used for the four caption variants within
each model and sampling seed. Model arms have different underlying audio
spaces and network sizes, so a raw/latent numerical ranking would be invalid.

## Generated audio: actual numeric health

These numbers are averages across four prompts per seed, then across three
model seeds, measured on **newly generated neural samples** rather than
training/reconstructed audio.

| Numerical diagnostic | Raw waveform | Latent |
| --- | ---: | ---: |
| Mean RMS | 0.59054 | 0.07571 |
| Mean peak | 1.00000 | 0.39756 |
| Samples at clipping boundary (abs ≥ 0.999) | **13.71%** | 0.00% |
| Samples near digital silence (abs < 0.00001) | 0.002% | 0.015% |
| Mean spectral centroid | 1,869 Hz | 622 Hz |
| Mean absolute prompt-change waveform difference | 0.02938 | 0.00137 |

The raw branch shows a **major hard-clipping warning**. The latent branch
is not hard-clipped, but it changes only modestly under these specific
caption variations. Absolute prompt difference depends on waveform
amplitude, so this is a *numerical sensitivity diagnostic*, **not a
valid comparison of semantic language understanding** between branches.

For basic scale context, the two corresponding held-out first-party DSP
references in this run had rain RMS **0.0323**, wind RMS **0.0749**,
rain spectral centroid **1,752 Hz**, and wind spectral centroid **207 Hz**.
Even when a model's aggregate spectrum looks similar to a reference,
this does **not** establish that it generates the correct sound.

A clipping fraction of 13.71% can be objectionable in listening,
but an actual blinded listening evaluation has **not yet been performed**.
No user-facing audio generation quality is certified.

## Root-cause hypothesis: diffusion terminal prior mismatch

The earlier 8-step linear-beta schedule uses betas from 0.0001 to 0.02.
Its cumulative \`alpha_bar\` at the final training step is approximately
**0.922212**. Therefore its final noisy training state has approximate
clean-signal coefficient **0.960319** and injected Gaussian-noise
coefficient **0.278905**. **Sampling nevertheless starts from pure
unit Gaussian noise**.

In DDPM terminology, this is a substantial *train-time terminal noise /
sampling initial-prior mismatch*. It is a **credible quality-risk
hypothesis**, not proof that this mechanism causes all clipping or
weak text control. The accompanying code now exposes this numerically
as a diagnostic rather than silently claiming noise-prediction gains
guarantee a good sample.

**Next controlled research experiment:** predefine a schedule with
terminal SNR close to zero, keep initial training budget and sample
audit identical, compare stability across independent seeds, and
reassess generation clipping, RMS, spectral health and prompt sensitivity.
Do not mask failures with output clipping/limiting alone or assume a
schedule change fixes codec collapse and language grounding.

## Conclusion and remaining research gates

- The training job is **operational** and learns nontrivial noise prediction
  against a zero-noise predictor, but the new waveform audit shows visible
  numerical problems in output generation.
- Any future high-quality release needs **real, rights-verified source
  recordings**, disjoint recording-session testing, higher sample rates,
  multi-second quality, independent human listening, and separate attribution/
  memorization and safety auditing.
- This experiment alone cannot establish whether speech, music, or
  open-ended sound prompting works; those remain future research work.
- **No release, pre-release, model generation, weight upload, version tag
  or PR merge is authorized by these results.**


## Follow-up schedule experiment

The [paired linear-versus-cosine study](SCHEDULE_ABLATION.md)
was introduced *after* this original generation pilot, specifically
to investigate its terminal DDPM train/sampling-prior mismatch. The
numerical results above remain historical evidence for the **linear**
schedule and are not proof that cosine will improve generated audio.
