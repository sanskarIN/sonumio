# Measured result: matched linear vs cosine schedule ablation

**16 November 2026 · research-only · negative experiment**

**Conclusion:** With exactly the tested models, synthetic first-party audio
and short training budget, the cosine schedule dramatically **worsened**
generated-waveform clipping. This negative result is preserved rather than
promoting the alternative merely because it has lower terminal SNR.
**Neither schedule produces a release-ready Sonumio model.**

## Verifiable evidence

- Successfully completed GitHub Actions job:
  [Private synthetic audio training (CPU) #43856670235](https://github.com/sanskarIN/sonumio/actions/runs/43856670235)
- Experiment code commit:
  [5f7b05f7](https://github.com/sanskarIN/sonumio/commit/5f7b05f7d5d71521a4daf62de8847fa46e5549aa)
- [Experimental protocol and code](SCHEDULE_ABLATION.md)
- Other checks on that commit: GitHub **Python, web, CPU research** all passed;
  **155 CPU research tests passed**.
- Report configuration SHA256:
  \`cb1bd14d32680174142edfb70dae38e8ce6d3deee6e7f90e899b894a78d525b8\`
- **No audio files or neural weights were written or uploaded**.

The two schedules were run with the **same synthetic audio corpus**,
identical initialization seeds **17, 29, 41**, 42 unique synthetic train
recordings and 84 bilingual training captions, 42 unique synthetic
evaluation recordings, 21 evaluation source combinations,
128 codec optimizer updates, 256 raw-DDPM optimizer updates, 256
latent-DDPM optimizer updates, eight diffusion timesteps and
1.024-second **8-kHz mono** generated-audio audits per model.
The only intended intervention was the diffusion schedule.

## Terminal noise hypothesis

| Schedule profile | Final alpha bar | Approximate retained clean signal power |
| --- | ---: | ---: |
| \`linear_beta_v1\` | 0.92221218 | 92.22% |
| \`cosine_zero_terminal_v1\` | 0.00003747 | 0.00375% |

The cosine profile fixes the *specific numerical terminal-noise mismatch*
diagnostic: the training terminal distribution is much closer to the
pure-Gaussian prior used to initialize sampling. **It did not fix
sample quality** in the tested short-epsilon-DDPM setup.

The cosine profile uses a terminal beta capped at **0.999**, corresponding
to a final alpha of ~0.001. Its epsilon-parameterized reverse sampler
divides by square-root alpha. This strongly amplifies estimation errors
at that step, a credible explanation for saturation, **not a proven
causal attribution**. Training convergence, codec scale and noise
parameterization also require investigation.

## Generated audio — observed outcomes

All values are averaged across four fixed rain/wind English/Hindi
prompts per seed, then over three matched model-initialization seeds.
The same starting noise is used for compared prompts within each arm.
These are *numerical measurements of freshly generated waveforms*
clamped to [-1,1], not subjective hearing or independent reference scores.

| Metric | Raw linear | Raw cosine | Latent linear | Latent cosine |
| --- | ---: | ---: | ---: | ---: |
| Clipped samples (abs ≥ 0.999) | **13.71%** | **94.62%** | **0.00%** | **63.86%** |
| RMS | 0.5905 | 0.9818 | 0.0757 | 0.9482 |
| Rain-vs-wind mean absolute waveform change | 0.02297 | 0.28582 | 0.00122 | 0.05915 |

For clipping, the paired **cosine minus linear** result was:

| Model | Increase in clipping (percentage points) | Seeds worse |
| --- | ---: | --- |
| Raw waveform DDPM | +80.91 | 3/3 |
| Latent DDPM | +63.86 | 3/3 |

The candidate raised clipping on every matched seed in **both**
model architectures. The raw model was already poor; the candidate
is clearly worse by this numerical measure. Latent clipping changed
from zero to a majority of generated samples at the hard amplitude
limit.

Prompt-dependent waveform differences increased, but these are
not evidence of better sound category control or bilingual semantics.
When most samples are saturated, amplitude differences and correlations
become especially hard to interpret. Cross-architecture waveform
differences also depend on output amplitude and therefore do **not**
form a direct raw-vs-latent quality ranking.

## Scientific decision

1. **Do not promote** the cosine-plus-epsilon profile to Sonumio's default.
   The default remains \`linear_beta_v1\` for compatibility and historical
   reproducibility. That default is also **not** release-quality audio.
2. The cosine profile remains research-only as a *negative control*
   in this specific experiment, not a proven superior training schedule.
3. Investigate **velocity (\`v\`) prediction** and the corresponding
   stable DDPM posterior reconstruction, which can avoid the
   high-noise epsilon-prediction division. This requires a separately
   trained objective, not changing the sampler for an epsilon-trained
   checkpoint.
4. Add **pre-clamp output-range diagnostics**, instead of hiding severe
   saturation by taking only clipped waveform measurements.
5. Repeat paired-seed generated-waveform audits after the controlled
   change; evaluate exact model/dataset provenance, clipping, RMS, broad
   spectral diagnostics and prompt sensitivity before listener studies.
6. Real-world rights-reviewed recordings, longer/high-rate audio,
   independent sound recognition/listening and full licensing/privacy
   audits remain mandatory before any original model release.

Passing CI here means the **experiment ran correctly**, not that it
produced acceptable generative audio. A failure of the tested
hypothesis is a legitimate scientific outcome.

**No release, pre-release, version tag, model-weight upload, public WAV,
or merge of draft PR #1 was authorized or performed.**


## Next controlled result: objective matters, not just schedule

The [independent v-versus-epsilon paired objective experiment](VELOCITY_OBJECTIVE_RESULTS.md)
on the **same cosine schedule** showed that training v-prediction
with the matching x0-based posterior substantially reduced clipping
compared with epsilon prediction. Thus the earlier negative cosine
finding must be read as *cosine plus epsilon under this training
budget*, not as proof that any cosine schedule is universally inferior.
The trained networks, timestep/noise draws, and codec were tightly
matched within each model arm. Do not promote a production model
without independent sound quality, prompt relevance and real-audio
evaluations.
