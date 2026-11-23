# Experimental multi-clip Sonumio v-DDIM synthesis

This CPU-only research module extends the *duration of an in-memory audio
buffer* using the existing original Sonumio velocity-trained raw/latent
diffusion samplers. It does **not** extend the model's learned temporal
context beyond its independently generated 1.024-second clips.

## Algorithm and bounds

`research/longform_ddim.py` generates 2–8 separate 8192-frame (8-kHz)
mono clips with distinct deterministic seeds and either one shared
caption or a tuple of per-segment captions. It joins neighboring clips
using 128–4096 frames of overlap, in multiples of 64, with
**complementary raised-cosine** weights:

```text
w[k] = (1 - cos(pi * (k + 1)/(overlap_frames + 1))) / 2
crossfade[k] = (1 - w[k]) * prior_tail[k] + w[k] * next_head[k]
output_frames = 8192 + (segments - 1) * (8192 - overlap_frames)
```

Complementary (rather than equal-power) weights preserve identical
constant-amplitude signals without artificial 3-dB overlap gain.
The transition smooths abrupt wave splices but does NOT train a
continuation-conditioned neural model. Its sound category, speaker,
melody, and rhythm may still jump between clips.

The returned waveform is **unclipped and unnormalized**. Input clips
must be finite, detached CPU float32, exactly 8192 frames, and
numerically bounded. Invalid shapes, seeds, frame counts, captions,
timestep plans or NaNs are rejected. Explicit sampling seeds and model
parameter hashes prove reproducibility and guard against inferences
that mutate weights or global RNG state.

The numeric report includes output length, SHA256, amplitude/RMS and
fraction outside unit audio range; individual short-clip hashes and
seeds; and each seam's hard-concatenation sample jump, new crossfade
start/end jumps, and mean overlap discrepancy. For latent sampling,
pre-Tanh decoder activation saturation is explicitly labeled as
*internal neural activation*, not unclipped waveform audio.

These are sample-level diagnostics, NOT psychoacoustic click scores,
speech quality or semantic prompt adherence.

## Research-only usage

```python
from research.longform_ddim import sample_overlapped_velocity_ddim

# model must be an original velocity-trained denoiser, not epsilon weights.
waveform, audit = sample_overlapped_velocity_ddim(
    model, schedule, ("rain", "बारिश"),
    segments=2, overlap_frames=1024,
    evaluations=8, seed=101,
)
```

At 8192 frames per clip, two segments with 1024-frame overlap produce
15360 samples = 1.92 seconds at 8 kHz. A longer waveform buffer
is **not** evidence that the model understands long audio sequences.
The current neural model is still preliminary; a randomly initialized
denoiser produces meaningless noise. Do not expose this as Sonumio
speech, music or production-quality text-to-sound generation.

```bash
python -m pytest -q tests/test_longform_ddim.py
```

The tests include constant-amplitude preservation, join-boundary
diagnostics, signed/NaN/oversized rejection, shape bounds, exact seed
reproducibility, English/Hindi UTF-8 captions,
unclipped raw waveform provenance, pre-Tanh latent saturation,
global RNG/model-mode/model-parameter preservation, and absence of
filesystem or network output.

No external training recordings, weights, WAV files, tags, public model
identity or released assets are created. Before any actual coherent
long-form model claim, original continuation-conditioned training,
new temporally held-out real-rights-reviewed audio, blind listening
panels, and semantic/musical alignment validation are necessary.
