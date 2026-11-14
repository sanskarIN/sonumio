# Two-sound conditional diffusion — experimental

This is a research-only extension to the 8 kHz, 0.256-second waveform DDPM.
It is NOT a language model, general text-to-audio or Sonumio-1, and no
weights have been approved for publication.

## How it works

- Six categories: rain, thunder, wind, forest, ocean and ambient.
- Narrow English/Hindi keyword parsing converts a prompt to a binary [6]
  vector containing one or two supported sound classes, not arbitrary text.
- The existing 1D denoiser learns a normalized combination of the class
  embeddings plus a distinct null embedding for classifier-free guidance.
- Training uses all six single-category DSP sounds and all 15 possible
  two-category compositions. Four training seeds make 84 clips; two
  disjoint held-out seeds make 42 clips.
- Training reports conditioning_mode=multi-label; older checkpoints remain
  explicitly single-class. Numeric-only hash-verified checkpoints are used.

## Train and generate locally

~~~bash
python -m pip install -e '.[research,test]'
python -m research.wave_diffusion train --steps 50 --mixed-synthetic --output checkpoints/mixed-private
python -m research.wave_diffusion sample --checkpoint checkpoints/mixed-private --prompt "rain and thunder" --output outputs/mixed-en.wav
python -m research.wave_diffusion sample --checkpoint checkpoints/mixed-private --prompt "बारिश और गरज" --output outputs/mixed-hi.wav
~~~

These clips are far too short and undertrained for credible professional
sound effects. The experiment has no human listening, independent fidelity,
semantic alignment or memorization evaluation. Composition of learned class
embeddings does NOT establish robust multi-source generation.

Speech, voice-cloning and impersonation prompts are excluded. More than
two sound classes are rejected. Rights-reviewed local WAV ingestion remains
single-category and must not be silently relabeled as mixed pairs.

## Next experiments

- Genuine trainable text representations with provenance-reviewed captions.
- Longer audio via latent/codec representations rather than raw 8 kHz DDPM.
- Independent listening and prompt-alignment evaluations on held-out audio.
- Voice-safety and memorization audits before any publication or deployment.
