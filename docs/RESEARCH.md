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
