# Crop codec fixed-width rate-distortion research

The existing one-second neural codec uses 8192 samples. The reviewed local
archive uses 2048 samples. This independent experimental codec accepts both
lengths at 8 kHz, with a common stride-64, 16-channel latent geometry.

For 2048 samples it emits 512 floating-point latent values (4x scalar
compression). A tanh bounds the latent in every precision arm. Uniform
symmetric quantization uses signed integers from -qmax to qmax, where
qmax = 2^(bits-1)-1. Training uses a straight-through estimator; evaluation
uses hard quantization and decodes actual fixed-width packets.

## Numeric packet format

The little-endian header contains eight magic bytes SLCROP01, uint16 source
frames, uint8 channels, uint8 precision bits and uint32 code count.
The unsigned words are signed codes plus qmax, packed least-significant bit
first in C order. The all-ones word is reserved. SHA256 of the complete
header and payload follows the payload. Shapes and byte counts are checked
before unpacking. The format is bounded to 2048 or 8192 source frames and
2, 4, 6 or 8 bits per latent scalar. No model is embedded, and the checksum
is not authentication or encryption.

At 2048 frames and 4 bits, nominal payload is 256 bytes plus 48 bytes of
header and checksum. This is an actual fixed-width packet, not an entropy
coded or production audio bitstream. A decoder must use the exact same
privately trained codec state. The format does not prove model provenance.

## Paired experiment

```bash
python -m research.crop_codec_study --model-seeds 17,29,41 \
  --training-seeds 0,1,2 --validation-seeds 101,102 \
  --precision-bits 4,8 --updates 32 --batch-size 4
```

All precision arms start from identical original weights and receive the
same sequence of synthetic DSP crops and optimizer settings. Only precision
differs. The loss combines waveform L1, preemphasis, log spectral magnitude,
spectral convergence and RMS envelope errors. Evaluation on seed-disjoint
synthetic crops exposes per-example waveform and spectral reconstruction
errors, SNR and pre-tanh decoder saturation. Packet decode must equal hard
quantization. The report includes actual packet size and nominal payload
rate. Independent model seed summaries remain descriptive, not significance
tests. No external audio, checkpoints, generated WAVs or trained model
weights are saved. This establishes neither perceptual fidelity nor
Sonumio-1 readiness. Private reviewed archive training integration is a
separate step.


### Source recipe correction

The crop study uses the versioned seeded-window source from the longer
first-party latent research waveform. The older fixed-start source may
produce identical audio for different seeds, including deterministic
ambient synthesis. Seed disjointness alone is insufficient. Exact PCM
overlap checks remain mandatory; different crop offsets do not establish
independent recordings or real-world generalization.
