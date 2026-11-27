# Local postproduction pipeline

A single JSON plan can combine the newly implemented DSP operations
without writing or integer-quantizing intermediate audio. The operation
order is fixed: duration or pitch, optional loop search/render, optional
mastering. Loop source positions refer to the frame clock after the
first stage. Mastering acts on the assembled output, so dynamic limiting
can change the gain across repeats; do not interpret a repeated source
cycle as bit-identical after arbitrary dynamic postprocessing.

Example plan:

```json
{
  "duration_ratio": 1.25,
  "time_scale": {
    "method": "wsola",
    "window_frames": 1024,
    "search_frames": 128
  },
  "loop": {
    "start_frame": 2400,
    "stop_frame": 24000,
    "blend_frames": 256,
    "repeats": 2,
    "law": "cosine"
  },
  "mastering": {
    "target_rms_dbfs": -20,
    "max_boost_db": 6,
    "estimated_peak_ceiling_dbfs": -1
  }
}
```

```bash
sonumio wav-postproduce original.wav --plan plan.json --output finished.wav --output-bits 24
```

The plan parser rejects unknown or duplicate JSON properties, malformed
nested settings, and oversized files. A bounded preflight validates
intermediate and output geometry before processing. The original WAV
descriptor is verified before any DSP stage. Output publication occurs
once, atomically, after all stages and optional encoded-peak validation
have succeeded.

The manifest records the canonical plan hash, source PCM hash, exact
stage input/output frames and PCM hashes, and the final checksum-bound
receipt. Optional source file, source PCM, output PCM and encoded WAV
SHA256 pins allow deterministic verification. The receipt does not
authenticate the person who created it; a malicious party can rehash
modified data. It is not a license, consent or model-quality attestation.

When mastering is selected the actual integer output is remeasured and
may receive an additional quantization-aware attenuation. The manifest
binds pre-encoding float PCM; the file receipt separately binds the
encoded WAV and reports that attenuation. The system does not claim
certified LUFS, true-peak compliance or neural model inference.
