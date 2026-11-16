# Sonumio rights-gated local WAV ingestion

This experimental pipeline augments the original six-category short-clip DDPM.
It does NOT constitute open-vocabulary sound generation, Sonumio-1, or approval
to distribute training data or model weights.

## Requirements

- Provide a manually rights-reviewed manifest with train and validation recording
  groups that do not overlap; the declaration validator rejects duplicate asset
  checksums and records marked as containing a voice.
- Each row needs a local_path relative to an explicit audio-root directory,
  plus the existing manifest fields: asset_id, source_url, sha256, caption,
  language, group_id, split, license, source_kind, rights_reviewed,
  creator_permission, voice_present, sample_rate and duration_seconds.
- sha256 is the lowercase SHA256 of the WHOLE local WAV FILE, not decoded PCM.
- WAV must be uncompressed 16-bit PCM mono or stereo, at most 20 MiB per asset,
  with sample rate and duration matching the declaration. Stereo is averaged
  to mono; scipy's polyphase filter resamples to 8 kHz.
- Paths escaping audio-root through traversal, absolute paths or external
  symlinks are rejected. Network fetches and executable codecs are not used.
- Captions must match exactly one supported category (rain, thunder, wind,
  forest, ocean, ambient), in a small English/Hindi keyword vocabulary.
  Multi-category and unrecognized captions fail explicitly.
- At least one train and one validation recording must exist. The test split
  may be supplied but is never used to choose model weights.

## Minimal JSONL example

Replace this illustrative placeholder with an ACTUAL recorded file and hash.
The URL below is not a usable data source.

~~~json
{"asset_id":"rain-001","source_url":"https://example.org/owned-rain","local_path":"rain/001.wav","sha256":"<SHA256 OF WAV FILE BYTES>","caption":"rain","language":"en","group_id":"session-a","split":"train","license":"CC0-1.0","source_kind":"cc0","rights_reviewed":true,"creator_permission":false,"voice_present":false,"sample_rate":24000,"duration_seconds":1.0}
~~~

Provide a separate validation asset with a different group_id and correct
checksum. This example is NOT itself a valid dataset.

## Use locally

~~~bash
python -m pip install -e '.[research,test]'
python -m research.dataset_manifest /path/to/manifest.jsonl
python -m research.wave_diffusion train --steps 50 --seed 9 --manifest /path/to/manifest.jsonl --audio-root /path/to/rights-reviewed-wavs --clips-per-asset 2 --output checkpoints/private-research
python -m pytest -q
~~~

No remote recordings are downloaded. Checkpoints must remain private.

## Important limitations

Manifest fields are DECLARATIONS; code cannot independently establish
ownership, confirm CC0 licenses, obtain consent, or detect voices in recordings.
Human source-level review is mandatory. Denoising MSE is not perceptual
quality, and the present 8 kHz 0.256-second fixed-category DDPM is still a
small exploratory model, not ready for useful real-world sound generation.

## Snapshot integrity for reproducible experiments

The manifest is read **once** into a bounded, validated in-memory snapshot
(maximum 32 MiB, 10,000 records, and 16 KiB per JSONL line). The validator
rejects duplicate JSON keys, non-finite durations and malformed typed fields;
the report fingerprint hashes the **exact manifest file bytes** consumed.

Per-split fingerprints incorporate the exact caption wording, source metadata,
verified WAV checksum and crop settings. Each loaded WAV clip retains its
caption from the same in-memory snapshot; the bilingual text-training path
never reopens the manifest after decoding audio. Editing a caption without
changing any WAV bytes therefore changes the research split fingerprint.

This prevents accidental silent caption changes between validation and text
training. **It does not verify whether declarations are truthful**, nor does
it make future filesystem modifications or unreviewed recordings safe.
