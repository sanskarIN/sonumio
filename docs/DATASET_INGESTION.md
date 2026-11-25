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

## Exact model-input crop leakage gate

In addition to whole-WAV and decoded-recording checksum gates, ingestion now
SHA256-fingerprints the **actual float32, 8 kHz, 2048-frame training crops**.
If one crop is exactly repeated across train/validation/test splits, loading
fails closed even when the full files and declared recording groups differ.
Such an overlap could otherwise contaminate held-out model evaluation.

Repeated crops **within** one split remain accepted for compatibility with
short recordings, but their count is surfaced as
`within_split_repeated_model_input_clips` instead of silently implying
they are new independent source groups. The report includes
`distinct_model_input_clip_hashes` and labels the exact gate explicitly.

This gate does **not** identify near-duplicates, pitch/time shifts, overlapping
but non-identical excerpts, or related recording sessions. It does not prove
source independence or declared recording rights.


## Narrow shift-aware overlap audit

After exact 8 kHz crop hashes and aligned gain/DC/polarity correlation,
a second fail-closed check looks for highly correlated *time-shifted*
copies across train, validation and test. It scans all integer offsets
within **1024 frames (128 ms at 8 kHz)**, retaining at least 50% of the
original 2048-frame crop for each verified comparison, but uses only high-variation
65-frame derivative-sign anchors to retrieve plausible candidates.
For each candidate, the decision uses centered absolute correlation
of the full real overlap (threshold 0.9995), never matching a zero-padded
tail. The index separates splits, limits both candidate comparisons and
per-key fanout, and **total posting-list visits per new clip**. Each
new clip is limited to 100,000 posting visits, regardless of how many
candidate lags survive the time-offset prefilter. The report includes
clip, anchor, candidate-comparison and posting-visit counts. Any work-budget
saturation aborts ingestion rather than silently passing.

This is not comprehensive near-duplicate search: low-variation tones,
offsets beyond 128 ms, time stretching, sample-rate edits, mixing,
complex processing and semantically similar recordings may evade the gate.
It cannot prove recording independence, legal rights or model quality.
Human review remains mandatory.


## Bounded local snapshot and corpus allocation

Manifest and WAV reads use a regular-file snapshot helper, rejecting leaf
symlinks, FIFOs and other special files before opening and enforcing a hard
byte cap during reads. Path and descriptor identity/metadata are compared
before and after reading, using platform-specific snapshot semantics.
Resolved WAV paths must remain inside the configured root. This is best-effort
local filesystem protection, not a sandbox against a hostile directory owner.
The approved source WAV SHA256 remains mandatory. Optional JSON metadata must
also be finite and UTF-8 encodable.

Before opening WAV assets, the loader plans all model-input crop counts and
rejects any corpus above the 256 MiB materialized float32 PCM budget.
Each split receives one preallocated array; leakage indexes reference the
stored rows instead of a second stack of the entire corpus. The limit does
not include Python metadata, leakage index objects, temporary source decoding,
or model/training allocations. Large-scale dataset sharding is a separate
future feature. The report records planned PCM bytes and the configured cap.
Existing crop geometry, waveform checks and provenance fingerprint are retained.
No real recordings, rights approvals, checkpoints or model quality claims
are produced by this change.
