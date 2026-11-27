# Deterministic original sound packs

The sound-pack CLI turns a small local JSON plan into a single canonical
ZIP file containing original procedural mono WAV assets and an exact
machine-readable manifest. It uses the existing 24 kHz nature and
procedural sound generators. It does not use third-party samples, a
trained model, network access or voice imitation.

Example plan:

```json
{
  "pack_id": "forest-effects",
  "seed": 42,
  "output_bits": 24,
  "recipes": [
    {
      "asset_id": "rain",
      "frames": 48000,
      "variants": 4,
      "nature_kind": "rain",
      "tags": ["ambient", "rain"]
    },
    {
      "asset_id": "wind",
      "frames": 36000,
      "variants": 3,
      "nature_kind": "wind",
      "motion": 0.8,
      "tags": ["ambient", "wind"]
    }
  ]
}
```

Recipe identifiers and tags use portable lowercase ASCII. Recipes and tags
are sorted and unique. Variants are numbered starting from one. Each
seed derives from the master seed, asset identifier, variant and a domain
label. Adding a different recipe does not change the seed of an existing
asset. Dither is explicitly optional and has a separate seed domain.

For the original prompt baseline use source "procedural", nature_kind
null and a supported prompt such as "rain and wind". Nature controls are
not applied to that source. The historical prompt restrictions remain in
force. Each recipe has an exact integer frame count. All assets use the
selected common integer PCM depth (8, 16, 24 or 32 bits).

```bash
python -m sonumio.audio_core.asset_pack_cli sound-pack-plan --plan library.json
python -m sonumio.audio_core.asset_pack_cli sound-pack --plan library.json --output library.zip
python -m sonumio.audio_core.asset_pack_cli sound-pack-verify library.zip
python -m sonumio.audio_core.asset_pack_cli sound-pack-verify library.zip --replay
```

The preview validates the plan and reports its jobs without synthesis.
Generation applies the declared floating sample ceiling and derives exact
integer-code diagnostics from the encoded audio. The manifest contains
per-file and PCM-payload SHA256, sample geometry, integer sums and squared
sums, rail counts and duplicate WAV groups. The default quality policy
rejects rail occupancy and RMS below 0.0001; duplicate audio is reported
but allowed because some procedural sounds do not vary with the seed.

All archive members are stored without compression, use fixed metadata
and appear in a defined order. Verification does not extract files and
rejects extra members, alternate ZIP representations, incorrect geometry,
bad padding and inconsistent checksums or integer diagnostics. The archive
and expected output digest are validated before staging and publication.
Existing output requires explicit overwrite. Input and output directories
are trusted local filesystem locations, not hostile concurrent sandboxes.

The ordinary verifier establishes container and receipt consistency,
not source authorship. The optional replay runs the declared original
generators again and requires byte-identical encoded WAVs. Replay can
fail across numerical runtimes even for equivalent audio. Neither mode
establishes perceptual quality or a cryptographic signature. Asset count,
total frames, payload bytes, manifest bytes and archive bytes are bounded.

## Select individual verified assets

The list command filters the verified archive by recipe tags and source.
Multiple tags are an AND match. The select command checks the entire archive,
selects the exact asset identifier and numbered variant, optionally verifies
its WAV SHA256, and publishes one caller-selected local destination without
extracting ZIP paths.

```bash
python -m sonumio.audio_core.asset_pack_cli sound-pack-list library.zip --tag ambient --tag rain
python -m sonumio.audio_core.asset_pack_cli sound-pack-select library.zip --asset rain --variant 2 --output rain-02.wav
```

Existing WAV destinations require explicit overwrite; an invalid identifier,
missing variant or failed digest leaves the output unchanged.
