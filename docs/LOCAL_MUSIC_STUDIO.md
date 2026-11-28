# Local original music studio

The browser studio contains a second workspace for Sonumio's procedural
symbolic arrangement engine. It does not use a trained music model,
third-party recordings, external inference, or a model download.

## Workflow

1. Choose a style, major/minor mode, MIDI root, tempo, bar count, seed and swing.
2. Compose the score to inspect section lengths, harmony and per-track events.
3. Set eight independent track gains, mute/solo controls, room amount and
   a sample-peak ceiling. The composition remains unchanged by the mixer.
4. Render a stereo WAV through the shared local audio job queue.
5. Listen or save the WAV. Export canonical score JSON or type-1 MIDI
   independently of audio synthesis.

The preview limit is 30 seconds at 24 kHz stereo PCM16. Longer arrangements
remain available through the local arrangement CLI. MIDI stores symbolic
events, not Sonumio's timbres.

## Endpoints

All requests are JSON POSTs to the local API:

| Path | Body | Response |
| --- | --- | --- |
| /v1/music/plan | composition, optional expected_score_sha256 | score summary and digest |
| /v1/music/score | composition, optional expected_score_sha256 | canonical score JSON |
| /v1/music/midi | composition, optional expected_score_sha256 | binary MIDI |
| /v1/jobs/music | composition, optional expected_score_sha256, optional mix | shared queue job |

Example body:

```json
{
  "composition": {
    "style": "cinematic", "root": 60, "mode": "minor",
    "bpm": 120, "bars": 4, "seed": 17, "swing": 0.1
  },
  "mix": {
    "gains": [0.2, 0.14, 0.26, 0.25, 0.16, 0.23, 0.18, 0.12],
    "mute": [], "solo": [], "room": 0.15, "peak_ceiling": 0.9
  }
}
```

Omit mix for plan and export endpoints. Unknown fields, invalid numbers,
out-of-budget compositions and inconsistent score digests are rejected
before the job is submitted. The browser pins its computed composition
digest when requesting renders and exports; JSON/MIDI downloads also
verify the byte count and SHA-256 content receipt.

## Local queue and retention

Music shares the same single-worker queue and capacity as sound/scene
rendering. The client sends a UUIDv4 idempotency key; a request with the
same ID but different parameters fails rather than replacing a job.
Cancellation of active synthesis is cooperative at the job boundary and
does not interrupt an in-progress oscillator or mix routine.

The browser keeps the last music job reference in session storage.
If an HTTP submission cannot be confirmed, use Check last render rather
than creating a duplicate. If the optional local journal is enabled,
recovered successes expose only the audio reference, not the original
composition or diagnostics. WAV retention and expiry are unchanged.

The score digest identifies the symbolic arrangement; the separate WAV
digest identifies actual rendered PCM bytes. Reproducibility across
different NumPy versions/platforms is not a quality or bit-exactness claim.
The peak ceiling is a sample-peak constraint, not a true-peak or LUFS
measurement. This remains a local development preview; do not expose the
API publicly without independent authentication and resource controls.
