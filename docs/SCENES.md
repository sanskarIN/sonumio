# Sound scenes: experimental timeline composition

Sonumio's scene compositor works with generated **procedural** sound-effects
layers. This is a multi-track DSP editor, not open-vocabulary neural audio.
All samples are rendered to 24 kHz mono PCM WAV.

## CLI

Save this as scene.json:

\`\`\`json
{
  "duration_seconds": 6,
  "layers": [
    {"prompt": "rain in forest", "start_seconds": 0, "duration_seconds": 6, "gain_db": -10, "seed": 1},
    {"prompt": "thunder", "start_seconds": 2, "duration_seconds": 2, "gain_db": -3, "seed": 2},
    {"prompt": "wind", "start_seconds": 3, "duration_seconds": 2, "gain_db": -9, "seed": 3}
  ]
}
\`\`\`

Run:
\`\`\`sh
sonumio scene scene.json --output outputs/forest-storm.wav
\`\`\`

Or submit the same JSON to \`POST /v1/scenes\`. The response contains
\`audio_url\`, \`layer_count\`, duration, peak and gain metrics.

## Bounds and safety

- Maximum 8 layers, 30 seconds total; each layer 0.5–15 seconds.
- \`start_seconds\` determines placement on the output timeline.
- \`gain_db\` supports −60 to +12 dB; fades may not exceed layer duration.
- The final mix is peak-scaled to 0.95 if needed; no hard clipping.
- Unsupported prompts and voice impersonation requests are rejected.
- Output remains local: the API must not be exposed publicly as-is.
- Source WAV uploads and arbitrary disk-path reads are not supported.
