# Original instrument sound design

Pitched tracks can use declarative additive oscillator patches instead of
the built-in timbres. A patch contains up to four oscillator banks with
up to sixteen unique harmonics each, independent partial decay and phase,
octave, detune, inharmonicity and bounded vibrato. An optional seeded
noise layer uses a white, finite moving-average low or high color.
Absolute-sample attack, decay, sustain and release controls have exact
zero endpoints and are shortened for very short notes.

Per-layer and per-partial amplitude budgets bound the output without
per-note normalization. The conservative oscillator guard discards
partials above the permitted frequency, rather than aliasing them
deliberately. This is not a proof of strict bandlimiting for all FM
sidebands. No external audio, plugins or learned weights are used.

Six first-party presets are included: glass, warm, reed, spark, sub and
air. A score-pinned assignment document can be created from presets
and edited as JSON:

    python -m sonumio.music_patch_cli presets score.json --assign lead=reed --assign keys=glass > patches.json
    python -m sonumio.music_patch_cli inspect score.json patches.json
    python -m sonumio.music_patch_cli render score.json patches.json --output custom.wav

An optional automation file preserves score-clock mixing. Only assigned
pitched tracks change; all other dry stems retain the original render
path. The processing budget is checked before synthesis. Render
receipts bind score, patch bank and WAV identities. This engine remains
separate from the native project and integrated render recipe formats.
