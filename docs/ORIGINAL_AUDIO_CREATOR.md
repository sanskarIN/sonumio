# Original audio creator workflow

The new music and nature tools are local, procedural and source-only.
The original melody, bar structure and combined neural studies are
research-only; no trained weights are distributed. The existing default
SFX engine remains unchanged.

## Controlled English and Hindi prompts

```bash
python -m sonumio.audio_prompt_cli "music cinematic minor key C bpm 120 bars 4 seed 17" --output outputs/music.wav
python -m sonumio.audio_prompt_cli "प्रकृति जंगल सेकंड 2 सीड 29" --output outputs/forest.wav
```

The grammar requires a leading music/instrumental or nature/sound
keyword, with Hindi equivalents. Music accepts calm, cinematic or
rhythmic; major/minor; key (note name or MIDI tonic 48–71), bpm
60–160, bars 4–32 within 120 seconds, swing 0–0.6 and seed. Nature
accepts forest, storm, coast, creek or campfire; seconds 1–30 and
seed. Numeric arguments use ASCII digits. Unsupported words and
duplicate directives fail closed. This is not unrestricted natural
language understanding. Reports contain a canonical prompt that
reconstructs the same plan.

## Editable source documents and soundtrack

```bash
python -m sonumio.arrangement_cli compose --style cinematic --mode minor --bars 4 --bpm 160 --seed 17 --output outputs/score.json
python -m sonumio.nature_cli plan forest --seconds 2 --seed 29 --output outputs/nature.json
python -m sonumio.music_edit_cli outputs/score.json edits.json --output outputs/edited.json
python -m sonumio.arrangement_cli midi outputs/edited.json --output outputs/edited.mid
python -m sonumio.soundtrack_cli outputs/edited.json outputs/nature.json --frames 48000 --music-source-start 12000 --music-frames 48000 --ambience-frames 48000 --duck --output outputs/soundtrack.wav
```

For example, edits.json can contain:

```json
{
  "format": "sonumio-score-edits-v1",
  "operations": [
    {"type": "retime", "parameters": {"bpm": 160, "swing": 0.15}},
    {"type": "transpose", "parameters": {"semitones": 1}},
    {"type": "articulation", "parameters": {"track": "lead", "first_bar": 0, "last_bar": 4, "velocity_scale": 0.8}},
    {"type": "remove_track", "parameters": {"track": "hat", "first_bar": 1, "last_bar": 3}}
  ]
}
```

Bar indexes are zero-based with an exclusive end. Edits preserve the
original immutable score and produce an ordered semantic digest trace.
They reject out-of-range pitch or timing instead of silently wrapping
notes. Other supported operations are lead_bar (onset, duration,
absolute MIDI pitch, velocity arrays) and quantize_velocity.

The nature engine has seven independent noise/event sources: rain,
wind, ocean, thunder, birds, stream and fire sound. Each supports
intensity, texture, motion and independent seeded variation. Scene
layers use exact frames, bounded gain/pan automation, width and fades.
The scene JSON envelope carries a semantic digest, not a digital
signature or guarantee of authorship.

The soundtrack renderer crops and places both original sources at
24 kHz, applies independent fades/gains and optionally reduces the
ambience in response to music energy. The causal detector uses the
previous completed 10 ms block; no future samples affect earlier
gain. Sample-peak attenuation does not measure LUFS or true peak.
Source and output digests can be pinned independently. Every CLI
publishes one file atomically; no multi-file transaction is claimed.

## Verified research results and limitations

The [CPU music research workflow](https://github.com/sanskarIN/sonumio/actions/runs/45843782604)
at commit aeeb78ca0bc1775f4ce024b0d733559376004fdb passed its
338-test suite plus melody, structure and nature research experiments.

The bar-structure GRU achieved mean joint NLL 0.185944 versus
1.519173 for conditional counts. However, all 36 unique-context
development rows had exact chord/rhythm sequence matches in training.
The novel-token subset was empty. This is evidence of learning a finite
synthetic source, not generalization to unseen musical structure.

All seven nature source families passed exact seeded replay and
zero-intensity silence controls; all varied across the three tested
seeds and increased RMS between the prespecified low/high intensity
settings. These are numerical controls, not listening-quality evidence.
The old and new procedural renderers are not loudness-matched.

The combined experiment separately trains the chord/rhythm and melody
models on paired source scores, then compares procedural, learned
structure, learned melody and learned-both variants on the same
generation plans. It does not feed held-out target tokens into sampling.
Actual stereo renders are measured in memory. The combined experiment,
controlled prompt and latest integration tests require their own
final-head verification; prior CI success does not cover new commits.
