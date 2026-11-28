# Original percussion patches

Modal percussion patches define up to twelve independently damped sinusoidal
modes and three finite colored-noise layers. Modes can have static frequency
or a bounded exponential pitch sweep, integrated analytically so that the
oscillator phase is continuous. Attack and release use the same bounded
absolute-sample envelope as the pitched sound-design engine.

Nine first-party drum presets provide round, tight and deep kicks; brush,
snap and ring snares; and soft, bright and metal hats. These are original
mathematical timbres, not recordings or reproductions of named instruments.

    python -m sonumio.music_drum_cli presets score.json --assign kick=deep --assign hat=metal > drums.json
    python -m sonumio.music_drum_cli inspect score.json drums.json
    python -m sonumio.music_drum_cli render score.json drums.json --pitched patches.json --output mix.wav

A pitched bank can be combined with a percussion bank under a shared
processing budget. Unassigned tracks use the original renderer. Noise is
seeded by the event identity and layer index, independent of render order.
The score, MIDI events and mixer remain unchanged. An optional automation
document applies the existing score-clock mix. No external samples, model
weights or executable plugins are accepted.
