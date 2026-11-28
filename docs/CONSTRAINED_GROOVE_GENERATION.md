# Constrained percussion grooves

The original symbolic arrangement can be revised on its 16-step percussion
grid. Each selected track has an exact hit count, mandatory and forbidden
positions, internal spacing bounds and optional cyclic spacing. A bounded
dynamic program minimizes a declared integer objective containing original
hit retention, metrical accents, deterministic seed variation and spacing.

    python -m sonumio.music_groove_cli plan score.json --rule kick:5 --rule snare:2 --rule hat:8 --style syncopated --alternate-bars > groove.json
    python -m sonumio.music_groove_cli inspect score.json groove.json
    python -m sonumio.music_groove_cli apply score.json groove.json --output varied.json

The defaults preserve intro and break sections. A zero-hit rule removes
the selected drum events. Infeasible constraints and crossing note
boundaries are rejected; no adjacent bar is silently modified. The score
clock, harmony, section layout and every pitched event remain unchanged.
Original-score identity is checked before any transformation, and output
is validated by the existing arrangement contract. Cyclic spacing is
within each solved bar, not a cross-bar spacing guarantee.

The optimizer uses a deterministic lexicographic tie break and reports
exact cost and transition count. Numeric tests compare the solver with
independent exhaustive enumeration. These are symbolic rhythm-generation
rules, not a learned model or evidence of musical preference.
