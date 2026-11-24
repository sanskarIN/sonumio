"""Independent noise-panel construction and codec-induced acceptance checks."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from research.codec_noise_controls import (
    NOISE_VALIDATION_SEEDS, NOISE_CONFIRMATION_SEEDS, build_noise_panel,
    compare_noise_predictions, noise_predictions,
)
from research.latent_audio import FRAMES
from research.open_set_source_probe import NOISE_SEEDS, NOISE_FAMILIES
from research.wave_diffusion import CATEGORIES


def test_panel_seed_and_content_isolation():
    a = build_noise_panel(NOISE_VALIDATION_SEEDS)
    b = build_noise_panel(NOISE_CONFIRMATION_SEEDS)
    repeated = build_noise_panel(NOISE_VALIDATION_SEEDS)
    assert a.waves.shape == (21, FRAMES)
    assert b.waves.shape == (28, FRAMES)
    assert not a.waves.flags.writeable
    assert a.hashes == repeated.hashes
    assert a.fingerprint == repeated.fingerprint
    assert len(set(a.hashes + b.hashes)) == 49
    reference = {seed for split in NOISE_SEEDS.values() for seed in split}
    assert not reference.intersection(a.seeds + b.seeds)
    for family in NOISE_FAMILIES:
        assert a.families.count(family) == 3
        assert b.families.count(family) == 4


@pytest.mark.parametrize("seeds", [(1,), (1, 1), (1, True), (-1, 2),
                                    (1, 2**32), tuple(range(9)), [1, 2]])
def test_invalid_panel_seeds(seeds):
    with pytest.raises(ValueError, match="noise panel seeds"):
        build_noise_panel(seeds)


def test_transition_accounting_distinguishes_existing_and_new_acceptances():
    panel = build_noise_panel((1921, 1922))
    before = [None] * len(panel.waves)
    after = [None] * len(panel.waves)
    before[0], after[0] = None, "rain"
    before[1], after[1] = "rain", None
    before[2], after[2] = "wind", "ocean"
    before[3], after[3] = "forest", "forest"
    result = compare_noise_predictions(panel, tuple(before), tuple(after))
    assert result["original_accepted"] == 3
    assert result["reconstructed_accepted"] == 3
    assert result["new_false_acceptances"] == 1
    assert result["removed_false_acceptances"] == 1
    assert result["changed_accepted_categories"] == 1
    assert result["reconstructed_accepted_by_category"]["ocean"] == 1
    assert result["reconstructed_accepted_by_category"]["wind"] == 0
    assert sum(row["new_false_acceptances"] for row in result["by_family"].values()) == 1
    assert not result["zero_original_and_reconstruction_acceptances"]
    assert not result["real_world_false_positive_rate_established"]
    json.dumps(result, allow_nan=False)


def test_all_abstaining_panel_passes_narrow_noise_screen():
    panel = build_noise_panel((1921, 1922))
    decisions = (None,) * len(panel.waves)
    result = compare_noise_predictions(panel, decisions, decisions)
    assert result["zero_original_and_reconstruction_acceptances"]
    assert result["reconstructed_accepted"] == 0
    assert all(row["recordings"] == 2 for row in result["by_family"].values())


@pytest.mark.parametrize("bad", [(), ("unknown",) * 14])
def test_invalid_noise_decisions(bad):
    panel = build_noise_panel((1921, 1922))
    with pytest.raises(ValueError, match="align"):
        compare_noise_predictions(panel, (None,) * 14, bad)


def test_probe_receives_finite_clipped_pcm_and_abstention_is_preserved():
    class Probe:
        def __init__(self):
            self.calls = 0
        def detailed_score(self, wave):
            assert wave.dtype == np.float32
            assert np.abs(wave).max() <= 1
            self.calls += 1
            return SimpleNamespace(source=SimpleNamespace(
                accepted=self.calls == 1, order=(1,),
            ))
    waves = np.full((2, FRAMES), 3.0, dtype=np.float64)
    probe = Probe()
    assert noise_predictions(probe, waves) == (CATEGORIES[1], None)
    assert probe.calls == 2
    assert np.all(waves == 3)


@pytest.mark.parametrize("wave", [np.zeros((1, FRAMES - 1)),
    np.full((1, FRAMES), np.nan), np.zeros((1, FRAMES), np.int16)])
def test_bad_waveform_panel_fails_before_probe_call(wave):
    with pytest.raises(ValueError, match="noise waveform panel"):
        noise_predictions(None, wave)
