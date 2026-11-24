import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research import open_set_source_probe as probe
from research import open_set_source_study as study
from research.latent_audio import FRAMES


@pytest.fixture(scope="module")
def panels():
    return probe.reference_panels()


@pytest.fixture(scope="module")
def fitted(panels):
    sources, noises, _ = panels
    return probe.fit_probe(sources["fit"], sources["calibration"],
                           noises["fit"], noises["calibration"])


@pytest.mark.parametrize("family", probe.NOISE_FAMILIES)
def test_noise_controls_are_bounded_and_deterministic(family):
    a = probe.noise_wave(family, 17)
    b = probe.noise_wave(family, 17)
    c = probe.noise_wave(family, 18)
    assert a.shape == (FRAMES,)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()
    assert np.abs(a).max() <= 0.900001
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_phase_surrogate_preserves_relative_spectral_power(panels):
    sources, _, _ = panels
    original = sources["holdout"][2, 0].astype(np.float64)
    surrogate = probe.phase_surrogate(original, 42).astype(np.float64)
    assert not np.array_equal(original, surrogate)
    def power(x):
        p = np.abs(np.fft.rfft(x - x.mean())) ** 2
        return p / p.sum()
    np.testing.assert_allclose(power(original), power(surrogate), rtol=1e-4, atol=1e-9)


def test_extended_signature_gain_and_dc_invariant(panels):
    sources, _, _ = panels
    wave = sources["fit"][0, 0].astype(np.float64)
    np.testing.assert_allclose(probe.extended_signature(wave),
                               probe.extended_signature(wave * 0.3 + 0.02),
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("invalid", [
    np.zeros(FRAMES - 1), np.ones(FRAMES, dtype=np.int16),
    np.full(FRAMES, np.nan), np.full(FRAMES, 2.0),
])
def test_invalid_audio_rejected(invalid):
    with pytest.raises(ValueError, match="Audio must be finite"):
        probe.extended_signature(invalid)


def test_silence_and_dc_abstain(fitted):
    for wave in (np.zeros(FRAMES), np.full(FRAMES, 0.25)):
        decision = fitted.detailed_score(wave)
        assert decision.source.order == ()
        assert not decision.source.accepted
        assert decision.noise_margin is None


def test_panels_have_disjoint_hashes_and_expected_sizes(panels):
    sources, noises, hashes = panels
    assert [sources[s].shape[:2] for s in ("fit", "calibration", "holdout")] == [(6, 4), (6, 2), (6, 3)]
    assert [noises[s].shape[:2] for s in ("fit", "calibration", "holdout")] == [(7, 4), (7, 3), (7, 4)]
    all_hashes = [h for domain in hashes.values() for values in domain.values() for h in values]
    assert len(all_hashes) == len(set(all_hashes)) == 131


def test_noise_calibration_rejects_every_calibration_negative(panels, fitted):
    _, noises, _ = panels
    for waveform in noises["calibration"].reshape(-1, FRAMES):
        decision = fitted.detailed_score(waveform)
        assert not decision.noise_margin_passed
        assert not decision.source.accepted
    assert not fitted.source_centers.flags.writeable


def test_heldout_decisions_and_gate_are_consistent(panels, fitted):
    sources, noises, _ = panels
    report = probe.evaluate_probe(fitted, sources["holdout"], noises["holdout"])
    assert report["source_cases"] == 18
    assert report["noise_cases"] == 28
    assert report["correct_accepted_sources"] == sum(
        x["correct_accepted"] for x in report["by_source_category"].values()
    )
    assert report["false_accepted_noise"] == sum(
        x["false_acceptances"] for x in report["by_noise_family"].values()
    )
    for row in report["source_decisions"] + report["noise_decisions"]:
        assert row["within_reference_radius"] == (row["source_radius_passed"] and row["noise_margin_passed"])
    expected = (report["correct_accepted_sources"] >= 12
                and all(x["correct_accepted"] >= 1 for x in report["by_source_category"].values())
                and report["false_accepted_noise"] == 0)
    assert report["gate"]["passed"] == expected


def test_rejects_seed_overlap_before_generation(monkeypatch):
    monkeypatch.setattr(probe, "NOISE_SEEDS", {**probe.NOISE_SEEDS, "fit": (0, 1801)})
    monkeypatch.setattr(probe, "_reference_data", lambda: pytest.fail("Overlap must fail first"))
    with pytest.raises(ValueError, match="seed overlap"):
        probe.reference_panels()


def test_failed_gate_never_calls_trainer(monkeypatch):
    original = study.evaluate_probe
    def rejected(*args):
        result = original(*args)
        result["gate"]["passed"] = False
        return result
    monkeypatch.setattr(study, "evaluate_probe", rejected)
    monkeypatch.setattr(study, "_train_models", lambda **kw: pytest.fail("Gate failed"))
    result = study.run_study(train=True, train_steps=1, codec_steps=1, evaluations=2)
    assert result["status"] == "reference_gate_failed"
    assert result["training"] is None
    assert result["generated_cases"] == []
    assert not result["model_readiness_established"]


def test_real_holdout_study_reports_controls_and_leakage_checks():
    state = torch.random.get_rng_state().clone()
    result = study.run_study()
    assert torch.equal(torch.random.get_rng_state(), state)
    assert result["status"] == "probe_only"
    assert not result["overlap_with_generator_data"]
    assert result["open_set_holdout"]["noise_cases"] == 28
    assert len(result["phase_surrogate_challenge"]["cases"]) == 18
    assert not result["phase_surrogate_challenge"]["included_in_noise_gate"]
    assert not result["weights_or_audio_saved"]
    assert result["training"] is None
    json.dumps(result, allow_nan=False)


def test_real_training_if_reference_gate_passes():
    result = study.run_study(train=True, train_steps=1, codec_steps=1, evaluations=2)
    if result["open_set_holdout"]["gate"]["passed"]:
        assert result["status"] == "generated"
        assert len(result["generated_cases"]) == 24
        assert len(result["null_guidance_checks"]) == 2
        assert result["training"] is not None
    else:
        assert result["status"] == "reference_gate_failed"
        assert not result["generated_cases"]
        assert result["trained_model_sha256"] is None


@pytest.mark.parametrize("options", [
    {"train": 1}, {"seed": True}, {"evaluations": 9},
    {"sampling_seeds": (301, 301)}, {"codec_steps": 0},
])
def test_invalid_study_settings(options):
    with pytest.raises(ValueError, match="bounded open-set study"):
        study.run_study(**options)
