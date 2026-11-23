import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research import generated_source_grounding as assay
from research.latent_audio import FRAMES
from research.wave_diffusion import CATEGORIES


@pytest.fixture(scope="module")
def references():
    return assay._reference_data()


def test_spectral_signature_ignores_overall_gain_and_dc(references):
    waves, _ = references
    wave = waves["fit"][0, 0].astype(np.float64)
    np.testing.assert_allclose(
        assay.spectral_signature(wave),
        assay.spectral_signature(wave / 3 + 0.04),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("invalid", [
    np.full(FRAMES, np.nan), np.zeros(FRAMES - 1), np.ones(FRAMES, dtype=np.int16),
    np.full(FRAMES, 1.1), np.zeros((2, FRAMES)),
])
def test_rejects_invalid_waveforms(invalid):
    with pytest.raises(ValueError, match="Audio must be finite"):
        assay.spectral_signature(invalid)


def test_source_probe_keeps_reference_splits_isolated_and_scores_missing_sound(references):
    waves, hashes = references
    assert [len(hashes[split]) for split in ("fit", "calibration", "holdout")] == [24, 12, 18]
    assert len({digest for rows in hashes.values() for digest in rows}) == 54
    probe = assay._fit(waves["fit"], waves["calibration"])
    result = assay._reference_controls(probe, waves["holdout"])
    assert result["supports_interpreting_generated_scores"]
    assert result["heldout"]["accepted_rank_one"] >= 12
    assert result["silence_abstains"]
    assert result["heldout"]["mean_target_distance_margin"] > 0
    assert result["noise_controls"].keys() == {"white_8k", "filtered_to_8k"}
    assert all(len(cases) == 3 for cases in result["noise_controls"].values())
    assert all("target_rank" not in row for rows in result["noise_controls"].values() for row in rows)
    expected_noise_confounds = {
        row["closest"] for rows in result["noise_controls"].values() for row in rows
        if row["within_reference_radius"]
    }
    assert set(result["categories_accepted_for_noise"]) == expected_noise_confounds
    assert probe.score(np.full(FRAMES, 0.2)).as_dict(0) == {
        "closest": None, "distance": None, "within_reference_radius": False,
        "target_rank": None, "target_distance_margin": None,
    }


def test_probe_holdout_gate_skips_training_if_assigned_source_labels_are_wrong(monkeypatch, references):
    waves, hashes = references
    swapped = {**waves, "holdout": np.roll(waves["holdout"], 1, axis=0)}
    monkeypatch.setattr(assay, "_reference_data", lambda: (swapped, hashes))
    monkeypatch.setattr(assay, "_train_models", lambda **_: pytest.fail("Probe failure should skip training"))
    report = assay.run_study(train_steps=1, codec_steps=1, evaluations=2, sampling_seeds=(301,))
    assert report["status"] == "reference_gate_failed"
    assert report["training"] is None
    assert report["cases"] == []


def test_recording_seed_leakage_is_rejected_before_loading_data(monkeypatch):
    monkeypatch.setattr(assay, "REFERENCE_SEEDS", {**assay.REFERENCE_SEEDS, "fit": (0, 701)})
    monkeypatch.setattr(assay, "_reference_data", lambda: pytest.fail("Leaking seed should be rejected"))
    with pytest.raises(ValueError, match="seed splits overlap"):
        assay.run_study()


@pytest.mark.parametrize("options", [
    {"guidance": 0}, {"sampling_seeds": (301, 301)}, {"evaluations": 9}, {"train_steps": True},
])
def test_invalid_study_settings_fail_before_training(options):
    with pytest.raises(ValueError, match="bounded source-grounding"):
        assay.run_study(**options)


def test_trained_sampling_compares_fixed_waves_and_balanced_counterfactuals():
    rng_before = torch.random.get_rng_state().clone()
    report = assay.run_study(train_steps=2, codec_steps=1, evaluations=2, sampling_seeds=(301,))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert report["status"] == "scored"
    assert report["training"]["unique_training_waveforms"] == 42
    assert not report["reference_probe"]["overlap_with_generator_data"]
    assert report["uniform_rank_one_chance"] == 1 / 6
    assert len(report["trained_model_sha256"]) == 64
    assert len(report["cases"]) == 24
    assert len(report["null_guidance_checks"]) == 2
    assert all(row["caption_invariant"] for row in report["null_guidance_checks"])
    assert not report["weight_or_audio_files_saved"]
    json.dumps(report, allow_nan=False)

    for arm in ("raw", "latent"):
        for language in ("en", "hi"):
            rows = [row for row in report["cases"] if (row["arm"], row["language"]) == (arm, language)]
            assert {row["requested"] for row in rows} == set(CATEGORIES)
            assert {row["rotated_label"] for row in rows} == set(CATEGORIES)
            assert all(row["requested"] != row["rotated_label"] for row in rows)
            assert all(row["matched"]["closest"] == row["rotated"]["closest"] for row in rows)
            assert len({row["null_guidance"]["closest"] for row in rows}) == 1
            summaries = report["by_arm_and_language"][arm][language]
            assert summaries["null_guidance"]["rank_one"] == 1
            for condition, summary in summaries.items():
                assert summary["cases"] == 6
                assert summary["rank_one"] == sum(row[condition]["target_rank"] == 1 for row in rows)
                assert summary["accepted_rank_one"] <= summary["rank_one"]
                assert summary["accepted_rank_one_excluding_noise_confounds"] == sum(
                    row[condition]["target_rank"] == 1
                    and row[condition]["within_reference_radius"]
                    and row[condition]["closest"] not in report["reference_probe"]["categories_accepted_for_noise"]
                    for row in rows
                )
