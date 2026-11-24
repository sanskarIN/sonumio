"""Reversible waveform basis numerical and actual training regressions."""
import json

import pytest

torch = pytest.importorskip("torch")

from research.latent_audio import FRAMES
from research.multiscale_waveform import PATCH, pack_waveform
from research.multiscale_waveform_study import state_sha256
from research.reversible_waveform_bases import (
    BASES, ReversibleBasisDenoiser, analysis_waveform, synthesis_waveform,
    orthonormal_basis, basis_sha256,
)
from research.reversible_basis_study import run_study
from research.text_audio import ByteTextTokenizer


@pytest.mark.parametrize("name", BASES)
def test_basis_orthogonality_and_full_waveform_reconstruction(name):
    basis = orthonormal_basis(name)
    torch.testing.assert_close(basis @ basis.T, torch.eye(PATCH), atol=2e-7, rtol=0)
    wave = torch.randn(3,1,FRAMES,generator=torch.Generator().manual_seed(13))
    coordinates = analysis_waveform(wave,basis)
    restored = synthesis_waveform(coordinates,basis)
    assert restored.shape == wave.shape
    torch.testing.assert_close(restored,wave,atol=2e-6,rtol=1e-6)
    torch.testing.assert_close(coordinates.square().sum(),wave.square().sum(),rtol=2e-6,atol=1e-4)


@pytest.mark.parametrize("name", ["dct","haar"])
def test_dc_energy_concentrates_in_first_coordinate(name):
    wave = torch.ones(1,1,FRAMES)
    coordinates = pack_waveform(analysis_waveform(wave,orthonormal_basis(name)))
    torch.testing.assert_close(coordinates[:,0],torch.full_like(coordinates[:,0],PATCH**.5))
    torch.testing.assert_close(coordinates[:,1:],torch.zeros_like(coordinates[:,1:]),atol=2e-7,rtol=0)


@pytest.mark.parametrize("name", BASES)
def test_inverse_gradient_preserves_waveform_error(name):
    basis = orthonormal_basis(name)
    wave = torch.randn(2,1,FRAMES,generator=torch.Generator().manual_seed(14),requires_grad=True)
    restored = synthesis_waveform(analysis_waveform(wave,basis),basis)
    restored.square().mean().backward()
    torch.testing.assert_close(wave.grad,2*wave.detach()/wave.numel(),atol=2e-9,rtol=2e-6)


@pytest.mark.parametrize("basis", [torch.zeros(8,8),torch.eye(7),torch.eye(8).double(),
                                  torch.full((8,8),float("nan"))])
def test_invalid_matrix_rejected(basis):
    with pytest.raises(ValueError,match="orthonormal"):
        analysis_waveform(torch.zeros(1,1,FRAMES),basis)


def test_basis_names_and_distinct_hashes():
    assert len({basis_sha256(orthonormal_basis(n)) for n in BASES}) == 3
    with pytest.raises(ValueError,match="Unknown"):
        orthonormal_basis("unsupported")


@pytest.mark.parametrize("name", BASES)
def test_aligned_initial_function_and_parameter_count(name):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        reference = ReversibleBasisDenoiser("identity").eval()
        torch.manual_seed(17)
        model = ReversibleBasisDenoiser(name).eval()
    assert state_sha256(reference) == state_sha256(model)
    assert set(reference.state_dict()) == set(model.state_dict())
    assert "basis" not in model.state_dict()
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in model.parameters())
    wave = torch.randn(3,1,FRAMES,generator=torch.Generator().manual_seed(15))
    time = torch.tensor([0,4,7])
    tokens = ByteTextTokenizer().encode(["rain","हवा","ocean"])
    tokens[1].zero_()
    model.align_initial_function()
    with torch.no_grad():
        torch.testing.assert_close(model(wave,time,tokens),reference(wave,time,tokens),atol=2e-5,rtol=2e-5)
    assert model.describe()["initial_function_aligned"]
    with pytest.raises(ValueError,match="already aligned"):
        model.align_initial_function()


def test_actual_training_and_ddim_study():
    before = torch.random.get_rng_state().clone()
    report = run_study(train_steps=1,evaluations=2,spectral_weight=.2)
    assert torch.equal(before,torch.random.get_rng_state())
    assert report["same_initial_unrotated_state"]
    assert report["aligned_initial_waveform_functions"]
    assert report["same_training_input_trace"]
    assert report["equal_trainable_parameter_counts"]
    assert report["coefficient_count_preserved"]
    assert report["unique_training_recordings"] == report["unique_validation_recordings"] == 42
    assert not report["exact_split_overlap"]
    assert not report["weights_or_audio_saved"]
    assert set(report["arms"]) == set(BASES)
    for arm in report["arms"].values():
        assert arm["initial_function_check"]["alignment_within_tolerance"]
        assert arm["heldout_interventions"]["caption_cases_per_timestep"] == 84
        if report["reference_probe"]["gate"]["passed"]:
            assert len(arm["generated"]["cases"]) == 12
            assert len(arm["generated"]["balanced_assignment"]) == 2
            assert all(p["accepted"]["permutations"] == 720 for p in arm["generated"]["balanced_assignment"])
        else:
            assert arm["generated"] is None
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize("kwargs",[{"train_steps":0},{"sampling_seeds":(1,1)},
                                   {"evaluations":1},{"spectral_weight":float("nan")}])
def test_invalid_study_config(kwargs):
    with pytest.raises(ValueError,match="bounded"):
        run_study(**kwargs)
