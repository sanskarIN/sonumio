"""False-positive controls and hard-negative evaluation for latent research."""
import pytest

torch = pytest.importorskip('torch')

from research.latent_alignment import evaluate, _summarize
from research.latent_audio import LatentResearchModel, DiffusionSchedule
from research.text_alignment import CaptionScores


def test_retrieval_ties_split_credit_in_latent_benchmark_summary():
    scores = CaptionScores(1.0, (1.0, 1.0, 2.0), 0.2)
    result = _summarize([(scores, 1), (scores, 2)])
    assert result['pairwise_accuracy'] == pytest.approx(2 / 3)
    assert result['top1_retrieval_accuracy'] == pytest.approx(1 / 3)
    assert result['top1_uniform_chance'] == pytest.approx(1 / 4)


def test_latent_caption_blind_control_is_chance_and_reproducible():
    class Blind(torch.nn.Module):
        def forward(self, latent, steps, tokens):
            return torch.zeros_like(latent)
    model = LatentResearchModel(16)
    model.denoiser = Blind()
    model.train()
    state = torch.random.get_rng_state().clone()
    kwargs = {'seeds': (1001,), 'combinations': ((0,), (0, 1))}
    one = evaluate(model, DiffusionSchedule(8), **kwargs)
    again = evaluate(model, DiffusionSchedule(8), **kwargs)
    assert model.training
    assert torch.equal(state, torch.random.get_rng_state())
    assert one == again
    assert one['unique_rendered_clips'] == 2
    assert one['overall']['cases'] == 24
    assert one['overall']['pairwise_accuracy'] == 0.5
    assert one['overall']['top1_retrieval_accuracy'] == pytest.approx(
        one['overall']['top1_uniform_chance'])
    assert one['overall']['mean_noise_mse_advantage'] == pytest.approx(0.0)
    assert set(one['by_source_count']) == {'1', '2'}


def test_rejects_invalid_benchmark_seeds_and_members():
    model = LatentResearchModel(16)
    schedule = DiffusionSchedule(8)
    with pytest.raises(ValueError):
        evaluate(model, schedule, seeds=(1, 1))
    with pytest.raises(ValueError):
        evaluate(model, schedule, combinations=((7,),))
    with pytest.raises(ValueError):
        evaluate(model, schedule, noise_seed=-1)
