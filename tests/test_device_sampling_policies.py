"""Coverage and matched-noise invariants for opt-in sampling policies."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from research.device_velocity_training import BatchPlanner, TrainConfig, train_model
from research.latent_audio import FRAMES
from research.text_audio import TextWaveDenoiser, ByteTextTokenizer


@pytest.mark.parametrize("paired,batch", [(False, 5), (True, 6), (True, 2)])
def test_stratified_cycles_cover_every_step_with_bounded_remainder(paired, batch):
    cfg = TrainConfig(steps=13, batch_size=batch, microbatch_size=2,
                      schedule_steps=8, timestep_sampling="stratified",
                      caption_sampling="bilingual_pair" if paired else "independent")
    planner = BatchPlanner(cfg, 12)
    all_steps = []
    for _ in range(cfg.steps):
        plan = planner.next()
        all_steps.extend(plan.timesteps[::2].tolist() if paired else plan.timesteps.tolist())
        if paired:
            assert torch.equal(plan.timesteps[::2], plan.timesteps[1::2])
            assert torch.equal(plan.noise[::2], plan.noise[1::2])
            assert torch.equal(plan.keep_caption[::2], plan.keep_caption[1::2])
            assert torch.equal(plan.indices[::2] + 1, plan.indices[1::2])
            assert torch.all(plan.indices[::2] % 2 == 0)
    for offset in range(0, len(all_steps)-7, 8):
        assert sorted(all_steps[offset:offset+8]) == list(range(8))
    assert planner.describe()["timestep_count_spread"] <= (2 if paired else 1)
    assert sum(planner.describe()["timestep_histogram"]) == cfg.steps * batch


@pytest.mark.parametrize("sampling", ["independent", "bilingual_pair"])
def test_stratification_changes_only_timesteps(sampling):
    cfg = TrainConfig(steps=4,batch_size=6,microbatch_size=2,caption_sampling=sampling)
    a = BatchPlanner(cfg,12)
    b = BatchPlanner(replace(cfg,timestep_sampling="stratified"),12)
    for _ in range(cfg.steps):
        x,y = a.next(),b.next()
        for field in ("indices","noise","keep_caption"):
            assert torch.equal(getattr(x,field),getattr(y,field))
    assert a.describe()["input_trace_sha256"] != b.describe()["input_trace_sha256"]


@pytest.mark.parametrize("kwargs", [
    {"timestep_sampling":"unsupported"}, {"caption_sampling":"unsupported"},
    {"caption_sampling":"bilingual_pair","batch_size":3},
])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_pair_sampling_rejects_odd_rows():
    cfg = TrainConfig(caption_sampling="bilingual_pair")
    with pytest.raises(ValueError,match="adjacent"):
        BatchPlanner(cfg,5)


def paired_corpus():
    waves = torch.randn(3,1,FRAMES,generator=torch.Generator().manual_seed(51))*0.1
    tokens = ByteTextTokenizer().encode(["rain","बारिश","wind","हवा","ocean","समुद्र की लहर"])
    return waves.repeat_interleave(2,dim=0),tokens


def test_pair_corpus_is_verified_before_training():
    waves,tokens = paired_corpus()
    cfg = TrainConfig(steps=1,batch_size=4,microbatch_size=2,caption_sampling="bilingual_pair")
    model = TextWaveDenoiser(16)
    bad_waves = waves.clone()
    bad_waves[1,0,0] += 1
    bad_tokens = tokens.clone()
    bad_tokens[1] = bad_tokens[0]
    null_tokens = tokens.clone()
    null_tokens[0].zero_()
    for x,y in ((bad_waves,tokens),(waves,bad_tokens),(waves,null_tokens)):
        with pytest.raises(ValueError,match="adjacent waveforms"):
            train_model(model,x,y,cfg)


def test_real_stratified_pair_training_and_microbatch_invariance():
    waves,tokens = paired_corpus()
    cfg = TrainConfig(steps=4,batch_size=4,microbatch_size=3,
                      caption_sampling="bilingual_pair",timestep_sampling="stratified")
    reports = []
    with torch.random.fork_rng(devices=[]):
        for micro in (3,1):
            torch.manual_seed(17)
            reports.append(train_model(TextWaveDenoiser(16),waves,tokens,replace(cfg,microbatch_size=micro)))
    assert reports[0]["batch_plan"] == reports[1]["batch_plan"]
    assert reports[0]["batch_plan"]["timestep_histogram"] == [2]*8
    assert reports[0]["microbatch_backward_calls"] == 8
    assert reports[1]["microbatch_backward_calls"] == 16
    assert reports[0]["batch_plan"]["distinct_caption_rows_seen"] % 2 == 0
    assert reports[0]["initial_state_sha256"] == reports[1]["initial_state_sha256"]
