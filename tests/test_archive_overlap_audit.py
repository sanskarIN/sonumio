"""Exhaustive numerical overlap tests, including private archive round trips."""
import json

import numpy as np
import pytest

from research.archive_overlap_audit import (
    CorrelationPlan, OverlapPolicy, _pair_plan, audit_archive_overlap, main,
)


def noise(seed, frames):
    return np.random.default_rng(seed).normal(0, .1, frames).astype(np.float32)


def oracle(a, b, shift):
    result = []
    for lag in range(-shift, shift + 1):
        n = len(a) - abs(lag)
        x = a[max(-lag, 0):max(-lag, 0)+n].astype(np.float64)
        y = b[max(lag, 0):max(lag, 0)+n].astype(np.float64)
        x -= x.mean()
        y -= y.mean()
        result.append(np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y)))
    return result


@pytest.mark.parametrize("frames,shift", [(31, 0), (32, 16), (127, 53), (2048, 1024)])
def test_every_fft_lag_matches_direct_oracle(frames, shift):
    a, b = noise(1, frames), noise(2, frames)
    plan = CorrelationPlan(frames, OverlapPolicy(max_shift=shift))
    actual = plan.correlations(plan.prepare(a), plan.prepare(b))
    np.testing.assert_allclose(actual, oracle(a, b, shift), atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("lag", [-64, -15, 0, 17, 64])
@pytest.mark.parametrize("gain", [-.5, .25, 2.0])
def test_shift_gain_offset_and_polarity(lag, gain):
    frames = 128
    a, b = noise(12, frames), noise(13, frames)
    length = frames - abs(lag)
    x, y = max(-lag, 0), max(lag, 0)
    b[y:y+length] = gain*a[x:x+length] + .04
    plan = CorrelationPlan(frames, OverlapPolicy(max_shift=64))
    pa, pb = plan.prepare(a), plan.prepare(b)
    values = plan.correlations(pa, pb)
    assert values[lag+64] == pytest.approx(np.sign(gain), abs=1e-12)
    assert plan.compare(pa, pb).screened_match
    np.testing.assert_allclose(values, plan.correlations(pb, pa)[::-1], atol=1e-11)


def test_no_circular_wrap_or_padding_in_overlap():
    a = noise(3, 128)
    b = np.roll(a, 20)
    plan = CorrelationPlan(128, OverlapPolicy(max_shift=19))
    assert not plan.compare(plan.prepare(a), plan.prepare(b)).screened_match
    plan = CorrelationPlan(128, OverlapPolicy(max_shift=20))
    assert plan.compare(plan.prepare(a), plan.prepare(b)).screened_match


def test_constant_and_low_energy_overlap_remain_undefined():
    plan = CorrelationPlan(128, OverlapPolicy(max_shift=64))
    a = noise(9, 128)
    result = plan.compare(plan.prepare(a), plan.prepare(np.full(128, .1, np.float32)))
    assert result.defined_lags == 0 and result.max_absolute_correlation is None
    assert not result.screened_match
    b = a.copy()
    b[:64] = 0
    values = plan.correlations(plan.prepare(a), plan.prepare(b))
    assert np.isnan(values[0])
    assert np.isfinite(values[64])


def test_independent_noise_not_flagged():
    plan = CorrelationPlan(2048, OverlapPolicy())
    result = plan.compare(plan.prepare(noise(1, 2048)), plan.prepare(noise(2, 2048)))
    assert result.defined_lags == 2049
    assert result.max_absolute_correlation < .3
    assert not result.screened_match


@pytest.mark.parametrize("bad", [
    OverlapPolicy(max_shift=-1), OverlapPolicy(max_shift=True),
    OverlapPolicy(max_shift=65), OverlapPolicy(max_shift=1.0),
    OverlapPolicy(max_shift=0, threshold=float("nan")),
    OverlapPolicy(max_shift=0, threshold=True),
    OverlapPolicy(max_shift=0, threshold=1.0),
    OverlapPolicy(max_shift=0, threshold=.8),
    OverlapPolicy(max_shift=0, pair_budget=0),
    OverlapPolicy(max_shift=0, pair_budget=100001),
])
def test_policy_rejection(bad):
    with pytest.raises(ValueError):
        CorrelationPlan(128, bad)


@pytest.mark.parametrize("bad", [
    np.zeros(32, np.float64), np.zeros((1, 32), np.float32),
    np.zeros(31, np.float32), np.full(32, np.nan, np.float32),
    np.full(32, np.inf, np.float32),
])
def test_input_validation(bad):
    plan = CorrelationPlan(32, OverlapPolicy(max_shift=0))
    with pytest.raises(ValueError):
        plan.prepare(bad)


def test_pair_budget_is_global_and_never_partial():
    counts = {"train": 5, "validation": 3, "test": 2}
    assert sum(n for _, _, n in _pair_plan(counts, OverlapPolicy(pair_budget=31))) == 31
    with pytest.raises(ValueError, match="budget"):
        _pair_plan(counts, OverlapPolicy(pair_budget=30))


@pytest.fixture
def archive_builder(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from research.corpus_archive import _write_archive
    from test_corpus_archive import fixture_splits

    def build(kind, shard_clips):
        splits, report = fixture_splits()
        a = splits["train"].clips[0, 0]
        if kind == "shifted":
            b = splits["validation"].clips[0, 0]
            b[37:] = -.5*a[:-37] + .02
        elif kind == "constant":
            splits["validation"].clips[0, 0] = .02
        path = tmp_path / f"{kind}-{shard_clips}.scorpus"
        result = _write_archive(splits, report, path, shard_clips=shard_clips)
        return path, result.archive_sha256
    return build


@pytest.mark.parametrize("shard_clips", [1, 2, 5])
def test_archive_counts_matches_and_private_report(archive_builder, shard_clips):
    from research.corpus_archive import CorpusArchiveReader
    path, digest = archive_builder("shifted", shard_clips)
    with CorpusArchiveReader(path, expected_sha256=digest) as reader:
        report = audit_archive_overlap(reader, OverlapPolicy(max_shift=40))
    assert report["examined_pairs"] == 31
    assert report["potential_overlap_pairs"] == 1
    assert report["undefined_lags"] == 0
    assert not report["screening_clear"]
    assert report["split_pairs"]["train_vs_validation"]["potential_overlap_pairs"] == 1
    assert report["split_pairs"]["validation_vs_test"]["examined_pairs"] == 6
    payload = json.dumps(report, allow_nan=False)
    for forbidden in ("asset-train", "group-train", "rain", str(path)):
        # The split names themselves include "train"; check whole caption.
        if forbidden == "rain":
            assert '"rain"' not in payload
        else:
            assert forbidden not in payload


def test_cli_clear_and_candidate_exit_codes(archive_builder, capsys):
    for kind, expected in (("independent", 0), ("shifted", 2)):
        path, digest = archive_builder(kind, 2)
        assert main([str(path), "--expected-sha256", digest, "--max-shift", "40",
                     "--require-clear"]) == expected
        assert json.loads(capsys.readouterr().out)["screening_clear"] == (expected == 0)


@pytest.mark.parametrize("invalid", ["budget", "digest"])
def test_cli_failed_verification_emits_no_report(archive_builder, capsys, invalid):
    path, digest = archive_builder("independent", 2)
    args = [str(path), "--expected-sha256", digest]
    if invalid == "budget":
        args += ["--pair-budget", "30"]
    else:
        args[2] = "0"*64
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    capture = capsys.readouterr()
    assert not capture.out and str(path) not in capture.err
