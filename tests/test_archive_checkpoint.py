"""Private numeric checkpoint format, exact resume, and fail-closed regressions."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.archive_checkpoint import (
    CheckpointOptions, plan_checkpoint_outputs, restore_training_checkpoint,
    tensor_specs, MAX_PERIODIC_CHECKPOINTS,
)
from research.archive_checkpoint_format import (
    MAGIC, HEADER_LENGTH, MAX_HEADER_BYTES, TensorSpec, encode_checkpoint,
    decode_checkpoint, read_checkpoint_metadata, maximum_encoded_bytes,
)
from research.archive_training_contracts import (
    ArchiveTrainingConfig, TrainingProgress, make_archive_optimizer,
    require_optimizer_contract, runtime_identity,
)
from research.archive_wave_training import train_archive, main
from research.archive_validation import model_fingerprint
from research.archive_readiness import audit_archive
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError, canonical_json, digest
from research.wave_diffusion import TinyWaveDenoiser
from test_archive_training import fixture, build


def unpack(blob):
    start = len(MAGIC)+HEADER_LENGTH.size
    size, = HEADER_LENGTH.unpack_from(blob, len(MAGIC))
    stop = start+size
    return json.loads(blob[start:stop]), blob[stop:]


def pack(header, payload):
    header_bytes = canonical_json(header)
    return MAGIC+HEADER_LENGTH.pack(len(header_bytes))+header_bytes+payload


def mutate_header(blob, change):
    header, payload = unpack(blob)
    change(header)
    return pack(header, payload)


def mutate_tensor(blob, predicate, change):
    header, payload = unpack(blob)
    payload = bytearray(payload)
    descriptor = next(item for item in header["tensors"] if predicate(item["name"]))
    start = descriptor["offset"]
    stop = start+descriptor["length"]
    dtype = "<f4" if descriptor["dtype"] == "f32" else "u1"
    array = np.frombuffer(payload[start:stop], dtype=dtype).copy().reshape(descriptor["shape"])
    change(array)
    raw = array.tobytes()
    assert len(raw) == stop-start
    payload[start:stop] = raw
    descriptor["sha256"] = digest(raw)
    return pack(header, bytes(payload))


@pytest.fixture
def example():
    specs = {
        "weights": TensorSpec("f32", (2, 3)),
        "step": TensorSpec("f32", ()),
        "rng": TensorSpec("u8", (5,)),
    }
    arrays = {
        "weights": np.arange(6, dtype=np.float32).reshape(2, 3),
        "step": np.array(2, dtype=np.float32),
        "rng": np.arange(5, dtype=np.uint8),
    }
    metadata = {"version": 1, "private": True}
    return specs, arrays, metadata, encode_checkpoint(metadata, arrays, specs)


def test_codec_exact_deterministic_and_read_only(example):
    specs, arrays, metadata, blob = example
    assert len(blob) <= maximum_encoded_bytes(specs)
    assert read_checkpoint_metadata(blob) == metadata
    actual, restored = decode_checkpoint(blob, specs)
    assert actual == metadata
    for name, expected in arrays.items():
        assert restored[name].dtype == expected.dtype
        assert restored[name].shape == expected.shape
        assert not restored[name].flags.writeable
        np.testing.assert_array_equal(restored[name], expected)
    assert encode_checkpoint(metadata, dict(reversed(list(arrays.items()))), specs) == blob


@pytest.mark.parametrize("change", [
    lambda b: b"",
    lambda b: b"WRONG123"+b[len(MAGIC):],
    lambda b: MAGIC+HEADER_LENGTH.pack(0)+b[12:],
    lambda b: MAGIC+HEADER_LENGTH.pack(MAX_HEADER_BYTES+1)+b[12:],
    lambda b: b[:12],
    lambda b: b[:-1],
    lambda b: b+b"x",
    lambda b: b[:-1]+bytes([b[-1]^1]),
])
def test_codec_rejects_corruption(example, change):
    specs, _, _, blob = example
    with pytest.raises(ArchiveError):
        decode_checkpoint(change(blob), specs)


@pytest.mark.parametrize("change", [
    {"name": "unknown"},
    {"dtype": "f64"},
    {"shape": [True]},
    {"shape": [2**60]},
    {"offset": False},
    {"offset": 1},
    {"length": True},
    {"length": 1000000000},
    {"sha256": "x"*64},
    {"sha256": "0"*64},
    {"extra": 1},
])
def test_codec_rejects_untrusted_tensor_descriptors(example, change):
    specs, _, _, blob = example
    broken = mutate_header(blob, lambda h: h["tensors"][0].update(change))
    with pytest.raises(ArchiveError):
        decode_checkpoint(broken, specs)


@pytest.mark.parametrize("change", [
    lambda h: h["tensors"].reverse(),
    lambda h: h["tensors"].append(h["tensors"][0]),
    lambda h: h["tensors"].pop(),
    lambda h: h.update({"unexpected": True}),
    lambda h: h.update({"metadata": []}),
    lambda h: h.update({"tensors": {}}),
])
def test_codec_rejects_missing_duplicate_reordered_and_extra_fields(example, change):
    specs, _, _, blob = example
    with pytest.raises(ArchiveError):
        decode_checkpoint(mutate_header(blob, change), specs)


def test_codec_rejects_noncanonical_metadata(example):
    specs, _, _, blob = example
    header, payload = unpack(blob)
    raw = json.dumps(header, indent=2).encode()
    with pytest.raises(ArchiveError):
        decode_checkpoint(MAGIC+HEADER_LENGTH.pack(len(raw))+raw+payload, specs)


def test_codec_rejects_nonfinite_values_even_after_rehash(example):
    specs, arrays, metadata, blob = example
    arrays["weights"][0, 0] = np.nan
    with pytest.raises(ArchiveError, match="nonfinite"):
        encode_checkpoint(metadata, arrays, specs)
    broken = mutate_tensor(blob, lambda n: n == "weights", lambda a: a.fill(np.inf))
    with pytest.raises(ArchiveError, match="nonfinite"):
        decode_checkpoint(broken, specs)


@pytest.mark.parametrize("dtype,shape", [
    ("f64", (1,)), ("f32", [1]), ("f32", (0,)), ("f32", (-1,)),
    ("f32", (True,)), ("u8", (1,)*9), ("f32", (2**40,)),
])
def test_tensor_schema_rejects_invalid_dimensions(dtype, shape):
    with pytest.raises(ArchiveError):
        TensorSpec(dtype, shape)


@pytest.mark.parametrize("name,value", [
    ("weights", np.zeros((3, 2), dtype=np.float32)),
    ("weights", np.zeros((2, 3), dtype=np.float64)),
    ("rng", np.zeros(5, dtype=np.int8)),
    ("step", np.array([2], dtype=np.float32)),
])
def test_encoder_rejects_array_schema_mismatch(example, name, value):
    specs, arrays, metadata, _ = example
    arrays[name] = value
    with pytest.raises(ArchiveError):
        encode_checkpoint(metadata, arrays, specs)


@pytest.mark.parametrize("change", [
    {"completed_steps": True}, {"completed_steps": -1},
    {"completed_steps": 50001}, {"first_loss": 1.0},
    {"last_loss": 0.0}, {"loss_total": 2.0},
    {"completed_steps": 1, "first_loss": 1.0, "last_loss": 2.0, "loss_total": 2.0},
    {"completed_steps": 2, "first_loss": 1.0, "last_loss": 2.0, "loss_total": 0.0},
    {"completed_steps": 1, "first_loss": float("nan"), "last_loss": 0, "loss_total": 0},
    {"completed_steps": 1, "first_loss": True, "last_loss": 1, "loss_total": 1},
])
def test_training_progress_contract(change):
    with pytest.raises(ArchiveError):
        TrainingProgress(**change)


def test_progress_roundtrip_and_advance():
    first = TrainingProgress().advance(.25)
    second = first.advance(.5)
    assert first.completed_steps == 1
    assert second.as_dict() == {
        "completed_steps": 2, "first_loss": .25, "last_loss": .5, "loss_total": .75,
    }
    assert TrainingProgress.parse(second.as_dict()) == second
    with pytest.raises(ArchiveError):
        TrainingProgress.parse({**second.as_dict(), "extra": 1})


@pytest.mark.parametrize("loss", [-1, float("nan"), float("inf"), True, "2"])
def test_progress_refuses_invalid_losses(loss):
    with pytest.raises(ArchiveError):
        TrainingProgress().advance(loss)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("private-training-checkpoint")
    archive = directory/"fixture.scorpus"
    result = build(archive, fixture())
    config = ArchiveTrainingConfig(
        steps=2, batch_size=2, channels=8, diffusion_steps=8,
        validation_repeats=1, validation_batch_size=3,
    )
    path = directory/"prefix.stcheckpoint"
    report = train_archive(archive, expected_sha256=result.archive_sha256, config=config,
                           checkpoint=CheckpointOptions(final_path=path))
    return archive, result.archive_sha256, config, path, report


def test_checkpoint_saved_private_with_verified_receipt(trained):
    _, _, config, path, report = trained
    blob = path.read_bytes()
    receipt, = report["checkpoint_receipts"]
    assert receipt["sha256"] == hashlib.sha256(blob).hexdigest()
    assert receipt["bytes_written"] == len(blob)
    assert receipt["completed_steps"] == config.steps
    assert report["private_checkpoint_files_written"]
    assert not report["audio_files_written"]
    assert not report["model_publication_authorized"]
    metadata = read_checkpoint_metadata(blob)
    assert metadata["private_checkpoint"]
    assert not metadata["original_sonumio_1"]
    assert metadata["model_sha256"] == report["trained_model_sha256"]
    assert metadata["sampler"] == report["sampler"]
    assert "steps" not in metadata["configuration"]


def test_exact_interrupted_resume_matches_uninterrupted_checkpoint(trained, tmp_path):
    archive, sha, config, prefix, prefix_report = trained
    full_config = replace(config, steps=4)
    uninterrupted = tmp_path/"full.stcheckpoint"
    continued = tmp_path/"resumed.stcheckpoint"
    rng_before = torch.get_rng_state().clone()
    direct = train_archive(archive, expected_sha256=sha, config=full_config,
                           checkpoint=CheckpointOptions(final_path=uninterrupted))
    resumed = train_archive(
        archive, expected_sha256=sha, config=full_config,
        checkpoint=CheckpointOptions(
            resume_from=prefix, resume_sha256=prefix_report["checkpoint_receipts"][0]["sha256"],
            final_path=continued,
        ),
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    for key in ("sampler", "initial_model_sha256", "trained_model_sha256",
                "initial_step_noise_mse", "last_step_noise_mse", "mean_training_noise_mse",
                "validation"):
        assert direct[key] == resumed[key]
    assert resumed["resumed_steps"] == 2 and resumed["new_steps"] == 2
    assert direct["new_steps"] == 4
    assert continued.read_bytes() == uninterrupted.read_bytes()


def test_resume_at_existing_step_does_not_optimize(trained, tmp_path, monkeypatch):
    archive, sha, config, prefix, report = trained
    def forbidden(*args, **kwargs):
        pytest.fail("completed checkpoint should not execute an optimizer step")
    monkeypatch.setattr("research.archive_wave_training.training_step", forbidden)
    output = tmp_path/"same.stcheckpoint"
    resumed = train_archive(
        archive, expected_sha256=sha, config=config,
        checkpoint=CheckpointOptions(resume_from=prefix,
                                     resume_sha256=report["checkpoint_receipts"][0]["sha256"],
                                     final_path=output),
    )
    assert resumed["new_steps"] == 0
    assert resumed["trained_model_sha256"] == report["trained_model_sha256"]
    assert output.read_bytes() == prefix.read_bytes()


@pytest.mark.parametrize("change", [
    {"seed": 44}, {"batch_size": 3}, {"channels": 16},
    {"diffusion_steps": 9}, {"validation_seed": 44},
    {"validation_repeats": 2}, {"steps": 1},
])
def test_resume_rejects_changed_training_contract(trained, change):
    archive, sha, config, prefix, report = trained
    with pytest.raises(ArchiveError):
        train_archive(
            archive, expected_sha256=sha, config=replace(config, **change),
            checkpoint=CheckpointOptions(resume_from=prefix,
                                         resume_sha256=report["checkpoint_receipts"][0]["sha256"]),
        )


def test_resume_rejects_wrong_external_digest(trained):
    archive, sha, config, prefix, _ = trained
    with pytest.raises(ArchiveError, match="SHA256 mismatch"):
        train_archive(archive, expected_sha256=sha, config=config,
                      checkpoint=CheckpointOptions(resume_from=prefix, resume_sha256="0"*64))


@pytest.mark.parametrize("change", [
    lambda m: m.update({"protocol": "wrong"}),
    lambda m: m.update({"archive_sha256": "0"*64}),
    lambda m: m.update({"model_sha256": "0"*64}),
    lambda m: m.update({"initial_model_sha256": "0"*64}),
    lambda m: m.update({"cpu_rng_sha256": "0"*64}),
    lambda m: m.update({"private_checkpoint": False}),
    lambda m: m.update({"model_publication_authorized": True}),
    lambda m: m.update({"original_sonumio_1": True}),
    lambda m: m["runtime"].update({"num_threads": -1}),
    lambda m: m["optimizer"].update({"lr": .5}),
    lambda m: m["eligible_split_fingerprints"].update({"train": "0"*64}),
    lambda m: m["sampler"].update({"draws": 0}),
    lambda m: m["progress"].update({"completed_steps": 0}),
    lambda m: m.update({"unknown": 1}),
])
def test_semantic_metadata_tampering_rejected_even_after_rehash(trained, tmp_path, change):
    archive, sha, config, prefix, _ = trained
    blob = mutate_header(prefix.read_bytes(), lambda h: change(h["metadata"]))
    path = tmp_path/"tampered.stcheckpoint"
    path.write_bytes(blob)
    with pytest.raises(ArchiveError):
        train_archive(archive, expected_sha256=sha, config=config,
                      checkpoint=CheckpointOptions(resume_from=path, resume_sha256=digest(blob)))


@pytest.mark.parametrize("selector,change", [
    (lambda n: n.startswith("model/"), lambda a: a.fill(0)),
    (lambda n: n.endswith("/step"), lambda a: a.fill(99)),
    (lambda n: n.endswith("/exp_avg_sq"), lambda a: a.fill(-1)),
    (lambda n: n.endswith("/exp_avg"), lambda a: a.fill(np.inf)),
    (lambda n: n == "rng/cpu", lambda a: a.fill(0)),
])
def test_tensor_tampering_rejected_before_model_mutation(trained, tmp_path, selector, change):
    archive, sha, config, prefix, _ = trained
    blob = mutate_tensor(prefix.read_bytes(), selector, change)
    path = tmp_path/"bad-tensor.stcheckpoint"
    path.write_bytes(blob)
    with CorpusArchiveReader(archive, expected_sha256=sha) as reader:
        ready = audit_archive(reader)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = TinyWaveDenoiser(config.channels)
        optimizer = make_archive_optimizer(model)
        initial = model_fingerprint(model)
        rng_before = torch.get_rng_state().clone()
        with pytest.raises(ArchiveError):
            restore_training_checkpoint(
                path, expected_sha256=digest(blob), config=config, readiness=ready,
                initial_model_sha256=initial, model=model, optimizer=optimizer,
            )
        assert not optimizer.state
        assert model_fingerprint(model) == initial
        assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize("options", [
    {"resume_from": "a.stcheckpoint"}, {"resume_sha256": "a"*64},
    {"resume_from": "a.stcheckpoint", "resume_sha256": "wrong"},
    {"periodic_dir": "."}, {"interval": 1},
    {"periodic_dir": ".", "interval": 0}, {"periodic_dir": ".", "interval": True},
    {"final_path": 12},
])
def test_checkpoint_options_fail_closed(options):
    with pytest.raises(ArchiveError):
        CheckpointOptions(**options)


def test_plan_preflights_existing_paths_and_count(tmp_path):
    model = TinyWaveDenoiser(8)
    options = CheckpointOptions(periodic_dir=tmp_path, interval=2)
    plan = plan_checkpoint_outputs(options, 3, 8, model)
    assert list(plan.periodic) == [4, 6, 8]
    assert plan.final_path is None
    plan.periodic[6].write_bytes(b"keep")
    with pytest.raises(ArchiveError, match="already exists"):
        plan_checkpoint_outputs(options, 3, 8, model)
    assert plan.periodic[6].read_bytes() == b"keep"
    with pytest.raises(ArchiveError, match="Too many"):
        plan_checkpoint_outputs(options, 0, 2*(MAX_PERIODIC_CHECKPOINTS+1), model)


def test_plan_refuses_final_periodic_collision(tmp_path):
    options = CheckpointOptions(periodic_dir=tmp_path, interval=2,
                                final_path=tmp_path/"step-00000002.stcheckpoint")
    with pytest.raises(ArchiveError, match="collide"):
        plan_checkpoint_outputs(options, 0, 2, TinyWaveDenoiser(8))


def test_periodic_saves_and_no_clobber(trained, tmp_path, monkeypatch):
    archive, sha, config, _, _ = trained
    options = CheckpointOptions(periodic_dir=tmp_path, interval=1)
    report = train_archive(archive, expected_sha256=sha, config=config, checkpoint=options)
    assert [r["completed_steps"] for r in report["checkpoint_receipts"]] == [1, 2]
    paths = sorted(tmp_path.glob("*.stcheckpoint"))
    assert len(paths) == 2
    original = [path.read_bytes() for path in paths]
    assert not list(tmp_path.glob("*.partial"))
    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint conflict should fail before training")
    monkeypatch.setattr("research.archive_wave_training.training_step", forbidden)
    with pytest.raises(ArchiveError, match="already exists"):
        train_archive(archive, expected_sha256=sha, config=config, checkpoint=options)
    assert [path.read_bytes() for path in paths] == original


def test_optimizer_contract_rejects_changed_hyperparameters():
    model = TinyWaveDenoiser(8)
    optimizer = make_archive_optimizer(model)
    assert len(require_optimizer_contract(model, optimizer)) == len(list(model.parameters()))
    optimizer.param_groups[0]["lr"] = .1
    with pytest.raises(ArchiveError, match="settings changed"):
        require_optimizer_contract(model, optimizer)


def test_runtime_identity_is_explicit_cpu():
    identity = runtime_identity()
    assert identity["device"] == "cpu"
    assert identity["default_dtype"] == "float32"
    assert identity["num_threads"] > 0


def test_cli_checkpoint_receipt_and_no_private_paths(trained, tmp_path, capsys):
    archive, sha, config, prefix, report = trained
    target = tmp_path/"cli.stcheckpoint"
    assert main([
        str(archive), "--expected-sha256", sha, "--steps", str(config.steps),
        "--batch-size", str(config.batch_size), "--channels", str(config.channels),
        "--diffusion-steps", str(config.diffusion_steps),
        "--validation-repeats", str(config.validation_repeats),
        "--validation-batch-size", str(config.validation_batch_size),
        "--resume", str(prefix), "--resume-sha256", report["checkpoint_receipts"][0]["sha256"],
        "--checkpoint-out", str(target),
    ]) == 0
    output = capsys.readouterr().out
    receipt, = json.loads(output)["checkpoint_receipts"]
    assert receipt["filename"] == target.name
    assert str(tmp_path) not in output
    assert target.read_bytes() == prefix.read_bytes()
