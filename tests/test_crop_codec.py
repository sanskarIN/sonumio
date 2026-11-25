"""Crop codec shape, bit-packet integrity, differentiability and paired study."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

torch=pytest.importorskip("torch")

from research.crop_codec_packet import (
    CropCodecError,HEADER,DIGEST_BYTES,BITS,code_shape,signed_limit,pack_codes,unpack_codes,
    packet_bytes,payload_rate_report,
)
from research.crop_codec import CropAudioCodec,quantize_latent,reconstruction_metrics,codec_loss
from research.crop_codec_study import RateDistortionPlan,run_study,summarize


@pytest.mark.parametrize("frames",[2048,8192])
@pytest.mark.parametrize("bits",BITS)
def test_packet_roundtrip_exact_and_immutable(frames,bits):
    limit=signed_limit(bits)
    codes=np.random.default_rng(13).integers(-limit,limit+1,code_shape(frames),dtype=np.int16)
    packed=pack_codes(codes,frames=frames,bits=bits)
    restored=unpack_codes(packed)
    np.testing.assert_array_equal(restored.codes,codes)
    assert restored.frames==frames and restored.bits==bits
    assert restored.bytes_written==packet_bytes(frames,bits)
    assert restored.sha256==hashlib.sha256(packed).hexdigest()
    assert pack_codes(restored.codes,frames=frames,bits=bits)==packed
    with pytest.raises(ValueError):
        restored.codes.setflags(write=True)


@pytest.mark.parametrize("mutation",["magic","frames","channels","bits","count","payload","checksum","truncate","trailing"])
def test_packet_header_payload_and_digest_fail_closed(mutation):
    raw=bytearray(pack_codes(np.zeros(code_shape(2048),dtype=np.int16),frames=2048,bits=4))
    if mutation=="magic": raw[0]^=1
    elif mutation=="frames": raw[8]^=1
    elif mutation=="channels": raw[10]^=1
    elif mutation=="bits": raw[11]=3
    elif mutation=="count": raw[12]^=1
    elif mutation=="payload": raw[HEADER.size]^=1
    elif mutation=="checksum": raw[-1]^=1
    elif mutation=="truncate": raw.pop()
    else: raw.append(0)
    with pytest.raises(CropCodecError):
        unpack_codes(bytes(raw))


def test_reserved_word_rejected_even_after_rehash():
    packet=pack_codes(np.zeros(code_shape(2048),dtype=np.int16),frames=2048,bits=4)
    raw=bytearray(packet[:-DIGEST_BYTES])
    raw[HEADER.size]|=0x0f
    changed=bytes(raw)+hashlib.sha256(raw).digest()
    with pytest.raises(CropCodecError,match="Reserved"):
        unpack_codes(changed)


def test_quantizer_exact_grid_and_ste_gradient():
    x=torch.tensor([-1.,-.4,0.,.4,1.],requires_grad=True)
    hard,codes=quantize_latent(x,4)
    straight,other=quantize_latent(x,4,straight_through=True)
    assert torch.equal(codes,other)
    assert torch.equal(hard,straight)
    assert codes.tolist()==[-7,-3,0,3,7]
    assert not hard.requires_grad
    straight.sum().backward()
    assert torch.equal(x.grad,torch.ones_like(x))


@pytest.mark.parametrize("frames",[2048,8192])
def test_codec_geometry_and_packet_decode(frames):
    model=CropAudioCodec(frames)
    source=torch.rand(2,1,frames)*.4-.2
    model.eval()
    latent=model.encode(source)
    assert latent.shape==(2,*code_shape(frames))
    for bits in (2,4,8):
        direct=model(source,bits)
        packets=model.encode_packets(source,bits)
        pre,restored=model.decode_packets(packets)
        torch.testing.assert_close(direct,restored,rtol=0,atol=0)
        assert pre.shape==source.shape
        assert restored.abs().max()<=1
        assert all(len(packet)==packet_bytes(frames,bits) for packet in packets)


def test_codec_loss_gradients_and_exact_identity():
    model=CropAudioCodec()
    source=torch.rand(2,1,2048)*.4-.2
    restored=model(source,4)
    loss=codec_loss(source,restored,frames=2048)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad)>0 for p in model.encoder.parameters())
    metrics=reconstruction_metrics(source,source,frames=2048)
    for name,value in metrics.items():
        if name!="snr_db":
            assert torch.equal(value,torch.zeros_like(value))


@pytest.mark.parametrize("shape",[(1,1,2047),(1,2,2048),(17,1,2048),(0,1,2048)])
def test_invalid_wave_shape(shape):
    with pytest.raises(CropCodecError):
        CropAudioCodec().encode(torch.zeros(shape))


def test_packet_batch_rejects_mixed_precision_and_length():
    model=CropAudioCodec()
    source=torch.zeros(1,1,2048)
    q4=model.encode_packets(source,4)[0]
    q8=model.encode_packets(source,8)[0]
    with pytest.raises(CropCodecError):
        model.decode_packets([q4,q8])
    with pytest.raises(CropCodecError):
        CropAudioCodec(8192).decode_packets([q4])


def test_rate_report_distinguishes_actual_bytes_and_nominal_payload():
    report=payload_rate_report(2048,4)
    assert report["latent_scalars"]==512
    assert report["nominal_payload_bits"]==2048
    assert report["fixed_width_packet_bytes"]==256+HEADER.size+DIGEST_BYTES
    assert report["packet_bits_per_input_sample"]>1
    assert report["scalar_compression_ratio"]==4
    assert not report["entropy_coding_used"]


@pytest.mark.parametrize("settings",[
    {"model_seeds":(1,)},{"model_seeds":(1,1)},{"model_seeds":(True,2)},
    {"training_seeds":(101,)},{"precision_bits":(3,)},{"precision_bits":(4,4)},
    {"updates":0},{"batch_size":17},{"updates":500,"batch_size":16,"model_seeds":tuple(range(8))},
])
def test_study_plan_bounds(settings):
    with pytest.raises(CropCodecError):
        replace(RateDistortionPlan(),**settings)


@pytest.fixture(scope="module")
def study():
    plan=RateDistortionPlan(model_seeds=(17,29),training_seeds=(0,),validation_seeds=(101,),
                            precision_bits=(4,8),updates=1,batch_size=2)
    rng=torch.get_rng_state().clone()
    result=run_study(plan)
    assert torch.equal(rng,torch.get_rng_state())
    return plan,result


def test_paired_rate_distortion_real_training(study):
    plan,report=study
    assert report["data"]["training_clips"]==report["data"]["validation_clips"]==6
    assert report["training_examples_across_arms"]==12
    assert not report["data"]["exact_cross_split_clip_overlap"]
    for run in report["runs"]:
        assert tuple(run["arms"])==("float","q4","q8")
        for name,arm in run["arms"].items():
            assert arm["initialization"]["model_sha256"]==run["initial_model_sha256"]
            assert arm["trained"]["model_sha256"]!=run["initial_model_sha256"]
            assert len(arm["trained"]["rows"])==6
            assert arm["trained"]["packet_roundtrip_verified"]==(name!="float")
    assert report["seed_summaries"]["float"]["waveform_l1"]["delta_from_float"]["mean"]==0
    assert not report["audio_or_weights_written"]
    assert not report["perceptual_quality_established"]
    json.dumps(report,allow_nan=False)


def test_study_reproducible(study):
    plan,report=study
    assert run_study(plan)==report


def test_summary_rejects_seed_drift(study):
    import copy
    plan,report=study
    runs=copy.deepcopy(report["runs"])
    runs[0]["seed"]=99
    with pytest.raises(CropCodecError):
        summarize(runs,plan)


def test_seeded_crop_source_no_exact_overlap_and_preserves_rng():
    from research.crop_source_data import dataset
    from research.wave_diffusion import CATEGORIES
    before = torch.get_rng_state().clone()
    train, labels, pin = dataset((0, 1, 2))
    validation, _, other = dataset((101, 102))
    replay, replay_labels, again = dataset((0, 1, 2))
    assert torch.equal(train, replay) and torch.equal(labels, replay_labels)
    assert torch.equal(before, torch.get_rng_state())
    assert pin == again and pin != other
    assert labels.tolist() == [i for i in range(len(CATEGORIES)) for _ in range(3)]
    hashes = [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in train]
    heldout = [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in validation]
    assert len(set(hashes)) == len(hashes)
    assert len(set(heldout)) == len(heldout)
    assert not set(hashes) & set(heldout)


@pytest.mark.parametrize("category",["rain","thunder","wind","forest","ocean","ambient"])
def test_event_aware_crop_has_active_signal_and_valid_offset(category):
    from research.crop_source_data import render_crop, WINDOWS, MIN_CENTERED_RMS
    for seed in (0,1,2,101,102):
        wave, offset = render_crop(category,seed)
        assert WINDOWS[category][0] <= offset <= WINDOWS[category][1]
        assert wave.shape == (2048,)
        assert np.std(wave.astype(np.float64)) >= MIN_CENTERED_RMS
        replay, other = render_crop(category,seed)
        np.testing.assert_array_equal(wave,replay)
        assert offset == other
