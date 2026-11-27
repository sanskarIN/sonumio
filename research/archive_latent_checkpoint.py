"""Private complete-update checkpoints for crop latent velocity training.

Reuse the bounded STCP0001 numeric container, with a distinct metadata
protocol. Restore into a disposable model before accepting any tensors.
Counter RNGs and the sampler trace are replayed from the audited corpus;
no pickle, global RNG restoration, validation crops or identifiers.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from research.archive_checkpoint_format import (
    TensorSpec, encode_checkpoint, decode_checkpoint, MAX_CHECKPOINT_BYTES,
)
from research.archive_checkpoint import _optimizer_arrays
from research.archive_training_contracts import (
    make_archive_optimizer, require_optimizer_contract, optimizer_identity,
    runtime_identity,
)
from research.archive_latent_access import counter_generator, hash_tensor
from research.archive_sampler import CounterGroupSampler
from research.checkpoint_snapshot import read_verified_weights
from research.corpus_archive_format import ArchiveError, canonical_json, require_sha256
from research.crop_codec_packet import code_shape
from research.crop_latent import CropLatentDenoiser
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-private-latent-optimizer-checkpoint-v1"
TRACE_DOMAIN = b"sonumio-private-latent-denoiser-trace-v1\0"
TRAIN_FIELDS = ("seed","codec_steps","batch_size","bits","width",
                "schedule_steps","calibration_mode")


def identity(access,plan,codec,calibration):
    return {
        "protocol":PROTOCOL,"configuration":{key:getattr(plan,key) for key in TRAIN_FIELDS},
        "archive_sha256":access.ready.archive_sha256,
        "eligible_split_fingerprints":access.ready.eligible_fingerprints.copy(),
        "codec_sha256":model_state_sha256(codec),
        "calibration_sha256":calibration.fingerprint(),
        "runtime":runtime_identity(),"optimizer":optimizer_identity(),
        "objective":"velocity","schedule":"cosine_zero_terminal_v1",
        "condition_dropout":.1,"gradient_clip_norm":1.,
    }


def tensor_specs(model):
    if not isinstance(model,CropLatentDenoiser):
        raise ArchiveError("Expected crop latent denoiser")
    specs={}
    for name,tensor in model.state_dict().items():
        if tensor.dtype!=torch.float32 or tensor.device.type!="cpu":
            raise ArchiveError("Latent checkpoint requires CPU float32 tensors")
        specs["model/"+name]=TensorSpec("f32",tuple(tensor.shape))
    for name,param in model.named_parameters():
        for field in ("step","exp_avg","exp_avg_sq"):
            specs["optimizer/"+name+"/"+field]=TensorSpec(
                "f32",() if field=="step" else tuple(param.shape))
    return specs


def counter_inputs(plan,model,labels,step):
    """Same pure RNG inputs for training and audited prefix replay."""
    count=len(labels)
    timesteps=torch.randint(plan.schedule_steps,(count,),
        generator=counter_generator(plan.seed,b"time",step))
    noise=torch.randn((count,*code_shape(model.frames)),
        generator=counter_generator(plan.seed,b"noise",step),dtype=torch.float32)
    dropped=torch.rand(count,generator=counter_generator(plan.seed,b"dropout",step))<.1
    condition=torch.where(dropped,model.classes,labels)
    return timesteps,noise,condition,int(dropped.sum())


def update_trace(trace,indices,timesteps,noise,condition):
    trace.update(indices.astype("<i8",copy=False).tobytes())
    for value,dtype in ((timesteps,"<i8"),(noise,"<f4"),(condition,"<i8")):
        hash_tensor(trace,value,dtype)


def validate_losses(losses,plan):
    if (type(losses) is not list or not 1<=len(losses)<=plan.diffusion_updates
            or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in losses)):
        raise ArchiveError("Invalid latent checkpoint loss history")


@dataclass(slots=True)
class Prefix:
    sampler: CounterGroupSampler
    trace: object
    null_examples: int
    losses: list


def replay_prefix(access,plan,model,losses):
    validate_losses(losses,plan)
    sampler=CounterGroupSampler(access.ready,seed=plan.seed)
    trace=hashlib.sha256(TRACE_DOMAIN)
    null_examples=0
    labels={item.row.index:item.row.label for item in access.ready.eligible["train"]}
    for step in range(len(losses)):
        indices=sampler.draw(plan.batch_size)
        condition=torch.tensor([labels[int(i)] for i in indices],dtype=torch.long)
        times,noise,condition,dropped=counter_inputs(plan,model,condition,step)
        update_trace(trace,indices,times,noise,condition)
        null_examples+=dropped
    return Prefix(sampler,trace,null_examples,list(losses))


def progress_metadata(prefix):
    return {"losses":list(prefix.losses),"completed_steps":len(prefix.losses),
            "sampler":prefix.sampler.describe(),"trace_sha256":prefix.trace.hexdigest(),
            "null_examples":prefix.null_examples}


def destination(path,*,writing):
    path=Path(path)
    if path.suffix!=".stcheckpoint" or not path.parent.is_dir() or path.parent.is_symlink():
        raise ArchiveError("Latent checkpoint requires .stcheckpoint in an existing ordinary directory")
    if writing and os.path.lexists(path):
        raise ArchiveError("Latent checkpoint destination already exists")
    return path


def publish(path,blob):
    target=destination(path,writing=True)
    scratch=None
    pin=hashlib.sha256(blob).hexdigest()
    try:
        with tempfile.NamedTemporaryFile(mode="wb",dir=target.parent,
                prefix=".sonumio-train-",suffix=".partial",delete=False) as output:
            scratch=Path(output.name)
            if output.write(blob)!=len(blob):
                raise ArchiveError("Short latent checkpoint write")
            output.flush()
            os.fsync(output.fileno())
        read_verified_weights(scratch,pin,max_bytes=MAX_CHECKPOINT_BYTES)
        os.link(scratch,target)
    except OSError as exc:
        raise ArchiveError("Atomic latent checkpoint publication failed") from exc
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)
    return pin


def save_latent_checkpoint(path,*,access,plan,codec,calibration,model,optimizer,
                           initial_sha256,prefix):
    expected=replay_prefix(access,plan,model,prefix.losses)
    progress=progress_metadata(prefix)
    if canonical_json(progress)!=canonical_json(progress_metadata(expected)):
        raise ArchiveError("Latent checkpoint training prefix differs")
    trained=model_state_sha256(model)
    if trained==require_sha256(initial_sha256):
        raise ArchiveError("Cannot save unchanged latent model")
    arrays={"model/"+name:tensor.detach().numpy() for name,tensor in model.state_dict().items()}
    arrays.update(_optimizer_arrays(model,optimizer,len(prefix.losses)))
    metadata={"identity":identity(access,plan,codec,calibration),
              "initial_model_sha256":initial_sha256,"trained_model_sha256":trained,
              "progress":progress,"publication_authorized":False}
    blob=encode_checkpoint(metadata,arrays,tensor_specs(model))
    pin=publish(path,blob)
    return {"sha256":pin,"completed_steps":len(prefix.losses),
            "bytes_written":len(blob),"model_sha256":trained}


def restore_latent_checkpoint(path,*,expected_sha256,access,plan,codec,calibration,model):
    """Returns independent model, optimizer and exact replayed progress."""
    pin=require_sha256(expected_sha256)
    blob=read_verified_weights(destination(path,writing=False),pin,max_bytes=MAX_CHECKPOINT_BYTES)
    metadata,arrays=decode_checkpoint(blob,tensor_specs(model))
    if (set(metadata)!={"identity","initial_model_sha256","trained_model_sha256",
                       "progress","publication_authorized"}
            or metadata["publication_authorized"] is not False):
        raise ArchiveError("Unexpected latent checkpoint metadata")
    if canonical_json(metadata["identity"])!=canonical_json(identity(access,plan,codec,calibration)):
        raise ArchiveError("Latent checkpoint trajectory identity differs")
    initial=model_state_sha256(model)
    if require_sha256(metadata["initial_model_sha256"])!=initial:
        raise ArchiveError("Latent checkpoint initialization differs")
    progress=metadata["progress"]
    if type(progress) is not dict or "losses" not in progress:
        raise ArchiveError("Invalid latent checkpoint progress")
    prefix=replay_prefix(access,plan,model,progress["losses"])
    if canonical_json(progress)!=canonical_json(progress_metadata(prefix)):
        raise ArchiveError("Latent checkpoint prefix replay differs")
    candidate=copy.deepcopy(model)
    optimizer=make_archive_optimizer(candidate)
    parameters=require_optimizer_contract(candidate,optimizer)
    replacements={name:torch.from_numpy(arrays["model/"+name].copy())
                  for name in candidate.state_dict()}
    restored={}
    for name,param in parameters:
        key="optimizer/"+name+"/"
        if float(arrays[key+"step"])!=len(prefix.losses):
            raise ArchiveError("Latent optimizer step disagrees with progress")
        if np.any(arrays[key+"exp_avg_sq"]<0):
            raise ArchiveError("Latent optimizer second moment is negative")
        restored[param]={field:torch.from_numpy(arrays[key+field].copy())
                         for field in ("step","exp_avg","exp_avg_sq")}
    candidate.load_state_dict(replacements,strict=True)
    trained=model_state_sha256(candidate)
    if trained!=require_sha256(metadata["trained_model_sha256"]) or trained==initial:
        raise ArchiveError("Latent restored model digest differs")
    optimizer.state.update(restored)
    return candidate,optimizer,prefix
