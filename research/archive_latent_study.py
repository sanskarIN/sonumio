"""Private reviewed-corpus crop latent velocity diffusion research.

Bounded group-balanced codec and denoiser training, a frozen codec, train-only
streamed calibration, aggregate held-out validation and in-memory generation.
No audio or identifiers are exported; an explicit local codec handoff is optional.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from research.archive_latent_access import AuditedCropAccess, counter_generator, hash_tensor
from research.archive_latent_validation import validate_latent
from research.archive_latent_handoff import save_codec_stage, load_codec_stage
from research.archive_latent_calibration import fit_archive_calibration, MODES
from research.archive_readiness import audit_archive
from research.archive_sampler import CounterGroupSampler
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError, canonical_json, require_sha256
from research.crop_codec import CropAudioCodec, codec_loss
from research.crop_codec_packet import BITS, payload_rate_report
from research.crop_latent import (CropLatentDenoiser, encode_native,
                                  reverse_indices, velocity_loss)
from research.crop_latent_study import _generated
from research.wave_diffusion import CLIP_FRAMES, CATEGORIES, DiffusionSchedule
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-private-archive-latent-study-v1"
MAX_TRAIN_EXAMPLES = 32_000
MAX_VALIDATION_PREDICTIONS = 300_000
MAX_GENERATION_FORWARDS = 10_000


@dataclass(frozen=True, slots=True)
class ArchiveLatentPlan:
    seed: int = 17
    codec_steps: int = 32
    diffusion_updates: int = 64
    batch_size: int = 4
    bits: int | None = 8
    width: int = 32
    schedule_steps: int = 32
    reverse_steps: int = 8
    validation_seed: int = 301
    validation_repeats: int = 2
    validation_batch_size: int = 8
    generation_seeds: tuple[int, ...] = (701,709)
    calibration_mode: str = "group_balanced"

    def __post_init__(self):
        for value, lo, hi, name in (
            (self.seed,0,2**32-1,"seed"),(self.codec_steps,1,500,"codec steps"),
            (self.diffusion_updates,1,2000,"diffusion updates"),(self.batch_size,1,16,"batch size"),
            (self.schedule_steps,8,256,"schedule steps"),(self.validation_seed,0,2**32-1,"validation seed"),
            (self.validation_repeats,1,4,"validation repeats"),(self.validation_batch_size,1,16,"validation batch size"),
        ):
            if type(value) is not int or not lo <= value <= hi:
                raise ArchiveError("Invalid private latent " + name)
        if self.bits is not None and (type(self.bits) is not int or self.bits not in BITS):
            raise ArchiveError("Invalid private latent precision")
        if type(self.width) is not int or self.width not in (16,32,48,64):
            raise ArchiveError("Invalid private latent model width")
        if type(self.calibration_mode) is not str or self.calibration_mode not in MODES:
            raise ArchiveError("Invalid private latent calibration mode")
        reverse_indices(self.schedule_steps,self.reverse_steps)
        if (type(self.generation_seeds) is not tuple or not 2 <= len(self.generation_seeds) <= 8
                or any(type(x) is not int or not 0 <= x < 2**32 for x in self.generation_seeds)
                or len(set(self.generation_seeds)) != len(self.generation_seeds)):
            raise ArchiveError("Invalid private latent generation seeds")
        if self.training_examples > MAX_TRAIN_EXAMPLES or self.generation_forwards > MAX_GENERATION_FORWARDS:
            raise ArchiveError("Private latent training or generation workload exceeds budget")

    @property
    def training_examples(self):
        return (self.codec_steps+self.diffusion_updates)*self.batch_size

    @property
    def generation_forwards(self):
        return len(self.generation_seeds)*len(CATEGORIES)*self.reverse_steps*4

    def validation_predictions(self, count):
        if type(count) is not int or not 1 <= count <= 4096:
            raise ArchiveError("Invalid private latent validation clip count")
        work = count*3*self.validation_repeats*(len(CATEGORIES)+1)
        if work > MAX_VALIDATION_PREDICTIONS:
            raise ArchiveError("Private latent validation workload exceeds budget")
        return work


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=.001, foreach=False, fused=False)


def _loss_report(losses, sampler, trace):
    if not losses or not all(math.isfinite(x) for x in losses):
        raise ArchiveError("Invalid private latent training loss summary")
    return {"steps":len(losses),"first_loss":losses[0],"last_loss":losses[-1],
            "mean_loss":statistics.fmean(losses),"sampler":sampler.describe(),
            "training_trace_sha256":trace.hexdigest()}


def train_codec(access, plan, codec):
    sampler = CounterGroupSampler(access.ready,seed=plan.seed)
    optimizer = _optimizer(codec)
    trace = hashlib.sha256(b"sonumio-private-latent-codec-trace-v1\0")
    losses = []
    codec.train()
    for _ in range(plan.codec_steps):
        indices, wave, _ = access.sample(sampler,plan.batch_size)
        trace.update(indices.astype("<i8",copy=False).tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss = codec_loss(wave,codec(wave,plan.bits),frames=CLIP_FRAMES)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    codec.eval().requires_grad_(False)
    return _loss_report(losses,sampler,trace)


def train_denoiser(access, plan, codec, calibration, model, schedule):
    if codec.training or any(p.requires_grad for p in codec.parameters()):
        raise ArchiveError("Private latent denoising requires frozen codec")
    sampler = CounterGroupSampler(access.ready,seed=plan.seed)
    optimizer = _optimizer(model)
    trace = hashlib.sha256(b"sonumio-private-latent-denoiser-trace-v1\0")
    losses = []
    null_examples = 0
    model.train()
    for step in range(plan.diffusion_updates):
        indices, wave, labels = access.sample(sampler,plan.batch_size)
        with torch.no_grad():
            clean = calibration.normalize(encode_native(codec,wave,plan.bits))
        timesteps = torch.randint(schedule.steps,(len(wave),),generator=counter_generator(plan.seed,b"time",step))
        noise = torch.randn(clean.shape,generator=counter_generator(plan.seed,b"noise",step),dtype=torch.float32)
        dropped = torch.rand(len(wave),generator=counter_generator(plan.seed,b"dropout",step)) < .1
        condition = torch.where(dropped,model.classes,labels)
        null_examples += int(dropped.sum())
        trace.update(indices.astype("<i8",copy=False).tobytes())
        for value,dtype in ((timesteps,"<i8"),(noise,"<f4"),(condition,"<i8")):
            hash_tensor(trace,value,dtype)
        optimizer.zero_grad(set_to_none=True)
        loss = velocity_loss(model,schedule,clean,timesteps,noise,condition)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {**_loss_report(losses,sampler,trace),"null_examples":null_examples}


def run_archive_latent(path, *, expected_sha256, plan=None, codec_out=None,
                       codec_in=None, codec_sha256=None, codec_only=False):
    require_sha256(expected_sha256)
    plan = ArchiveLatentPlan() if plan is None else plan
    if not isinstance(plan,ArchiveLatentPlan):
        raise ArchiveError("Expected ArchiveLatentPlan")
    if (type(codec_only) is not bool or (codec_in is None) != (codec_sha256 is None)
            or (codec_out is not None and codec_in is not None)
            or (codec_only and (codec_out is None or codec_in is not None))):
        raise ArchiveError("Invalid private codec handoff options")
    if codec_sha256 is not None:
        require_sha256(codec_sha256)
    if torch.get_default_dtype() != torch.float32:
        raise ArchiveError("Private latent study requires default float32")
    with CorpusArchiveReader(path,expected_sha256=expected_sha256) as reader:
        ready = audit_archive(reader)
        access = AuditedCropAccess(reader,ready)
        validation_work = plan.validation_predictions(access.count("validation"))
        plan_sha = hashlib.sha256(canonical_json({
            "protocol":PROTOCOL,"configuration":asdict(plan),"archive_sha256":ready.archive_sha256,
            "eligible_train":ready.eligible_fingerprints["train"],
            "eligible_validation":ready.eligible_fingerprints["validation"],
        })).hexdigest()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(plan.seed)
            with torch.device("cpu"):
                codec = CropAudioCodec(CLIP_FRAMES)
            initial_codec = model_state_sha256(codec)
            receipt = None
            if codec_in is None:
                codec_report = train_codec(access,plan,codec)
                trained_codec = model_state_sha256(codec)
                if initial_codec == trained_codec:
                    raise ArchiveError("Private crop codec did not train")
                calibration, calibration_weights = fit_archive_calibration(
                    access,codec,bits=plan.bits,mode=plan.calibration_mode,
                )
                if codec_out is not None:
                    receipt = save_codec_stage(
                        codec_out,access=access,plan=plan,codec=codec,initial_sha256=initial_codec,
                        training=codec_report,calibration=calibration,weighting=calibration_weights,
                    )
            else:
                codec,codec_report,calibration,calibration_weights,receipt = load_codec_stage(
                    codec_in,expected_sha256=codec_sha256,access=access,plan=plan,codec=codec,
                )
                trained_codec = model_state_sha256(codec)
            if calibration.clips != access.count("train"):
                raise ArchiveError("Private latent calibration coverage changed")
            codec_summary = {
                "initial_sha256":initial_codec,"trained_sha256":trained_codec,"training":codec_report,
                "calibration":{**calibration.report(),"weighting":calibration_weights},
                "frozen_during_diffusion":True,
                "rate":payload_rate_report(CLIP_FRAMES,plan.bits) if plan.bits is not None else None,
            }
            handoff = {"mode":"restored" if codec_in is not None else "saved" if codec_out is not None else "none",
                       "receipt":receipt.report() if receipt is not None else None,
                       "codec_optimizer_steps_executed":0 if codec_in is not None else plan.codec_steps,
                       "full_optimizer_resume_supported":False}
            if codec_only:
                reader.report()
                return {"protocol":"sonumio-private-codec-stage-only-v1","codec":codec_summary,
                        "codec_handoff":handoff,"archive_sha256":ready.archive_sha256,
                        "denoiser_steps_executed":0,"checkpoint_written":True,
                        "audio_or_weights_written":True,"model_publication_authorized":False,
                        "original_sonumio_1":False}
            torch.random.default_generator.manual_seed(plan.seed)
            with torch.device("cpu"):
                model = CropLatentDenoiser(CLIP_FRAMES,plan.width)
                schedule = DiffusionSchedule(plan.schedule_steps,profile="cosine_zero_terminal_v1")
            initial_model = model_state_sha256(model)
            training = train_denoiser(access,plan,codec,calibration,model,schedule)
            trained_model = model_state_sha256(model)
            if initial_model == trained_model:
                raise ArchiveError("Private latent denoiser did not train")
            validation = validate_latent(access,codec,model,schedule,calibration,bits=plan.bits,
                                         seed=plan.validation_seed,repeats=plan.validation_repeats,
                                         batch_size=plan.validation_batch_size)
            generated = _generated(plan,codec,model,schedule,calibration)
            if (model_state_sha256(codec) != trained_codec or model_state_sha256(model) != trained_model):
                raise ArchiveError("Private latent frozen state changed during evaluation")
        reader.report()
    return {
        "protocol":PROTOCOL,"configuration":asdict(plan),"study_plan_sha256":plan_sha,
        "archive_sha256":ready.archive_sha256,"eligible_split_fingerprints":ready.eligible_fingerprints.copy(),
        "eligible_training_clips":access.count("train"),"eligible_validation_clips":access.count("validation"),
        "training_examples":plan.training_examples,"validation_prediction_rows":validation_work,
        "generation_forward_calls":plan.generation_forwards,
        "codec":codec_summary,"codec_handoff":handoff,
        "denoiser":{"initial_sha256":initial_model,"trained_sha256":trained_model,"training":training,
                    "objective":"velocity","schedule":"cosine_zero_terminal_v1",
                    "parameter_count":sum(p.numel() for p in model.parameters()),
                    "terminal_alpha_bar":float(schedule.alpha_bars[-1])},
        "validation":validation,"generation":generated,
        "calibration_train_only":True,"latent_corpus_materialized":False,
        "bounded_archive_batches":True,"test_split_scored":False,
        "recording_identifiers_exported":False,"audio_or_weights_written":codec_out is not None,
        "perceptual_quality_established":False,"model_publication_authorized":False,
        "original_sonumio_1":False,
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--expected-sha256",required=True)
    defaults=ArchiveLatentPlan()
    for name in ("seed","codec_steps","diffusion_updates","batch_size","width","schedule_steps",
                 "reverse_steps","validation_seed","validation_repeats","validation_batch_size"):
        parser.add_argument("--"+name.replace("_","-"),type=int,default=getattr(defaults,name))
    parser.add_argument("--bits",choices=("float","2","4","6","8"),default="8")
    parser.add_argument("--generation-seeds",default="701,709")
    parser.add_argument("--calibration-mode",choices=MODES,default=defaults.calibration_mode)
    parser.add_argument("--codec-out",type=Path)
    parser.add_argument("--codec-in",type=Path)
    parser.add_argument("--codec-sha256")
    parser.add_argument("--codec-only",action="store_true")
    args=parser.parse_args(argv)
    try:
        settings={key:getattr(args,key) for key in ArchiveLatentPlan.__dataclass_fields__}
        settings["bits"]=None if args.bits=="float" else int(args.bits)
        parts=args.generation_seeds.split(",")
        if any(not x.isascii() or not x.isdecimal() or len(x)>10 or str(int(x))!=x for x in parts):
            raise ArchiveError("Invalid canonical generation seed list")
        settings["generation_seeds"]=tuple(map(int,parts))
        report=run_archive_latent(args.archive,expected_sha256=args.expected_sha256,
                                  plan=ArchiveLatentPlan(**settings),codec_out=args.codec_out,
                                  codec_in=args.codec_in,codec_sha256=args.codec_sha256,codec_only=args.codec_only)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Private archive latent experiment: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
