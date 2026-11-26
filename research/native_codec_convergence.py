"""Phase-sensitive convergence study for the native 24 kHz neural codec.

The first pilot reduced magnitude error while failing the silence control.
Here the architecture and initialization are fixed and the objective is
ablated. Measure per-clip energy-normalized errors rather than equating
small magnitude loss with successful reconstruction.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import statistics

import torch

from research.native_codec import (
    Native24kCodec, FRAMES, SAMPLE_RATE, FFT_SIZES, HIGHBAND_HZ,
    MAX_BATCH, native_metrics, validate_wave,
)
from research.native_codec_study import (
    dataset, batch_plan, lowband_control, _parse_seeds, _seeds,
    SOURCE_PROTOCOL, mean_metrics,
)
from research.wave_diffusion import CATEGORIES
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-native24k-phase-convergence-v1"
ARMS = ("pilot_highband", "balanced_wave", "balanced_phase")
POWER_FLOOR = 1e-6
MAX_UPDATES = 256
MAX_TRAIN_EXAMPLES = 12_288


@dataclass(frozen=True, slots=True)
class ConvergencePlan:
    model_seeds: tuple[int, ...] = (17, 29, 41)
    training_seeds: tuple[int, ...] = (0, 1, 2)
    validation_seeds: tuple[int, ...] = (201, 202)
    updates: int = 128
    batch_size: int = 4
    checkpoints: tuple[int, ...] = (32, 64, 128)

    def __post_init__(self):
        _seeds(self.model_seeds, 2, 4)
        _seeds(self.training_seeds, 1, 6)
        _seeds(self.validation_seeds, 1, 4)
        if set(self.training_seeds) & set(self.validation_seeds):
            raise ValueError("Training and development source seeds overlap")
        if (type(self.updates) is not int or not 1 <= self.updates <= MAX_UPDATES
                or type(self.batch_size) is not int or not 1 <= self.batch_size <= MAX_BATCH
                or self.train_examples > MAX_TRAIN_EXAMPLES):
            raise ValueError("Convergence training budget exceeded")
        if (type(self.checkpoints) is not tuple or not 1 <= len(self.checkpoints) <= 8
                or any(type(x) is not int or not 1 <= x <= self.updates for x in self.checkpoints)
                or tuple(sorted(set(self.checkpoints))) != self.checkpoints
                or self.checkpoints[-1] != self.updates):
            raise ValueError("Expected increasing evaluation checkpoints ending at final update")

    @property
    def train_examples(self):
        return len(self.model_seeds)*len(ARMS)*self.updates*self.batch_size


def phase_components(source, restored):
    """Complex STFT error normalized by source energy, retaining phase.

    The high-band denominator uses total source energy, avoiding explosion
    for sources with no high frequencies. Near-silent input has a power
    floor. No square root of a zero residual appears in the objective.
    """
    validate_wave(source)
    validate_wave(restored, normalized=False)
    if source.shape != restored.shape:
        raise ValueError("Convergence batch geometry mismatch")
    a, b = source[:, 0], restored[:, 0]
    power = a.square().mean(1)
    waveform = (a-b).square().mean(1)/power.clamp_min(POWER_FLOOR)
    # Scale the DC residual by total power as well.
    dc = (a-b).mean(1).square()/power.clamp_min(POWER_FLOOR)
    full, high = [], []
    for fft in FFT_SIZES:
        window = torch.hann_window(fft, dtype=torch.float32)
        x = torch.stft(a, fft, hop_length=fft//4, window=window,
                       center=False, return_complex=True)
        y = torch.stft(b, fft, hop_length=fft//4, window=window,
                       center=False, return_complex=True)
        weights = torch.ones(fft//2+1, dtype=torch.float32)
        weights[1:-1] = 2
        weights = weights.view(1,-1,1)
        power_x = (x.real.square()+x.imag.square())*weights
        residual = ((x-y).real.square()+(x-y).imag.square())*weights
        # Relative energy floor based on STFT geometry and Hann energy.
        floor = POWER_FLOOR*fft*float(window.square().sum())*x.shape[-1]
        denom = power_x.sum((1,2)).clamp_min(floor)
        mask = torch.fft.rfftfreq(fft, d=1/SAMPLE_RATE)>=HIGHBAND_HZ
        full.append(residual.sum((1,2))/denom)
        high.append(residual[:,mask].sum((1,2))/denom)
    result = {"normalized_waveform_mse": waveform, "normalized_dc_error": dc,
              "complex_spectral_error": torch.stack(full).mean(0),
              "complex_highband_error": torch.stack(high).mean(0)}
    if any(not torch.isfinite(x).all() or (x<0).any() for x in result.values()):
        raise ValueError("Nonfinite or negative phase metric")
    return result


def objective(source, restored, arm):
    if arm not in ARMS:
        raise ValueError("Unknown convergence arm")
    v = native_metrics(source, restored)
    shared = v["waveform_mae"]+.5*v["log_spectral_mae"]+.15*v["spectral_convergence"]
    if arm == "pilot_highband":
        return (shared+.5*v["highband_magnitude_error"]).mean()
    p = phase_components(source, restored)
    balanced = shared+.5*p["normalized_waveform_mse"]+.1*p["normalized_dc_error"]
    if arm == "balanced_phase":
        balanced = balanced+.25*p["complex_spectral_error"]+.25*p["complex_highband_error"]
    loss = balanced.mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite convergence loss")
    return loss


@torch.no_grad()
def score(wave, labels, model=None, *, control=None):
    if (model is None)==(control is None) or len(wave)!=len(labels) or not len(wave):
        raise ValueError("Invalid convergence evaluation panel")
    before = model_state_sha256(model) if model is not None else None
    mode = model.training if model is not None else None
    if model is not None:
        model.eval()
    rows=[]
    try:
        for i,label in enumerate(labels):
            if type(label) is not int or not 0<=label<len(CATEGORIES):
                raise ValueError("Invalid evaluation category")
            source=wave[i:i+1]
            if model is not None:
                latent=model.encode(source)
                raw=model.decode_raw(latent)
                restored=raw.tanh()
            else:
                latent=None
                restored=raw=control(source)
            values={**native_metrics(source,restored),**phase_components(source,restored)}
            row={k:float(v[0]) for k,v in values.items()}
            # Strict paired silence comparison uses exact waveform MSE.
            silence_mse=float(source.double().square().mean())
            error=float((source.double()-restored.double()).square().mean())
            row["silence_mse_margin"]=silence_mse-error
            row["relative_mse_to_silence"]=error/max(silence_mse,1e-15)
            row["pre_output_peak"]=float(raw.abs().max())
            if latent is not None:
                row["latent_rms"]=float(latent.square().mean().sqrt())
                row["latent_channel_std"]=float(latent.std(dim=-1,unbiased=False).mean())
                row["latent_near_bound_fraction"]=float((latent.abs()>=.999).float().mean())
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError("Nonfinite convergence score")
            rows.append((CATEGORIES[label],row))
        if model is not None and model_state_sha256(model)!=before:
            raise ValueError("Model mutated during convergence evaluation")
    finally:
        if model is not None:
            model.train(mode)
    data=[row for _,row in rows]
    return {"model_sha256":before,"clips":len(rows),"mean":mean_metrics(data),
            "silence_wins":sum(row["silence_mse_margin"]>0 for row in data),
            "silence_ties":sum(row["silence_mse_margin"]==0 for row in data),
            "positive_snr_clips":sum(row["snr_db"]>0 for row in data),
            "median_relative_mse_to_silence":statistics.median(row["relative_mse_to_silence"] for row in data),
            "by_category":{c:mean_metrics([row for name,row in rows if name==c])
                           for c in CATEGORIES}}


def train_seed(plan, seed, train, development, labels):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            initial=Native24kCodec()
        identity=model_state_sha256(initial)
        batches,trace=batch_plan(plan,seed,len(train))
        baseline=score(development,labels,initial)
        arms={}
        for arm in ARMS:
            model=copy.deepcopy(initial)
            if model_state_sha256(model)!=identity:
                raise ValueError("Convergence arms do not share initialization")
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False,fused=False)
            reports=[]
            losses=[]
            for step,ids in enumerate(batches,1):
                model.train()
                source=train[ids]
                optimizer.zero_grad(set_to_none=True)
                loss=objective(source,model(source),arm)
                if not torch.isfinite(loss):
                    raise ValueError("Invalid training loss")
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if step in plan.checkpoints:
                    result=score(development,labels,model)
                    reports.append({"step":step,"mean_training_loss_since_previous":statistics.fmean(losses),
                                    "last_preclip_gradient_norm":float(grad),"evaluation":result})
                    losses=[]
            if reports[-1]["evaluation"]["model_sha256"]==identity:
                raise ValueError("Codec did not update")
            arms[arm]=reports
        return {"seed":seed,"initial_model_sha256":identity,"batch_trace_sha256":trace,
                "initial":baseline,"arms":arms}


def summarize(runs, plan, silence):
    if len(runs)!=len(plan.model_seeds):
        raise ValueError("Incomplete convergence seed panel")
    summary={}
    for arm in ARMS:
        panels=[]
        for run,seed in zip(runs,plan.model_seeds,strict=True):
            if run["seed"]!=seed or tuple(run["arms"])!=ARMS:
                raise ValueError("Convergence panel identity changed")
            if tuple(r["step"] for r in run["arms"][arm])!=plan.checkpoints:
                raise ValueError("Evaluation checkpoint drift")
            panels.append(run["arms"][arm][-1]["evaluation"])
        means=[p["mean"]["snr_db"] for p in panels]
        wins=[p["silence_wins"] for p in panels]
        high=[p["mean"]["complex_highband_error"] for p in panels]
        # Prespecified conservative development gate, not a release gate.
        gate=(all(x>0 for x in means)
              and all(w*3>=p["clips"]*2 for w,p in zip(wins,panels,strict=True))
              and all(x<silence["mean"]["complex_highband_error"] for x in high))
        summary[arm]={"mean_snr_db":statistics.fmean(means),"seed_snr_db":means,
                      "seed_silence_wins":wins,"seed_complex_highband_error":high,
                      "mean_complex_highband_error":statistics.fmean(high),
                      "development_reconstruction_gate_passed":gate}
    return summary


def run_study(plan=None):
    plan=ConvergencePlan() if plan is None else plan
    if not isinstance(plan,ConvergencePlan):
        raise ValueError("Expected ConvergencePlan")
    train,_,a,train_hash=dataset(plan.training_seeds)
    development,labels,b,dev_hash=dataset(plan.validation_seeds)
    if set(a)&set(b):
        raise ValueError("Exact PCM overlap in convergence source")
    silence=score(development,labels,control=torch.zeros_like)
    lowband=score(development,labels,control=lowband_control)
    runs=[train_seed(plan,seed,train,development,labels) for seed in plan.model_seeds]
    return {"protocol":PROTOCOL,"source_protocol":SOURCE_PROTOCOL,"plan":asdict(plan),
            "sample_rate":SAMPLE_RATE,"clip_frames":FRAMES,
            "training_sha256":train_hash,"development_sha256":dev_hash,
            "training_clips":len(train),"development_clips":len(development),
            "train_examples":plan.train_examples,"silence_control":silence,
            "lowband_control":lowband,"runs":runs,"summary":summarize(runs,plan,silence),
            "matched_architecture_initialization_and_batches":True,
            "exact_cross_split_pcm_overlap":False,"source_is_synthetic":True,
            "development_panel_used_for_interpretation":True,
            "independent_test_scored":False,"perceptual_quality_established":False,
            "model_release_authorized":False,"files_written":0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    default=ConvergencePlan()
    for name in ("model_seeds","training_seeds","validation_seeds","checkpoints"):
        parser.add_argument("--"+name.replace("_","-"),default=",".join(map(str,getattr(default,name))))
    parser.add_argument("--updates",type=int,default=default.updates)
    parser.add_argument("--batch-size",type=int,default=default.batch_size)
    args=parser.parse_args(argv)
    try:
        plan=ConvergencePlan(_parse_seeds(args.model_seeds),_parse_seeds(args.training_seeds),
                             _parse_seeds(args.validation_seeds),args.updates,args.batch_size,
                             _parse_seeds(args.checkpoints))
        report=run_study(plan)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Native convergence study: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
