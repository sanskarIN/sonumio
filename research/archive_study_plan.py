"""Prespecified, bounded experimental design for paired archive studies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import struct

import numpy as np
import torch

from research.corpus_archive_format import ArchiveError, canonical_json, require_sha256
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES, DiffusionSchedule

PROTOCOL = "sonumio-paired-archive-label-study-v1"
ARMS = ("matched", "random_label")
MAX_TRAIN_EXAMPLES = 100_000
MAX_ASSAY_PREDICTIONS = 2_000_000


def strict_int(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ArchiveError(f"Invalid study {name}")
    return value


@dataclass(frozen=True, slots=True)
class StudyPlan:
    seeds: tuple[int, ...] = (17, 29, 41)
    steps: int = 64
    batch_size: int = 8
    channels: int = 16
    diffusion_steps: int = 32
    diffusion_profile: str = "cosine_zero_terminal_v1"
    drop_condition: float = .1
    assay_seed: int = 301
    assay_repeats: int = 2
    assay_batch_size: int = 8

    def __post_init__(self):
        if (type(self.seeds) is not tuple or not 2 <= len(self.seeds) <= 8
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ArchiveError("Use 2..8 distinct explicit integer study seeds")
        for name, lo, hi in (
            ("steps", 1, 2000), ("batch_size", 1, 32), ("channels", 8, 64),
            ("diffusion_steps", 8, 1000), ("assay_seed", 0, 2**32-1),
            ("assay_repeats", 1, 4), ("assay_batch_size", 1, 16),
        ):
            strict_int(getattr(self,name),lo,hi,name)
        if self.channels % 8 or self.train_examples > MAX_TRAIN_EXAMPLES:
            raise ArchiveError("Study training workload exceeds configured budget")
        if (type(self.diffusion_profile) is not str
                or self.diffusion_profile not in DiffusionSchedule.PROFILES):
            raise ArchiveError("Unknown study diffusion profile")
        if (type(self.drop_condition) not in (int,float)
                or not math.isfinite(self.drop_condition)
                or not 0 <= self.drop_condition <= .5):
            raise ArchiveError("Invalid study conditioning dropout")

    @property
    def train_examples(self):
        return len(self.seeds)*len(ARMS)*self.steps*self.batch_size

    def assay_predictions(self, validation_rows):
        strict_int(validation_rows,1,4096,"validation rows")
        # Three models (initial + both arms), three noise levels, six
        # categories plus null. Includes the entire declared seed list.
        count = len(self.seeds)*3*validation_rows*3*self.assay_repeats*(len(CATEGORIES)+1)
        if count > MAX_ASSAY_PREDICTIONS:
            raise ArchiveError("Study evaluation workload exceeds configured budget")
        return count

    def as_dict(self):
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result

    def fingerprint(self, archive_sha256, eligible_train, eligible_validation):
        return hashlib.sha256(canonical_json({
            "protocol": PROTOCOL, "plan": self.as_dict(),
            "archive_sha256": require_sha256(archive_sha256),
            "eligible_train": require_sha256(eligible_train),
            "eligible_validation": require_sha256(eligible_validation),
            "arms": list(ARMS),
            "control": "independent_uniform_category_with_shared_null_dropout",
        })).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class CorruptionBatch:
    steps: torch.Tensor
    noise: torch.Tensor
    dropped: torch.Tensor
    matched: torch.Tensor
    random_label: torch.Tensor


def _generator(seed, step, domain):
    material = hashlib.sha256(
        b"sonumio-paired-label-corruption-v1\0"+domain+struct.pack("<II",seed,step)
    ).digest()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(material[:8],"little") & ((1<<63)-1))


def corruption_batch(plan: StudyPlan, seed: int, step: int, labels: torch.Tensor) -> CorruptionBatch:
    """Draw once; share corruption and dropout exactly across treatment arms.

    The random-label control is independent uniform with replacement, NOT a
    derangement. Accidental matches are expected and explicitly counted.
    Separate counter domains mean changes in category drawing never change
    diffusion noise or dropout.
    """
    if not isinstance(plan,StudyPlan) or seed not in plan.seeds or type(seed) is not int:
        raise ArchiveError("Seed not in prespecified study plan")
    strict_int(step,0,plan.steps-1,"optimizer step")
    if (not isinstance(labels,torch.Tensor) or labels.device.type!="cpu"
            or labels.dtype!=torch.long or labels.shape!=(plan.batch_size,)
            or not ((labels>=0)&(labels<len(CATEGORIES))).all()):
        raise ArchiveError("Invalid study training labels")
    steps = torch.randint(plan.diffusion_steps,(len(labels),),generator=_generator(seed,step,b"time"))
    noise = torch.randn((len(labels),1,CLIP_FRAMES),generator=_generator(seed,step,b"noise"),
                        dtype=torch.float32)
    dropped = torch.rand(len(labels),generator=_generator(seed,step,b"drop")) < plan.drop_condition
    random = torch.randint(len(CATEGORIES),(len(labels),),generator=_generator(seed,step,b"label"))
    null = torch.full_like(labels,len(CATEGORIES))
    return CorruptionBatch(steps,noise,dropped,torch.where(dropped,null,labels),
                           torch.where(dropped,null,random))


def update_trace(trace, indices, corruption: CorruptionBatch):
    """Hash actual ordered inputs, including label-control interventions."""
    for values,dtype in (
        (indices,"<i8"),(corruption.steps.numpy(),"<i8"),
        (corruption.noise.numpy(),"<f4"),(corruption.dropped.numpy(),"u1"),
        (corruption.matched.numpy(),"<i8"),(corruption.random_label.numpy(),"<i8"),
    ):
        raw=np.asarray(values,dtype=dtype,order="C").tobytes()
        trace.update(len(raw).to_bytes(8,"little"))
        trace.update(raw)


def parse_seeds(raw):
    if type(raw) is not str or len(raw)>100:
        raise ArchiveError("Expected explicit comma-separated seed list")
    fields=raw.split(",")
    if any(not x.isascii() or not x.isdecimal() or len(x)>10 or str(int(x))!=x for x in fields):
        raise ArchiveError("Seed list must use canonical decimal integers")
    return tuple(int(x) for x in fields)
