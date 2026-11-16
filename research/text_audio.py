"""Original *experimental* byte-text-conditioned waveform diffusion research.

A small from-scratch Transformer text encoder controls a 1D DDPM denoiser.
English/Hindi text is tokenized as UTF-8 bytes without a fixed sound-class
lookup during model inference. Synthetic training pairs still cover only six
DSP source categories and their mixtures. This is not general text-to-audio,
not a useful-quality audio model, and NOT Sonumio-1.

Only run locally using rights-reviewed audio or synthetic DSP data. No model
weights may be published without separate explicit authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
import unicodedata

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from research.wave_diffusion import (
    CATEGORIES, CLIP_FRAMES, DEFAULT_DIFFUSION_STEPS, SAMPLE_RATE,
    ConditionedResidual, DiffusionSchedule, _downsample_24k_to_8k,
)
from sonumio.audio import encode_wav
from sonumio.engines.procedural import generate as dsp_generate
from sonumio.validation import validate_request

PAD, START, END, BYTE_OFFSET = 0, 1, 2, 3
BYTE_VOCAB = 259
MAX_BYTES = 128
MAX_TOKENS = MAX_BYTES + 2
CHECKPOINT_FORMAT = 'sonumio-text-conditioned-diffusion-experiment-1'
EN_LABELS = ('rain', 'thunder', 'wind', 'forest birds', 'ocean waves', 'ambient synth')
HI_LABELS = ('बारिश', 'गरज', 'हवा', 'जंगल में चिड़िया', 'समुद्र की लहर', 'मधुर धुन')
SYNTHETIC_RECIPE = 'active-window-dsp-v2'
SYNTHETIC_TRAIN_SEEDS = (0, 1, 2, 3)
SYNTHETIC_VALIDATION_SEEDS = (101, 102)
SYNTHETIC_COMBINATIONS = tuple(
    members
    for size in (1, 2)
    for members in combinations(range(len(CATEGORIES)), size)
)


def normalize_caption(caption: str) -> str:
    """Reject unsupported/unsafe input and normalize Unicode deterministically."""
    if not isinstance(caption, str):
        raise ValueError('Prompt must be text')
    text = unicodedata.normalize('NFKC', caption)
    text = ' '.join(text.split())
    validate_request(text, 0.5)  # excludes known speech/voice-cloning requests
    encoded = text.encode('utf-8')
    if len(encoded) > MAX_BYTES:
        raise ValueError(f'Prompt must fit in {MAX_BYTES} UTF-8 bytes (no truncation)')
    return text


class ByteTextTokenizer:
    """Reproducible, model-independent UTF-8 encoding; zero is padding."""

    def encode(self, captions: list[str] | tuple[str, ...]) -> torch.Tensor:
        if not captions:
            raise ValueError('Provide at least one caption')
        if len(captions) > 4096:
            raise ValueError('Caption batch exceeds memory limit')
        result = torch.zeros((len(captions), MAX_TOKENS), dtype=torch.long)
        for index, prompt in enumerate(captions):
            encoded = normalize_caption(prompt).encode('utf-8')
            values = [START] + [value + BYTE_OFFSET for value in encoded] + [END]
            result[index, :len(values)] = torch.tensor(values)
        return result


class TextEncoder(nn.Module):
    """Learned UTF-8 embedding + positional transformer and masked pooling."""

    def __init__(self, channels: int = 32):
        super().__init__()
        if channels < 8 or channels % 8:
            raise ValueError('channels must be a multiple of eight')
        self.channels = channels
        self.tokens = nn.Embedding(BYTE_VOCAB, channels, padding_idx=PAD)
        self.position = nn.Parameter(torch.randn(MAX_TOKENS, channels) * 0.01)
        layer = nn.TransformerEncoderLayer(d_model=channels, nhead=4,
                                           dim_feedforward=channels * 3,
                                           dropout=0.0, batch_first=True,
                                           activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1,
                                                  enable_nested_tensor=False)
        self.project = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels))
        self.null_embedding = nn.Parameter(torch.zeros(channels))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != MAX_TOKENS:
            raise ValueError(f'Expected token ids [batch, {MAX_TOKENS}]')
        if token_ids.dtype != torch.long or not (token_ids >= 0).all() or not (token_ids < BYTE_VOCAB).all():
            raise ValueError('Token IDs must be valid int64 UTF-8 vocabulary positions')
        is_null = (token_ids == PAD).all(dim=1)
        # A fully masked transformer attention row is undefined. Give null
        # samples a single START token, then replace output with null_embedding.
        safe = token_ids.clone()
        safe[is_null, 0] = START
        valid = safe != PAD
        hidden = self.tokens(safe) + self.position[None]
        hidden = self.transformer(hidden, src_key_padding_mask=~valid)
        pooled = (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = self.project(pooled)
        return torch.where(is_null[:, None], self.null_embedding[None], pooled)


class TextWaveDenoiser(nn.Module):
    """From-scratch text/time-conditioned 1D DDPM denoiser."""

    def __init__(self, channels: int = 32):
        super().__init__()
        if channels < 8 or channels % 8:
            raise ValueError('channels must be a multiple of eight')
        self.channels = channels
        self.text_encoder = TextEncoder(channels)
        self.time_mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(),
                                      nn.Linear(channels * 2, channels))
        self.input = nn.Conv1d(1, channels, 5, padding=2)
        self.layers = nn.ModuleList(ConditionedResidual(channels, d)
                                    for d in (1, 2, 4, 8, 16, 1))
        self.output = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(),
                                    nn.Conv1d(channels, 1, 3, padding=1))

    def forward(self, audio: torch.Tensor, timestep: torch.Tensor,
                token_ids: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[1] != 1 or audio.shape[-1] < 16:
            raise ValueError('audio must be [batch, 1, at least 16 frames]')
        if timestep.ndim != 1 or timestep.shape[0] != audio.shape[0]:
            raise ValueError('timestep must be [batch]')
        if token_ids.shape[0] != audio.shape[0] or token_ids.device != audio.device:
            raise ValueError('text batch and device must match audio')
        half = self.channels // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=audio.device)
                                / max(half - 1, 1))
        angles = timestep.float()[:, None] * frequencies[None]
        time_encoding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        condition = self.text_encoder(token_ids) + self.time_mlp(time_encoding)
        hidden = self.input(audio)
        for layer in self.layers:
            hidden = layer(hidden, condition)
        return self.output(hidden)


@dataclass(frozen=True)
class CaptionedClips:
    clips: torch.Tensor       # [N, 1, 2048], float32
    tokens: torch.Tensor      # [N, 130], int64
    fingerprint: str
    examples: int


def _synth_prompts(ids: tuple[int, ...], language: str, variant: int) -> str:
    names = EN_LABELS if language == 'en' else HI_LABELS
    words = [names[i] for i in ids]
    item = (' and '.join(words)) if language == 'en' else (' और '.join(words))
    if language == 'en':
        return ('sound of ' if variant % 2 == 0 else 'recording of ') + item
    return ('आवाज़ ' if variant % 2 == 0 else 'ध्वनि ') + item


def synthetic_recording(members: tuple[int, ...], seed: int) -> torch.Tensor:
    """Crop around the first bird call and the onset of the thunder rumble."""
    dsp_prompt = ' and '.join(CATEGORIES[i] for i in members)
    audio = dsp_generate(dsp_prompt, 1.0, seed).samples
    mono = _downsample_24k_to_8k(audio)
    # Crops starting at 300–400 ms include the bird call at 350–540 ms.
    # Vary the offset by seed so even the deterministic synth changes phase.
    first = int(np.random.default_rng(seed).integers(
        3 * SAMPLE_RATE // 10, 4 * SAMPLE_RATE // 10 + 1,
    ))
    return torch.from_numpy(mono[first:first + CLIP_FRAMES].copy()).view(1, -1)


def synthetic_captioned_dataset(seeds: tuple[int, ...] = SYNTHETIC_TRAIN_SEEDS) -> CaptionedClips:
    """DSP originals paired with *different* English/Hindi text descriptions.

    Evaluation seeds must not overlap training seeds; captions are templates,
    NOT evidence that arbitrary natural language is understood.
    """
    if not seeds or len(set(seeds)) != len(seeds) or any(
        isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds
    ):
        raise ValueError('seeds must be nonempty, unique 32-bit integers')
    clips, captions = [], []
    fingerprint = hashlib.sha256(SYNTHETIC_RECIPE.encode('ascii'))
    for members in SYNTHETIC_COMBINATIONS:
        for seed in seeds:
            audio = synthetic_recording(members, seed)
            for language in ('en', 'hi'):
                caption = _synth_prompts(members, language, seed)
                captions.append(caption)
                clips.append(audio)
                fingerprint.update(bytes(members) + seed.to_bytes(4, 'little'))
                fingerprint.update(caption.encode('utf-8') + audio.numpy().tobytes())
    return CaptionedClips(torch.stack(clips), ByteTextTokenizer().encode(captions),
                          fingerprint.hexdigest(), len(clips))


def reviewed_captioned_dataset(manifest: Path, audio_root: Path, *,
                                clips_per_asset: int = 2) -> tuple[dict[str, CaptionedClips], dict]:
    """Reuse verified local audio bytes; preserve caption-to-clip ordering.

    Upstream rights checks are declarations; humans must verify permissions.
    Training/validation separation is enforced by recording group upstream.
    """
    from research.audio_dataset import load_reviewed_splits
    loaded, report = load_reviewed_splits(manifest, audio_root, clips_per_asset=clips_per_asset)
    captions: dict[str, list[str]] = {key: [] for key in loaded}
    with Path(manifest).open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                captions[row['split']].extend([row['caption']] * clips_per_asset)
    tokenizer = ByteTextTokenizer()
    output = {}
    for split, item in loaded.items():
        if len(item.clips):
            output[split] = CaptionedClips(torch.from_numpy(item.clips.copy()),
                                          tokenizer.encode(captions[split]),
                                          item.fingerprint, len(captions[split]))
    return output, report


def text_noise_loss(model: TextWaveDenoiser, schedule: DiffusionSchedule,
                    audio: torch.Tensor, tokens: torch.Tensor, *,
                    drop_condition: float = 0.1) -> torch.Tensor:
    if not math.isfinite(drop_condition) or not 0 <= drop_condition <= 1:
        raise ValueError('drop_condition must be in [0, 1]')
    if audio.shape[0] != tokens.shape[0]:
        raise ValueError('audio and token batch sizes must match')
    timesteps = torch.randint(schedule.steps, (len(audio),), device=audio.device)
    noise = torch.randn_like(audio)
    drop = torch.rand(len(audio), device=audio.device) < drop_condition
    conditions = torch.where(drop[:, None], torch.zeros_like(tokens), tokens)
    return F.mse_loss(model(schedule.corrupt(audio, timesteps, noise), timesteps, conditions), noise)


@torch.no_grad()
def sample_text(model: TextWaveDenoiser, schedule: DiffusionSchedule,
                caption: str, *, seed: int = 0, guidance: float = 1.0,
                frames: int = CLIP_FRAMES) -> torch.Tensor:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    if not math.isfinite(guidance) or not 0 <= guidance <= 4:
        raise ValueError('guidance must be finite in [0, 4]')
    if frames != CLIP_FRAMES:
        raise ValueError(f'text conditioning experiment supports exactly {CLIP_FRAMES} frames')
    device = next(model.parameters()).device
    ids = ByteTextTokenizer().encode([caption]).to(device)
    blank = torch.zeros_like(ids)
    generator = torch.Generator(device=device).manual_seed(seed)
    audio = torch.randn((1, 1, frames), generator=generator, device=device)
    model.eval()
    for step in range(schedule.steps - 1, -1, -1):
        t = torch.tensor([step], dtype=torch.long, device=device)
        predicted = model(audio, t, ids)
        if guidance != 1:
            unconditional = model(audio, t, blank)
            predicted = unconditional + guidance * (predicted - unconditional)
        alpha, bar, beta = schedule.alphas[step], schedule.alpha_bars[step], schedule.betas[step]
        mean = (audio - beta / (1 - bar).sqrt() * predicted) / alpha.sqrt()
        if step:
            variance = beta * (1 - schedule.alpha_bars[step - 1]) / (1 - bar)
            audio = mean + variance.sqrt() * torch.randn(audio.shape, device=device, generator=generator)
        else:
            audio = mean
    return audio.clamp(-1, 1).cpu()[0, 0]



@torch.no_grad()
def caption_alignment_probe(model: TextWaveDenoiser, schedule: DiffusionSchedule,
                            audio: torch.Tensor, tokens: torch.Tensor, *,
                            seed: int = 123) -> dict[str, float]:
    """Controlled noise-prediction comparison, NOT sound quality evaluation.

    Paired, shuffled, and unconditional prompts use IDENTICAL noisy inputs and
    timesteps, allowing a small check on whether caption conditioning matters.
    Shuffling is circular and never used for training or model selection.
    """
    if len(audio) != len(tokens) or len(audio) < 2:
        raise ValueError('Alignment probe requires >= 2 paired examples')
    size = min(16, len(audio))
    audio, tokens = audio[:size], tokens[:size]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        steps = torch.randint(schedule.steps, (size,), device=audio.device)
        noise = torch.randn_like(audio)
    corrupted = schedule.corrupt(audio, steps, noise)
    was_training = model.training
    model.eval()
    try:
        paired = float(F.mse_loss(model(corrupted, steps, tokens), noise))
        shuffled = float(F.mse_loss(model(corrupted, steps, tokens.roll(1, 0)), noise))
        unconditional = float(F.mse_loss(model(corrupted, steps, torch.zeros_like(tokens)), noise))
    finally:
        model.train(was_training)
    return {
        'paired_noise_mse': paired,
        'shuffled_noise_mse': shuffled,
        'null_noise_mse': unconditional,
        'paired_minus_shuffled_mse': paired - shuffled,
        'probe_examples': size,
    }

def save_checkpoint(output: Path, model: TextWaveDenoiser,
                    schedule: DiffusionSchedule, report: dict) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    np.savez_compressed(output / 'text_weights.npz', **tensors)
    manifest = {
        'format': CHECKPOINT_FORMAT, 'channels': model.channels,
        'diffusion_steps': schedule.steps, 'sample_rate': SAMPLE_RATE,
        'clip_frames': CLIP_FRAMES, 'max_prompt_bytes': MAX_BYTES,
        'text_vocab': BYTE_VOCAB, 'training_report': report,
        'weights_sha256': hashlib.sha256((output / 'text_weights.npz').read_bytes()).hexdigest(),
        'model_generation': None, 'weights_publication_authorized': False,
    }
    (output / 'text_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def load_checkpoint(path: Path) -> tuple[TextWaveDenoiser, DiffusionSchedule, dict]:
    path = Path(path)
    info = json.loads((path / 'text_manifest.json').read_text(encoding='utf-8'))
    if (info.get('format') != CHECKPOINT_FORMAT or info.get('sample_rate') != SAMPLE_RATE
        or info.get('clip_frames') != CLIP_FRAMES or info.get('max_prompt_bytes') != MAX_BYTES
        or info.get('text_vocab') != BYTE_VOCAB or info.get('model_generation') is not None):
        raise ValueError('Incompatible text diffusion checkpoint manifest')
    weight_file = path / 'text_weights.npz'
    if weight_file.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Checkpoint exceeds research size limit')
    if hashlib.sha256(weight_file.read_bytes()).hexdigest() != info['weights_sha256']:
        raise ValueError('Checkpoint SHA256 mismatch')
    channels, steps = info.get('channels'), info.get('diffusion_steps')
    if channels not in (8, 16, 24, 32, 40, 48, 56, 64) or not isinstance(steps, int) or not 8 <= steps <= 1000:
        raise ValueError('Unsupported text diffusion model configuration')
    model = TextWaveDenoiser(channels=channels)
    schedule = DiffusionSchedule(steps=steps)
    state = model.state_dict()
    with np.load(weight_file, allow_pickle=False) as checkpoint:
        if set(checkpoint.files) != set(state):
            raise ValueError('Unexpected text checkpoint tensor names')
        arrays = {}
        for key, tensor in state.items():
            value = checkpoint[key]
            if value.shape != tuple(tensor.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid text checkpoint tensor: ' + key)
            arrays[key] = torch.from_numpy(value.copy())
    model.load_state_dict(arrays, strict=True)
    return model.eval(), schedule, info


def train(*, steps: int = 50, seed: int = 10, channels: int = 32,
          diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
          output: Path | None = None, manifest: Path | None = None,
          audio_root: Path | None = None, clips_per_asset: int = 2) -> dict:
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 50000:
        raise ValueError('steps must be in [1, 50000]')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 - 2:
        raise ValueError('seed must be in [0, 2**32-2)')
    if (manifest is None) != (audio_root is None):
        raise ValueError('manifest and audio_root must be used together')
    torch.manual_seed(seed)
    model, schedule = TextWaveDenoiser(channels), DiffusionSchedule(diffusion_steps)
    rights_report = None
    if manifest is not None:
        splits, rights_report = reviewed_captioned_dataset(manifest, audio_root, clips_per_asset=clips_per_asset)
        train_set, val_set = splits['train'], splits['validation']
        source = 'rights-declared-local-wav'
    else:
        train_set = synthetic_captioned_dataset()
        val_set = synthetic_captioned_dataset(SYNTHETIC_VALIDATION_SEEDS)
        source = 'sonumio-synthetic-bilingual-dsp'
    if train_set.fingerprint == val_set.fingerprint:
        raise ValueError('Train/validation fingerprints must differ')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    started = perf_counter()
    losses = []
    model.train()
    for _ in range(steps):
        choices = torch.randint(len(train_set.clips), (min(8, len(train_set.clips)),))
        optimizer.zero_grad(set_to_none=True)
        loss = text_noise_loss(model, schedule, train_set.clips[choices], train_set.tokens[choices])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite diffusion training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        heldout = float(text_noise_loss(model, schedule, val_set.clips, val_set.tokens,
                                        drop_condition=0.0))
    alignment = caption_alignment_probe(model, schedule, val_set.clips, val_set.tokens, seed=seed + 2)
    report = {
        'purpose': 'experimental learned UTF-8 text-conditioned short waveform diffusion',
        'model_generation': None, 'not_production_ready': True,
        'training_examples': train_set.examples, 'heldout_examples': val_set.examples,
        'data_source': source, 'conditioning_mode': 'learned-byte-text',
        'synthetic_recipe': SYNTHETIC_RECIPE if manifest is None else None,
        'manifest_review': rights_report,
        'training_fingerprint': train_set.fingerprint,
        'heldout_fingerprint': val_set.fingerprint,
        'steps': steps, 'seed': seed, 'channels': channels,
        'initial_step_noise_mse': losses[0], 'last_step_noise_mse': losses[-1],
        'heldout_noise_mse': heldout,
        'caption_alignment_probe': alignment,
        'caption_alignment_probe_method': 'legacy-prefix-caption-rotation',
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'elapsed_seconds': round(perf_counter() - started, 3),
        'weights_published': False,
    }
    if output is not None:
        save_checkpoint(output, model, schedule, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    trainer = commands.add_parser('train')
    trainer.add_argument('--steps', type=int, default=50)
    trainer.add_argument('--seed', type=int, default=10)
    trainer.add_argument('--channels', type=int, default=32)
    trainer.add_argument('--diffusion-steps', type=int, default=DEFAULT_DIFFUSION_STEPS)
    trainer.add_argument('--output', type=Path, default=Path('checkpoints/private-text-experiment'))
    trainer.add_argument('--manifest', type=Path, help='Rights-declared JSONL training assets')
    trainer.add_argument('--audio-root', type=Path, help='Local vetted WAV root')
    trainer.add_argument('--clips-per-asset', type=int, default=2)
    sample = commands.add_parser('sample')
    sample.add_argument('--checkpoint', type=Path, required=True)
    sample.add_argument('--prompt', required=True)
    sample.add_argument('--seed', type=int, default=20)
    sample.add_argument('--guidance', type=float, default=1.0)
    sample.add_argument('--output', type=Path, default=Path('outputs/text-research.wav'))
    args = parser.parse_args()
    if args.command == 'train':
        report = train(steps=args.steps, seed=args.seed, channels=args.channels,
                       diffusion_steps=args.diffusion_steps, output=args.output,
                       manifest=args.manifest, audio_root=args.audio_root,
                       clips_per_asset=args.clips_per_asset)
        print(json.dumps(report, indent=2))
    else:
        model, schedule, _ = load_checkpoint(args.checkpoint)
        waveform = sample_text(model, schedule, args.prompt, seed=args.seed, guidance=args.guidance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encode_wav(waveform.numpy(), SAMPLE_RATE))
        print(f'Experimental text-conditional WAV: {args.output} ({CLIP_FRAMES/SAMPLE_RATE:.3f}s)')


if __name__ == '__main__':
    main()
