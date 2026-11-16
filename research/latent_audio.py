"""Experimental one-second English/Hindi text-conditioned *latent* audio diffusion.

Stage A trains a tiny convolutional audio autoencoder on Sonumio's own DSP;
Stage B trains a text-conditioned DDPM in the 64x-downsampled latent domain.
Neither stage claims useful audio quality or a Sonumio model generation.
No pretrained weights, remote datasets, model downloads or network calls.

Checkpoint loading uses numerical NPZ arrays, bounded model configurations,
SHA256 integrity, and the existing archive header preflight. Keep all weights
private. To evaluate sound quality, a separate listening and similarity study
is still required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.checkpoint_archive import validate_numeric_archive
from research.text_audio import (
    ByteTextTokenizer, TextEncoder, SYNTHETIC_COMBINATIONS, _synth_prompts,
)
from research.wave_diffusion import (
    CATEGORIES, DiffusionSchedule, SAMPLE_RATE, _downsample_24k_to_8k,
)
from sonumio.audio import encode_wav
from sonumio.engines.procedural import generate as dsp_generate

FRAMES = 8192                    # 1.024s at 8 kHz (NOT long-form)
LATENT_FRAMES = FRAMES // 64
LATENT_CHANNELS = 16
CHECKPOINT_FORMAT = 'sonumio-latent-audio-experiment-1'
DATA_RECIPE = 'seeded-window-dsp-latent-v2'
TRAIN_SEEDS = (0, 1)
VALIDATION_SEEDS = (101, 102)
MAX_WEIGHTS_BYTES = 64 * 1024 * 1024


class TinyAudioAutoencoder(nn.Module):
    """Trainable 64x temporal compression; no codebook or pretrained codec."""

    def __init__(self, channels: int = LATENT_CHANNELS):
        super().__init__()
        if channels != LATENT_CHANNELS:
            raise ValueError('Research codec supports exactly 16 latent channels')
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 8, stride=4, padding=2), nn.SiLU(),
            nn.Conv1d(8, 16, 8, stride=4, padding=2), nn.SiLU(),
            nn.Conv1d(16, channels, 8, stride=4, padding=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels, 16, 8, stride=4, padding=2), nn.SiLU(),
            nn.ConvTranspose1d(16, 8, 8, stride=4, padding=2), nn.SiLU(),
            nn.ConvTranspose1d(8, 1, 8, stride=4, padding=2), nn.Tanh(),
        )

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1:] != (1, FRAMES):
            raise ValueError(f'Expected 8 kHz mono waveforms [batch, 1, {FRAMES}]')
        if waveform.dtype != torch.float32 or not torch.isfinite(waveform).all():
            raise ValueError('Waveforms must contain finite float32 samples')
        return self.encoder(waveform)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[1:] != (LATENT_CHANNELS, LATENT_FRAMES):
            raise ValueError('Invalid latent layout')
        if not torch.isfinite(latents).all():
            raise ValueError('Latents must be finite')
        return self.decoder(latents)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(waveform))


def reconstruction_loss(original: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Waveform L1 plus multiresolution log-magnitude loss (NOT perceptual FAD)."""
    if original.shape != predicted.shape or original.ndim != 3 or original.shape[1:] != (1, FRAMES):
        raise ValueError('Reconstruction tensors must be matching mono 8192-frame clips')
    if not torch.isfinite(original).all() or not torch.isfinite(predicted).all():
        raise ValueError('Audio reconstruction contains nonfinite samples')
    result = F.l1_loss(original, predicted)
    flat_ref, flat_pred = original[:, 0], predicted[:, 0]
    for n_fft in (128, 512, 1024):
        window = torch.hann_window(n_fft, device=original.device, dtype=original.dtype)
        reference = torch.stft(flat_ref, n_fft, hop_length=n_fft // 4,
                               window=window, return_complex=True).abs()
        restored = torch.stft(flat_pred, n_fft, hop_length=n_fft // 4,
                              window=window, return_complex=True).abs()
        result = result + F.l1_loss(torch.log1p(restored), torch.log1p(reference)) / 3
    return result


class LatentResidual(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv1d(channels, channels, 3,
                              padding=dilation, dilation=dilation)
        self.cond = nn.Linear(channels, channels * 2)
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond(condition).unsqueeze(-1).chunk(2, dim=1)
        hidden = self.conv(F.silu(self.norm(latent)))
        hidden = F.silu(hidden * (1 + scale) + shift)
        return (latent + self.projection(hidden)) * (2 ** -0.5)


class TextLatentDenoiser(nn.Module):
    """Text conditioned latent noise predictor with classifier-free guidance."""

    def __init__(self, channels: int = 16):
        super().__init__()
        if type(channels) is not int or channels not in (16, 24, 32):
            raise ValueError('Denoiser channels must be 16, 24, or 32')
        self.channels = channels
        self.text_encoder = TextEncoder(channels)
        self.time_mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(),
                                      nn.Linear(channels * 2, channels))
        self.input = nn.Conv1d(LATENT_CHANNELS, channels, 3, padding=1)
        self.residual = nn.ModuleList(LatentResidual(channels, d)
                                      for d in (1, 2, 4, 8, 1))
        self.output = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(),
                                    nn.Conv1d(channels, LATENT_CHANNELS, 3, padding=1))

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 3 or noisy.shape[1:] != (LATENT_CHANNELS, LATENT_FRAMES):
            raise ValueError('Incorrect latent noise shape')
        if (timestep.ndim != 1 or timestep.shape[0] != noisy.shape[0]
                or tokens.ndim != 2 or tokens.shape[0] != noisy.shape[0]
                or tokens.device != noisy.device or timestep.device != noisy.device):
            raise ValueError('Text, timestep, and latent batch/device must agree')
        half = self.channels // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=noisy.device)
                                / max(half - 1, 1))
        angles = timestep.float()[:, None] * frequencies[None]
        embedding = self.text_encoder(tokens) + self.time_mlp(
            torch.cat((angles.sin(), angles.cos()), dim=1))
        hidden = self.input(noisy)
        for block in self.residual:
            hidden = block(hidden, embedding)
        return self.output(hidden)


class LatentResearchModel(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.codec = TinyAudioAutoencoder()
        self.denoiser = TextLatentDenoiser(channels)


@dataclass(frozen=True)
class BilingualWaves:
    waves: torch.Tensor
    tokens: torch.Tensor
    fingerprint: str
    examples: int


def render_seeded_waveform(members: tuple[int, ...], seed: int) -> np.ndarray:
    """Seed-specific crop prevents exact duplicates of seed-invariant ambient DSP.

    Evaluation still uses the same source family: this is a narrow leakage
    control, not proof of meaningful real-world generalization.
    """
    if members not in SYNTHETIC_COMBINATIONS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid source combination or recording seed')
    prompt = ' and '.join(CATEGORIES[i] for i in members)
    raw = dsp_generate(prompt, 1.5, seed).samples
    downsampled = _downsample_24k_to_8k(raw)
    offset = int(np.random.default_rng(seed).integers(400, 1601))
    cropped = downsampled[offset:offset + FRAMES]
    if cropped.shape != (FRAMES,) or not np.isfinite(cropped).all():
        raise ValueError('DSP input failed latent research shape checks')
    return cropped.copy()


def synthetic_bilingual_waves(seeds: tuple[int, ...] = TRAIN_SEEDS) -> BilingualWaves:
    """Seed-disjoint synthetic full-clip training/evaluation pairs."""
    if (not seeds or len(seeds) > 32 or len(seeds) != len(set(seeds))
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Seeds must be unique unsigned 32-bit integers (1..32)')
    fingerprint = hashlib.sha256(DATA_RECIPE.encode('ascii'))
    waves, prompts = [], []
    for members in SYNTHETIC_COMBINATIONS:
        for seed in seeds:
            waveform = render_seeded_waveform(members, seed)
            for language in ('en', 'hi'):
                caption = _synth_prompts(members, language, seed)
                waves.append(torch.from_numpy(waveform.copy()).view(1, FRAMES))
                prompts.append(caption)
                fingerprint.update(bytes(members) + seed.to_bytes(4, 'little'))
                fingerprint.update(caption.encode('utf-8') + waveform.tobytes())
    return BilingualWaves(torch.stack(waves), ByteTextTokenizer().encode(prompts),
                          fingerprint.hexdigest(), len(waves))


def latent_noise_loss(denoiser: TextLatentDenoiser, schedule: DiffusionSchedule,
                      clean: torch.Tensor, tokens: torch.Tensor, *,
                      drop_condition: float = 0.1) -> torch.Tensor:
    if (clean.ndim != 3 or clean.shape[1:] != (LATENT_CHANNELS, LATENT_FRAMES)
            or len(clean) != len(tokens)):
        raise ValueError('Latent/noise training batch does not match')
    if not math.isfinite(drop_condition) or not 0 <= drop_condition <= 1:
        raise ValueError('Condition dropout must be finite and between 0 and 1')
    timesteps = torch.randint(schedule.steps, (len(clean),), device=clean.device)
    noise = torch.randn_like(clean)
    prompt = torch.where(torch.rand(len(clean), device=clean.device)[:, None]
                         < drop_condition, torch.zeros_like(tokens), tokens)
    corrupted = schedule.corrupt(clean, timesteps, noise)
    return F.mse_loss(denoiser(corrupted, timesteps, prompt), noise)


@torch.no_grad()
def sample_latent(model: LatentResearchModel, schedule: DiffusionSchedule,
                  caption: str, *, seed: int = 0, guidance: float = 1.0,
                  clamp_output: bool = True) -> torch.Tensor:
    """One-second neural research sample, NOT high-quality generative output."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    if not math.isfinite(guidance) or not 0 <= guidance <= 4:
        raise ValueError('Guidance must be finite and in [0, 4]')
    if type(clamp_output) is not bool:
        raise ValueError('clamp_output must be boolean')
    device = next(model.parameters()).device
    tokens = ByteTextTokenizer().encode([caption]).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((1, LATENT_CHANNELS, LATENT_FRAMES), generator=generator, device=device)
    previous_mode = model.training
    model.eval()
    blank = torch.zeros_like(tokens)
    try:
        for step in range(schedule.steps - 1, -1, -1):
            t = torch.tensor([step], dtype=torch.long, device=device)
            predicted = model.denoiser(x, t, tokens)
            if guidance != 1:
                unconditional = model.denoiser(x, t, blank)
                predicted = unconditional + guidance * (predicted - unconditional)
            beta, alpha, bar = schedule.betas[step], schedule.alphas[step], schedule.alpha_bars[step]
            mean = (x - beta * predicted / (1 - bar).sqrt()) / alpha.sqrt()
            if step:
                variance = beta * (1 - schedule.alpha_bars[step - 1]) / (1 - bar)
                x = mean + variance.sqrt() * torch.randn(x.shape, generator=generator, device=device)
            else:
                x = mean
    finally:
        model.train(previous_mode)
    waveform = model.codec.decode(x)[0, 0].cpu()
    return waveform.clamp(-1, 1) if clamp_output else waveform


def _save_checkpoint(path: Path, model: LatentResearchModel,
                     schedule: DiffusionSchedule, report: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    file = path / 'latent_weights.npz'
    state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    np.savez_compressed(file, **state)
    manifest = {
        'format': CHECKPOINT_FORMAT, 'channels': model.denoiser.channels,
        'latent_channels': LATENT_CHANNELS, 'frames': FRAMES,
        'sample_rate': SAMPLE_RATE, 'diffusion_steps': schedule.steps,
        'weights_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
        'training_report': report, 'model_generation': None,
        'weights_publication_authorized': False,
    }
    (path / 'latent_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                               encoding='utf-8')


def load_checkpoint(path: Path) -> tuple[LatentResearchModel, DiffusionSchedule, dict]:
    path = Path(path)
    manifest = json.loads((path / 'latent_manifest.json').read_text(encoding='utf-8'))
    channels, steps = manifest.get('channels'), manifest.get('diffusion_steps')
    if (manifest.get('format') != CHECKPOINT_FORMAT or manifest.get('sample_rate') != SAMPLE_RATE
            or manifest.get('latent_channels') != LATENT_CHANNELS
            or manifest.get('frames') != FRAMES or manifest.get('model_generation') is not None
            or manifest.get('weights_publication_authorized') is not False
            or type(channels) is not int or channels not in (16, 24, 32)
            or type(steps) is not int or not 8 <= steps <= 256):
        raise ValueError('Unsupported or unsafe latent checkpoint manifest')
    file = path / 'latent_weights.npz'
    if file.stat().st_size > MAX_WEIGHTS_BYTES:
        raise ValueError('Latent checkpoint is too large')
    if hashlib.sha256(file.read_bytes()).hexdigest() != manifest.get('weights_sha256'):
        raise ValueError('Latent checkpoint SHA256 mismatch')
    model = LatentResearchModel(channels)
    schedule = DiffusionSchedule(steps)
    expected = model.state_dict()
    validate_numeric_archive(file, expected)
    with np.load(file, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError('Unexpected latent checkpoint parameter names')
        weights = {}
        for name, target in expected.items():
            tensor = archive[name]
            if (tensor.shape != tuple(target.shape) or tensor.dtype != np.float32
                    or not np.isfinite(tensor).all()):
                raise ValueError('Invalid latent checkpoint tensor: ' + name)
            weights[name] = torch.from_numpy(tensor.copy())
        model.load_state_dict(weights, strict=True)
    return model.eval(), schedule, manifest


def train(*, ae_steps: int = 10, diffusion_steps: int = 10,
          schedule_steps: int = 8, seed: int = 7, channels: int = 16,
          output: Path | None = None) -> dict:
    """Sequential short smoke stages; no perceptual quality assertion."""
    if any(type(n) is not int or not 1 <= n <= 20000 for n in (ae_steps, diffusion_steps)):
        raise ValueError('Research training steps must be positive integers <= 20000')
    if type(schedule_steps) is not int or not 8 <= schedule_steps <= 256:
        raise ValueError('Diffusion schedule must be 8..256 steps')
    if type(seed) is not int or not 0 <= seed < 2**32 - 2:
        raise ValueError('Seed must be an unsigned integer < 2**32-2')
    if channels not in (16, 24, 32):
        raise ValueError('Unsupported latent denoiser width')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        train_set = synthetic_bilingual_waves()
        val_set = synthetic_bilingual_waves(VALIDATION_SEEDS)
        if train_set.fingerprint == val_set.fingerprint:
            raise ValueError('Audio research train and evaluation datasets overlap')
        model = LatentResearchModel(channels)
        schedule = DiffusionSchedule(schedule_steps)
        optimizer = torch.optim.AdamW(model.codec.parameters(), lr=0.001)
        started = perf_counter()
        ae_losses = []
        model.codec.train()
        for _ in range(ae_steps):
            indices = torch.randint(len(train_set.waves), (2,))
            batch = train_set.waves[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(batch, model.codec(batch))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite codec reconstruction loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.codec.parameters(), 1.0)
            optimizer.step()
            ae_losses.append(float(loss.detach()))
        model.codec.eval()
        # Bilingual captions share each underlying waveform. Evaluate the
        # entire 42-recording validation set once, not the first EN/HI pair.
        # Encode in small chunks to avoid holding long STFT graphs in RAM.
        with torch.no_grad():
            unique_validation_waves = val_set.waves[::2]
            codec_weighted_error = 0.0
            for start in range(0, len(unique_validation_waves), 4):
                batch = unique_validation_waves[start:start + 4]
                codec_weighted_error += (
                    float(reconstruction_loss(batch, model.codec(batch))) * len(batch)
                )
            val_codec_loss = codec_weighted_error / len(unique_validation_waves)
            train_latents = model.codec.encode(train_set.waves).detach()
            val_latents = model.codec.encode(val_set.waves).detach()
        optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=0.0005)
        diffusion_losses = []
        model.denoiser.train()
        for _ in range(diffusion_steps):
            ids = torch.randint(len(train_latents), (2,))
            optimizer.zero_grad(set_to_none=True)
            loss = latent_noise_loss(model.denoiser, schedule, train_latents[ids], train_set.tokens[ids])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite diffusion noise prediction loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)
            optimizer.step()
            diffusion_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            torch.manual_seed(seed + 1)
            heldout_weighted_error = 0.0
            for start in range(0, len(val_latents), 4):
                group = val_latents[start:start + 4]
                captions = val_set.tokens[start:start + 4]
                heldout_weighted_error += (
                    float(latent_noise_loss(model.denoiser, schedule, group,
                                            captions, drop_condition=0.0)) * len(group)
                )
            heldout = heldout_weighted_error / len(val_latents)
        report = {
            'purpose': 'one-second 8kHz bilingual latent audio diffusion research only',
            'model_generation': None, 'not_production_ready': True,
            'data_source': 'Sonumio-owned synthetic procedural DSP',
            'data_recipe': DATA_RECIPE, 'train_fingerprint': train_set.fingerprint,
            'validation_fingerprint': val_set.fingerprint, 'train_examples': train_set.examples,
            'validation_examples': val_set.examples,
            'heldout_unique_recordings': len(unique_validation_waves),
            'heldout_caption_noise_examples': len(val_latents),
            'validation_protocol': 'all-unique-recordings-for-codec_all-captions-for-noise',
            'sample_rate': SAMPLE_RATE,
            'audio_frames': FRAMES, 'latent_frames': LATENT_FRAMES,
            'latent_channels': LATENT_CHANNELS, 'seed': seed, 'channels': channels,
            'codec_steps': ae_steps, 'denoiser_steps': diffusion_steps,
            'diffusion_schedule_steps': schedule_steps,
            'codec_initial_loss': ae_losses[0], 'codec_last_loss': ae_losses[-1],
            'heldout_codec_reconstruction_loss': val_codec_loss,
            'denoiser_initial_loss': diffusion_losses[0],
            'denoiser_last_loss': diffusion_losses[-1],
            'heldout_noise_mse': heldout,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'elapsed_seconds': round(perf_counter() - started, 3),
            'weights_published': False,
        }
        if output is not None:
            _save_checkpoint(Path(output), model, schedule, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    trainer = commands.add_parser('train')
    trainer.add_argument('--ae-steps', type=int, default=10)
    trainer.add_argument('--diffusion-steps', type=int, default=10)
    trainer.add_argument('--schedule-steps', type=int, default=8)
    trainer.add_argument('--seed', type=int, default=7)
    trainer.add_argument('--channels', type=int, default=16)
    trainer.add_argument('--output', type=Path, default=Path('checkpoints/private-latent-experiment'))
    sampler = commands.add_parser('sample')
    sampler.add_argument('--checkpoint', type=Path, required=True)
    sampler.add_argument('--prompt', type=str, required=True)
    sampler.add_argument('--seed', type=int, default=20)
    sampler.add_argument('--guidance', type=float, default=1.0)
    sampler.add_argument('--output', type=Path, default=Path('outputs/latent-experiment.wav'))
    args = parser.parse_args()
    if args.command == 'train':
        report = train(ae_steps=args.ae_steps, diffusion_steps=args.diffusion_steps,
                       schedule_steps=args.schedule_steps, seed=args.seed,
                       channels=args.channels, output=args.output)
        print(json.dumps(report, indent=2))
    else:
        model, schedule, _ = load_checkpoint(args.checkpoint)
        waveform = sample_latent(model, schedule, args.prompt, seed=args.seed,
                                 guidance=args.guidance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encode_wav(waveform.numpy(), SAMPLE_RATE))
        print(f'Private experimental 8kHz WAV: {args.output} ({FRAMES / SAMPLE_RATE:.3f}s)')


if __name__ == '__main__':
    main()
