"""Original contrastive English/Hindi text-to-synthetic-audio alignment research.

This is an independent trainable alignment probe, not Sonumio-1 or CLAP.
A small original audio encoder is paired with Sonumio's own byte-token
Transformer. Training batches contain DIFFERENT DSP category-combination
groups so another recording of the same category is never incorrectly
used as a negative. Exact English/Hindi audio duplicates are paired
to the same source group, never double-counted as separate sounds.

The project currently covers only six procedural sound sources and their
15 two-source mixtures, short 8-kHz clips and templated bilingual captions.
No pretrained encoders, external datasets, model downloads, speech/voice
identification, saved model weights, WAV assets or model publication.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.latent_audio import FRAMES
from research.text_audio import (
    BYTE_VOCAB, MAX_TOKENS, TextEncoder,
)
from research.wave_diffusion import SAMPLE_RATE

PROTOCOL = "sonumio-from-scratch-bilingual-audio-text-alignment-v1"
CONTRASTIVE_PROTOCOL = "sonumio-category-disjoint-symmetric-infonce-v1"
CHANNEL_OPTIONS = (16, 24, 32)
EMBEDDING_OPTIONS = (16, 32, 64)
MIN_GROUPS = 2
MAX_GROUPS = 16
MAX_SIGNAL = 1.0001


def _inputs(audio: torch.Tensor, tokens: torch.Tensor) -> None:
    if (not isinstance(audio, torch.Tensor)
            or not isinstance(tokens, torch.Tensor)
            or audio.ndim != 3 or audio.shape[1:] != (1, FRAMES)
            or not MIN_GROUPS <= len(audio) <= MAX_GROUPS
            or tokens.shape != (len(audio), MAX_TOKENS)
            or audio.dtype != torch.float32
            or tokens.dtype != torch.long
            or audio.device.type != "cpu"
            or tokens.device.type != "cpu"
            or not torch.isfinite(audio).all()
            or not (audio.abs() <= MAX_SIGNAL).all()
            or not (tokens >= 0).all()
            or not (tokens < BYTE_VOCAB).all()):
        raise ValueError(
            "Alignment inputs must be valid grouped finite 8192-frame audio "
            "and UTF-8 token IDs on CPU"
        )


class SourceAudioEncoder(nn.Module):
    """Small differentiable 8-kHz mono audio feature tower; no pretrained net."""

    def __init__(self, *, channels: int = 16,
                 embedding_dim: int = 32) -> None:
        super().__init__()
        if (type(channels) is not int or channels not in CHANNEL_OPTIONS
                or type(embedding_dim) is not int
                or embedding_dim not in EMBEDDING_OPTIONS):
            raise ValueError("Invalid bounded audio alignment architecture")
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.convolutions = nn.Sequential(
            nn.Conv1d(1, 8, 9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(8, channels, 9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 7, stride=4, padding=3),
            nn.SiLU(),
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Sequential(
            nn.LayerNorm(channels + 16),
            nn.Linear(channels + 16, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, embedding_dim),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if (not isinstance(audio, torch.Tensor)
                or audio.ndim != 3
                or audio.shape[1:] != (1, FRAMES)
                or not 1 <= len(audio) <= MAX_GROUPS
                or audio.dtype != torch.float32
                or audio.device.type != "cpu"
                or not torch.isfinite(audio).all()
                or audio.abs().max().item() > MAX_SIGNAL):
            raise ValueError("Source audio encoder requires finite 8-kHz clips")
        feature = self.temporal_pool(self.convolutions(audio)).squeeze(-1)
        energy = audio.double().square().reshape(
            len(audio), 1, 16, FRAMES // 16,
        ).mean(-1).sqrt().reshape(len(audio), 16).float()
        result = self.project(torch.cat((feature, energy), dim=1))
        if (result.shape != (len(audio), self.embedding_dim)
                or not torch.isfinite(result).all()):
            raise ValueError("Nonfinite or incorrectly shaped audio embedding")
        return F.normalize(result, dim=-1, eps=1e-8)


class OriginalAudioTextAligner(nn.Module):
    """Original small audio tower and original UTF-8 text Transformer."""

    def __init__(self, *, channels: int = 16,
                 embedding_dim: int = 32) -> None:
        super().__init__()
        if (type(channels) is not int or channels not in CHANNEL_OPTIONS
                or type(embedding_dim) is not int
                or embedding_dim not in EMBEDDING_OPTIONS):
            raise ValueError("Invalid alignment model architecture")
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.audio_tower = SourceAudioEncoder(
            channels=channels, embedding_dim=embedding_dim,
        )
        self.text_tower = TextEncoder(channels)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, embedding_dim),
        )

    def encode_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.audio_tower(waveforms)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        if (not isinstance(tokens, torch.Tensor)
                or tokens.ndim != 2
                or not 1 <= len(tokens) <= MAX_GROUPS
                or tokens.shape[1] != MAX_TOKENS
                or tokens.dtype != torch.long
                or tokens.device.type != "cpu"):
            raise ValueError("Invalid bounded alignment text token batch")
        embedded = self.text_tower(tokens)
        projected = self.text_projection(embedded)
        if (projected.shape != (len(tokens), self.embedding_dim)
                or not torch.isfinite(projected).all()):
            raise ValueError("Invalid alignment text embedding")
        return F.normalize(projected, dim=-1, eps=1e-8)

    def forward(
        self, waveforms: torch.Tensor, tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _inputs(waveforms, tokens)
        return self.encode_audio(waveforms), self.encode_text(tokens)


def category_disjoint_info_nce(
    audio_features: torch.Tensor,
    text_features: torch.Tensor,
    category_groups: tuple[str, ...], *,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Symmetric paired contrastive loss on unique acoustic categories.

    A recording group with English and Hindi captions MUST NOT be
    inserted twice in the same batch. Two different recording seeds of
    one semantic category also cannot be treated as negatives. Explicit
    group keys implement that guard without a learned label classifier.
    """
    if (not isinstance(audio_features, torch.Tensor)
            or not isinstance(text_features, torch.Tensor)
            or audio_features.ndim != 2
            or audio_features.shape != text_features.shape
            or not MIN_GROUPS <= len(audio_features) <= MAX_GROUPS
            or audio_features.device.type != "cpu"
            or text_features.device.type != "cpu"
            or audio_features.dtype != torch.float32
            or text_features.dtype != torch.float32
            or not torch.isfinite(audio_features).all()
            or not torch.isfinite(text_features).all()
            or type(category_groups) is not tuple
            or len(category_groups) != len(audio_features)
            or any(type(key) is not str or not key for key in category_groups)
            or len(set(category_groups)) != len(category_groups)
            or type(temperature) not in (float, int)
            or not math.isfinite(temperature)
            or not 0.02 <= temperature <= 1):
        raise ValueError("Contrastive pairs require disjoint acoustic categories")
    normalized_audio = F.normalize(audio_features, dim=1, eps=1e-8)
    normalized_text = F.normalize(text_features, dim=1, eps=1e-8)
    logits = normalized_audio @ normalized_text.T / float(temperature)
    labels = torch.arange(len(audio_features), dtype=torch.long)
    loss_audio = F.cross_entropy(logits, labels)
    loss_text = F.cross_entropy(logits.T, labels)
    loss = (loss_audio + loss_text) / 2
    if not torch.isfinite(loss) or loss < 0:
        raise ValueError("Nonfinite original audio-text contrastive objective")
    with torch.no_grad():
        diagonal = logits.diagonal()
        negatives = logits[~torch.eye(len(logits), dtype=torch.bool)]
        positive_mean = float(diagonal.mean())
        negative_mean = float(negatives.mean())
        cosine_values = (normalized_audio @ normalized_text.T).diagonal()
        report = {
            "protocol": CONTRASTIVE_PROTOCOL,
            "category_groups": list(category_groups),
            "temperature": float(temperature),
            "unique_categories": len(category_groups),
            "positive_similarity_mean": positive_mean,
            "off_diagonal_similarity_mean": negative_mean,
            "positive_cosine_mean": float(cosine_values.mean()),
            "paired_loss": float(loss),
            "duplicates_or_same_category_false_negatives_allowed": False,
            "external_model_or_artifacts_used": False,
            "human_semantic_alignment_measured": False,
        }
    if any(not math.isfinite(report[key]) for key in (
        "positive_similarity_mean",
        "off_diagonal_similarity_mean",
        "positive_cosine_mean", "paired_loss",
    )):
        raise ValueError("Nonfinite contrastive research score")
    return loss, report


@torch.no_grad()
def retrieval_metrics(
    audio_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    category_groups: tuple[str, ...],
) -> dict:
    """One candidate per acoustic category; numeric ranking NOT semantics."""
    if (not isinstance(audio_embeddings, torch.Tensor)
            or not isinstance(text_embeddings, torch.Tensor)
            or audio_embeddings.ndim != 2
            or audio_embeddings.shape != text_embeddings.shape
            or not MIN_GROUPS <= len(audio_embeddings) <= 32
            or audio_embeddings.dtype != torch.float32
            or text_embeddings.dtype != torch.float32
            or not torch.isfinite(audio_embeddings).all()
            or not torch.isfinite(text_embeddings).all()
            or type(category_groups) is not tuple
            or len(category_groups) != len(audio_embeddings)
            or len(set(category_groups)) != len(category_groups)):
        raise ValueError("Retrieval requires one valid candidate per category")
    a = F.normalize(audio_embeddings, dim=1, eps=1e-8)
    b = F.normalize(text_embeddings, dim=1, eps=1e-8)
    similarities = a @ b.T
    ranks = []
    scores = []
    for index in range(len(a)):
        # Distinct acoustic categories only. Ties are broken by index
        # deterministically; a random untrained model may have any rank.
        ordering = sorted(
            range(len(a)),
            key=lambda k: (-float(similarities[index, k]), k),
        )
        rank = ordering.index(index) + 1
        others = [
            float(similarities[index, j])
            for j in range(len(a)) if j != index
        ]
        ranks.append(rank)
        scores.append(
            float(similarities[index, index]) - sum(others) / len(others)
        )
    return {
        "category_candidates": len(a),
        "rank1_matches": sum(x == 1 for x in ranks),
        "rank5_matches": sum(x <= 5 for x in ranks),
        "recall_at_1": sum(x == 1 for x in ranks) / len(a),
        "recall_at_5": sum(x <= 5 for x in ranks) / len(a),
        "mean_reciprocal_rank": sum(1 / rank for rank in ranks) / len(a),
        "mean_matched_minus_other_cosine": sum(scores) / len(scores),
        "rank_per_category": ranks,
        "all_candidates_from_synthetic_dsp_only": True,
        "english_hindi_same_recording_not_independent": True,
        "open_vocabulary_or_audio_quality_established": False,
    }
