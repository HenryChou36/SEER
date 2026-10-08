from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SelectorConfig:
    source_dim: int = 1024
    reference_dim: int = 2064
    hidden_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn_dim: int = 512
    pair_hidden_dim: int = 128
    pair_feature_dim: int = 64

    @classmethod
    def from_experiment(cls, values: Mapping[str, object]) -> "SelectorConfig":
        return cls(
            hidden_dim=int(values["hidden_dim"]),
            transformer_layers=int(values["transformer_layers"]),
            transformer_heads=int(values["transformer_heads"]),
            transformer_ffn_dim=int(values["transformer_ffn_dim"]),
            pair_hidden_dim=int(values["bandit_hidden_dim"]),
            pair_feature_dim=int(values["bandit_feature_dim"]),
        )


class SourceEncoder(nn.Module):
    def __init__(self, config: SelectorConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        self.source_dim = config.source_dim
        self.input = nn.Sequential(
            nn.LayerNorm(config.source_dim),
            nn.Linear(config.source_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.summary = nn.Parameter(torch.empty(1, config.hidden_dim))
        nn.init.normal_(self.summary, std=0.02)

    @staticmethod
    def positions(length: int, dimension: int, device: torch.device) -> Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(length, dimension, device=device)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        return encoding

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 2 or frames.shape[1] != self.source_dim or not len(frames):
            raise ValueError(f"source features must have shape T x {self.source_dim}")
        frames = frames.to(device=self.summary.device, dtype=torch.float32)
        sequence = torch.cat((self.summary, self.input(frames)), dim=0)
        sequence = sequence + self.positions(
            sequence.shape[0], sequence.shape[1], sequence.device
        )
        return self.transformer(sequence.unsqueeze(0))[0, 0]


class Selector(nn.Module):
    """Differentiable source-conditioned reference-utility model."""

    def __init__(
        self,
        emotion_count: int,
        config: SelectorConfig = SelectorConfig(),
    ) -> None:
        super().__init__()
        if emotion_count < 1:
            raise ValueError("emotion_count must be positive")
        self.config = config
        self.emotion_count = emotion_count
        hidden = config.hidden_dim
        self.source_encoder = SourceEncoder(config)
        self.reference_encoder = nn.Sequential(
            nn.LayerNorm(config.reference_dim),
            nn.Linear(config.reference_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        # This unused module consumes RNG before the scorer is initialized;
        # retain it to match the reported experiment's seeded initialization.
        self.identity_projection = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.pair_encoder = nn.Sequential(
            # The reported implementation reserved one final coordinate for
            # the source fallback. Real reference pairs always set it to zero.
            nn.LayerNorm(2 * hidden + 1),
            nn.Linear(2 * hidden + 1, config.pair_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.pair_hidden_dim),
            nn.Linear(config.pair_hidden_dim, config.pair_feature_dim),
            nn.GELU(),
        )
        self.utility = nn.Linear(config.pair_feature_dim, 1)
        self.router = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, emotion_count),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_source(self, frames: Tensor) -> Tensor:
        return self.source_encoder(frames)

    def encode_references(self, references: Tensor) -> Tensor:
        if references.ndim != 2 or references.shape[1] != self.config.reference_dim:
            raise ValueError(
                f"reference features must have shape N x {self.config.reference_dim}"
            )
        return self.reference_encoder(
            references.to(device=self.device, dtype=torch.float32)
        )

    def pair_features(self, source: Tensor, references: Tensor) -> Tensor:
        if source.shape != (self.config.hidden_dim,):
            raise ValueError("source summary has the wrong dimension")
        encoded = self.encode_references(references)
        repeated = source.unsqueeze(0).expand(len(encoded), -1)
        interactions = torch.cat(
            (
                torch.abs(repeated - encoded),
                repeated * encoded,
                torch.zeros(
                    len(encoded),
                    1,
                    device=self.device,
                    dtype=repeated.dtype,
                ),
            ),
            dim=-1,
        )
        return self.pair_encoder(interactions)

    def score_pairs(self, source: Tensor, references: Tensor) -> tuple[Tensor, Tensor]:
        features = self.pair_features(source, references)
        return self.utility(features).squeeze(-1), features

    def forward(self, frames: Tensor, references: Tensor) -> dict[str, Tensor]:
        source = self.encode_source(frames)
        means, features = self.score_pairs(source, references)
        return {
            "source": source,
            "route_logits": self.router(source),
            "pair_features": features,
            "utilities": means,
        }

    @torch.no_grad()
    def select_reference(
        self, frames: Tensor, references: Tensor, reference_emotions: Tensor
    ) -> tuple[int, int | None, float]:
        """Select the highest utility reference, with unconverted speech at zero."""
        source = self.encode_source(frames)
        emotion = int(self.router(source).argmax().cpu())
        eligible = torch.nonzero(
            reference_emotions.to(self.device) == emotion, as_tuple=False
        ).flatten()
        if not len(eligible):
            raise ValueError("predicted emotion has no bank references")
        means, _ = self.score_pairs(source, references[eligible])
        local = int(torch.topk(means, 1).indices[0].cpu())
        utility = float(means[local].cpu())
        reference = int(eligible[local].cpu()) if utility > 0 else None
        return emotion, reference, utility
