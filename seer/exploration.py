"""Exploration strategies for selecting source--reference training pairs."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from seer.retriever import Selector


@dataclass(frozen=True)
class Decision:
    """Chosen bank indices with optional explorer-specific score diagnostics."""

    emotion: int
    reference_indices: tuple[int, ...]
    means: Tensor | None = None
    bonuses: Tensor | None = None


class Explorer(Protocol):
    def observe(
        self, pairs: Iterable[tuple[int, int]]
    ) -> tuple[tuple[int, int], ...]: ...

    def choose(
        self,
        model: Selector,
        source_frames: Tensor,
        references: Tensor,
        reference_emotions: Tensor,
        count: int,
        forced_emotion: int | None = None,
        explore: bool = True,
    ) -> Decision: ...

    def refresh(self, features: Tensor) -> None: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object], device: torch.device) -> None: ...


class UCBExplorer:
    """Last-layer exploration adapted from Neural-LinUCB for sampled EVC rewards."""

    def __init__(
        self,
        feature_dim: int,
        alpha: float = 1.0,
        ridge: float = 1.0,
        device: str | torch.device = "cpu",
        *,
        history_capacity: int | None = None,
    ) -> None:
        if history_capacity is not None and history_capacity < 1:
            raise ValueError("history_capacity must be positive")
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.feature_dim = int(feature_dim)
        self.history_capacity = history_capacity
        self._history: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.covariance = torch.eye(
            self.feature_dim + 1, device=device, dtype=torch.float32
        ) * self.ridge

    def observe(
        self, pairs: Iterable[tuple[int, int]]
    ) -> tuple[tuple[int, int], ...]:
        for pair in pairs:
            self._history.setdefault(pair, None)
            if self.history_capacity is not None and len(self._history) > self.history_capacity:
                self._history.popitem(last=False)
        return tuple(self._history)

    @staticmethod
    def linear_features(features: Tensor) -> Tensor:
        normalized = F.normalize(features, dim=-1)
        bias = torch.ones(*normalized.shape[:-1], 1, device=features.device)
        return torch.cat((normalized, bias), dim=-1) / math.sqrt(2.0)

    def bonus(self, features: Tensor) -> Tensor:
        linear = self.linear_features(features)
        solved = torch.linalg.solve(self.covariance, linear.T).T
        variance = torch.clamp((linear * solved).sum(dim=-1), min=0.0)
        return self.alpha * torch.sqrt(variance)

    def choose(
        self,
        model: Selector,
        source_frames: Tensor,
        references: Tensor,
        reference_emotions: Tensor,
        count: int,
        forced_emotion: int | None = None,
        explore: bool = True,
    ) -> Decision:
        source = model.encode_source(source_frames)
        route = model.router(source)
        emotion = (
            int(route.argmax().detach().cpu())
            if forced_emotion is None
            else int(forced_emotion)
        )
        eligible = torch.nonzero(
            reference_emotions.to(model.device) == emotion, as_tuple=False
        ).flatten()
        if count < 1 or count > len(eligible):
            raise ValueError("requested actions exceed the routed reference bank")
        means, features = model.score_pairs(source, references[eligible])
        bonuses = self.bonus(features) if explore else torch.zeros_like(means)
        local = torch.topk(means + bonuses, count).indices
        chosen = tuple(int(value) for value in eligible[local].detach().cpu())
        return Decision(emotion, chosen, means, bonuses)

    @torch.no_grad()
    def refresh(self, features: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("covariance features have the wrong shape")
        linear = self.linear_features(features.detach())
        value = torch.eye(
            self.feature_dim + 1,
            device=features.device,
            dtype=features.dtype,
        ) * self.ridge
        self.covariance = value + linear.T @ linear

    def state_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "ridge": self.ridge,
            "feature_dim": self.feature_dim,
            "history_capacity": self.history_capacity,
            "history_keys": tuple(self._history),
            "covariance": self.covariance.cpu(),
        }

    def load_state_dict(self, state: dict[str, object], device: torch.device) -> None:
        if (
            float(state["alpha"]) != self.alpha
            or float(state["ridge"]) != self.ridge
            or int(state["feature_dim"]) != self.feature_dim
            or state["history_capacity"] != self.history_capacity
        ):
            raise ValueError("UCB explorer checkpoint configuration differs")
        keys = [tuple(pair) for pair in state["history_keys"]]
        if (
            any(len(pair) != 2 for pair in keys)
            or len(set(keys)) != len(keys)
            or (self.history_capacity is not None and len(keys) > self.history_capacity)
        ):
            raise ValueError("UCB explorer checkpoint history differs")
        self._history = OrderedDict((pair, None) for pair in keys)
        self.covariance = torch.as_tensor(state["covariance"]).to(device)
