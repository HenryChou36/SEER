"""Shared speech-emotion recognition adapter and persistent model loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


NATIVE_LABELS = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)
EMBEDDING_DIM = 1024
LABEL_ALIASES = {"disgust": "disgusted", "surprise": "surprised"}


def normalize_label(value: object) -> str:
    label = str(value).strip().casefold()
    if "/" in label:
        label = label.rsplit("/", 1)[-1]
    return "unknown" if label == "<unk>" else label


@dataclass(frozen=True)
class EmotionResult:
    scores: np.ndarray
    embedding: np.ndarray | None

    def project(self, labels: Sequence[str]) -> tuple[str, dict[str, float]]:
        canonical = [LABEL_ALIASES.get(label.casefold(), label.casefold()) for label in labels]
        indices = [NATIVE_LABELS.index(label) for label in canonical]
        values = self.scores[indices]
        if values.sum() <= 0:
            raise ValueError("projected Emotion2Vec+ scores have zero mass")
        values = values / values.sum()
        probabilities = dict(zip((label.casefold() for label in labels), values.tolist()))
        prediction = max(probabilities, key=probabilities.get)
        return prediction, probabilities


class Emotion2Vec:
    """Resident wrapper around the unmodified Emotion2Vec+ inference API."""

    def __init__(self, model: str | Path, device: str = "cuda", batch_size: int = 64):
        from funasr import AutoModel

        model_root = Path(model).resolve()
        for name in ("model.pt", "config.yaml", "tokens.txt"):
            if not (model_root / name).is_file():
                raise FileNotFoundError(f"missing Emotion2Vec+ Large file: {name}")
        self.model_path = str(model_root)
        self.batch_size = batch_size
        self.model = AutoModel(
            model=self.model_path,
            device=device,
            disable_update=True,
        )
        labels = [
            normalize_label(label)
            for label in self.model.kwargs["tokenizer"].token_list
        ]
        head = self.model.model.proj
        if (
            labels != list(NATIVE_LABELS)
            or head.in_features != EMBEDDING_DIM
            or head.out_features != len(labels)
        ):
            raise ValueError("expected the Emotion2Vec+ Large 1024-D nine-class head")

    def predict(
        self,
        inputs: Sequence[str | Path | np.ndarray],
        include_embedding: bool = True,
    ) -> list[EmotionResult]:
        values = [str(value) if isinstance(value, Path) else value for value in inputs]
        records = self.model.generate(
            input=values,
            granularity="utterance",
            extract_embedding=include_embedding,
            disable_pbar=True,
            batch_size=self.batch_size,
        )
        if len(records) != len(values):
            raise ValueError("Emotion2Vec+ returned an incomplete batch")
        results = []
        for record in records:
            labels = [normalize_label(label) for label in record["labels"]]
            if labels != list(NATIVE_LABELS):
                raise ValueError(f"unexpected Emotion2Vec+ labels: {labels}")
            scores = np.asarray(record["scores"], dtype=np.float32)
            embedding = (
                np.asarray(record["feats"], dtype=np.float32).squeeze()
                if include_embedding
                else None
            )
            if (
                scores.shape != (len(NATIVE_LABELS),)
                or not np.isfinite(scores).all()
                or scores.sum() <= 0
            ):
                raise ValueError("invalid Emotion2Vec+ output")
            if include_embedding and (
                embedding is None
                or embedding.ndim != 1
                or embedding.shape != (EMBEDDING_DIM,)
                or not np.isfinite(embedding).all()
            ):
                raise ValueError("invalid Emotion2Vec+ embedding")
            results.append(EmotionResult(scores / scores.sum(), embedding))
        return results

    def extract_one(self, audio: Path, output: Path) -> None:
        if self.valid(output):
            return
        if output.exists():
            raise ValueError(f"existing Emotion2Vec+ feature is invalid: {output}")
        result = self.predict([audio])[0]
        self.save(output, result)

    @staticmethod
    def valid(output: Path) -> bool:
        if not output.is_file():
            return False
        try:
            with np.load(output, allow_pickle=False) as saved:
                return bool(
                    saved["labels"].shape == (9,)
                    and saved["scores"].shape == (9,)
                    and saved["embedding"].shape == (EMBEDDING_DIM,)
                    and np.isfinite(saved["scores"]).all()
                    and np.isfinite(saved["embedding"]).all()
                )
        except (OSError, ValueError, KeyError, EOFError):
            return False

    @staticmethod
    def save(output: Path, result: EmotionResult) -> None:
        if result.embedding is None:
            raise ValueError("preprocessing requires an Emotion2Vec+ embedding")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                labels=np.asarray(NATIVE_LABELS),
                scores=result.scores,
                embedding=result.embedding,
            )
        temporary.replace(output)

    def extract_many(
        self,
        jobs: Sequence[tuple[Path, Path]],
        progress_every: int = 100,
    ) -> None:
        pending = []
        for audio, output in jobs:
            if self.valid(output):
                continue
            if output.exists():
                raise ValueError(f"existing Emotion2Vec+ feature is invalid: {output}")
            pending.append((audio, output))
        completed = len(jobs) - len(pending)
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            results = self.predict([audio for audio, _ in batch])
            for (_, output), result in zip(batch, results):
                self.save(output, result)
            completed += len(batch)
            if completed % progress_every < len(batch) or completed == len(jobs):
                print(f"Emotion2Vec+ {completed:,}/{len(jobs):,}", flush=True)


def load_ser(
    backend: str = "emotion2vec",
    *,
    model: Path,
    device: str = "cuda",
    batch_size: int = 64,
) -> Emotion2Vec:
    """Explicitly load and reuse one instance per model, device, and batch size."""
    backend = backend.lower()
    if backend != "emotion2vec":
        raise ValueError(f"unsupported SER backend: {backend}")
    return _load_ser(backend, Path(model).resolve(), str(device), int(batch_size))


@lru_cache(maxsize=None)
def _load_ser(backend: str, model: Path, device: str, batch_size: int) -> Emotion2Vec:
    return Emotion2Vec(model, device=device, batch_size=batch_size)


def clear_ser_cache() -> None:
    """Drop cached instances; existing callers retain their loaded models."""
    _load_ser.cache_clear()
