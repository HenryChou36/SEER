"""Evaluate speech emotion retention and derive conversion utility targets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isclose
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from seer.ci import CIConfig, simulate
from seer.ser import Emotion2Vec, EmotionResult, NATIVE_LABELS


@dataclass(frozen=True)
class RewardConfig:
    target_labels: tuple[str, ...]
    response_labels: tuple[str, ...]
    selected_channels: int = 8
    tie_scale: float = 0.1


@dataclass(frozen=True)
class RewardRequest:
    source_id: str
    source_path: Path
    intended_emotion: str
    reference_index: int | None
    reference_path: Path | None


@dataclass(frozen=True)
class RewardGroup:
    source_id: str
    source_path: Path
    intended_emotion: str
    references: tuple[tuple[int, Path], ...]


@dataclass(frozen=True)
class ActionOutcome:
    clean_native_prediction: str
    ci_native_prediction: str
    clean_native_scores: tuple[float, ...]
    ci_native_scores: tuple[float, ...]
    clean_prediction: str
    ci_prediction: str
    clean_scores: dict[str, float]
    ci_scores: dict[str, float]
    jointly_correct: bool
    gated_score: float


@dataclass(frozen=True)
class RewardTarget:
    reference_index: int
    reference: Path
    target: float
    primary: int
    secondary: float
    outcome: ActionOutcome


def _project(result: EmotionResult, labels: tuple[str, ...]) -> tuple[str, dict[str, float]]:
    return result.project(labels)


class RewardCache:
    """Small restart cache containing scores, never generated audio."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = None if path is None else path.resolve()
        self.table: dict[tuple[str, int | None, str], ActionOutcome] = {}
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        outcome = record["outcome"]
                        outcome["clean_native_scores"] = tuple(
                            outcome["clean_native_scores"]
                        )
                        outcome["ci_native_scores"] = tuple(outcome["ci_native_scores"])
                        self.table[self.key_from_record(record)] = ActionOutcome(**outcome)

    @staticmethod
    def key(request: RewardRequest) -> tuple[str, int | None, str]:
        return (
            request.source_id,
            request.reference_index,
            request.intended_emotion.casefold(),
        )

    @staticmethod
    def key_from_record(record: dict[str, object]) -> tuple[str, int | None, str]:
        return (
            str(record["source_id"]),
            record["reference_index"],
            str(record["intended_emotion"]),
        )

    def get(self, request: RewardRequest) -> ActionOutcome | None:
        return self.table.get(self.key(request))

    def add(self, rows: list[tuple[RewardRequest, ActionOutcome]]) -> None:
        if not rows:
            return
        records = []
        for request, outcome in rows:
            key = self.key(request)
            self.table[key] = outcome
            record = {
                "source_id": request.source_id,
                "intended_emotion": request.intended_emotion.casefold(),
                "reference_index": request.reference_index,
                "outcome": asdict(outcome),
            }
            records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")


class RewardModel:
    """Score 16-kHz waveforms with a resident SER model and CI simulator."""

    def __init__(
        self,
        evaluator: Emotion2Vec,
        config: RewardConfig,
        ci_config: CIConfig = CIConfig(),
        cache_path: Path | None = None,
    ) -> None:
        targets = {label.casefold() for label in config.target_labels}
        responses = {label.casefold() for label in config.response_labels}
        if not targets or len(targets) != len(config.target_labels):
            raise ValueError("target label inventory is empty or duplicated")
        if len(responses) != len(config.response_labels) or not targets.issubset(
            responses
        ):
            raise ValueError("SER response labels must uniquely include every target")
        if not 0 <= config.tie_scale < 1:
            raise ValueError("tie_scale must be in [0, 1)")
        if ci_config.sample_rate != 16_000:
            raise ValueError("reward waveforms must use the 16-kHz SER sample rate")
        self.evaluator = evaluator
        self.config = config
        self.ci_config = ci_config
        self.cache = RewardCache(cache_path)

    def score(
        self,
        waveforms: Iterable[np.ndarray],
        intended_emotions: Iterable[str],
    ) -> list[ActionOutcome]:
        """Evaluate already produced speech in clean and CI-simulated views."""

        waveforms = list(waveforms)
        labels = [emotion.casefold() for emotion in intended_emotions]
        if len(waveforms) != len(labels):
            raise ValueError("every reward waveform needs an intended emotion")
        targets = {label.casefold() for label in self.config.target_labels}
        if any(label not in targets for label in labels):
            raise ValueError("intended emotion is outside the enhancement targets")
        views: list[np.ndarray] = []
        for waveform in waveforms:
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                raise ValueError("reward audio must be finite, nonempty mono speech")
            views.extend(
                (
                    waveform,
                    simulate(waveform, self.config.selected_channels, self.ci_config),
                )
            )
        predictions = (
            self.evaluator.predict(views, include_embedding=False) if views else []
        )
        if len(predictions) != len(views):
            raise ValueError("SER returned an incomplete reward batch")
        outcomes = []
        for offset, label in enumerate(labels):
            clean_result, ci_result = predictions[2 * offset : 2 * offset + 2]
            clean_prediction, clean_scores = _project(
                clean_result, self.config.response_labels
            )
            ci_prediction, ci_scores = _project(
                ci_result, self.config.response_labels
            )
            clean_correct = clean_prediction == label
            outcome = ActionOutcome(
                clean_native_prediction=NATIVE_LABELS[int(clean_result.scores.argmax())],
                ci_native_prediction=NATIVE_LABELS[int(ci_result.scores.argmax())],
                clean_native_scores=tuple(float(value) for value in clean_result.scores),
                ci_native_scores=tuple(float(value) for value in ci_result.scores),
                clean_prediction=clean_prediction,
                ci_prediction=ci_prediction,
                clean_scores=clean_scores,
                ci_scores=ci_scores,
                jointly_correct=clean_correct and ci_prediction == label,
                gated_score=ci_scores[label] if clean_correct else 0.0,
            )
            outcomes.append(outcome)
        return outcomes

    def utility(
        self, original: ActionOutcome, converted: ActionOutcome
    ) -> tuple[float, int, float]:
        """Return utility, correctness difference, and gated score difference."""

        primary = int(converted.jointly_correct) - int(original.jointly_correct)
        secondary = converted.gated_score - original.gated_score
        return primary + self.config.tie_scale * secondary, primary, secondary


@dataclass(frozen=True)
class EvaluatedPair:
    source_index: int
    reference_index: int
    target: float
    primary: int


class EvaluationHistory:
    """All distinct evaluated pairs, retaining deterministic utility targets."""

    def __init__(self) -> None:
        self._records: dict[tuple[int, int], EvaluatedPair] = {}

    def add(self, records: list[EvaluatedPair]) -> None:
        for record in records:
            key = (record.source_index, record.reference_index)
            previous = self._records.get(key)
            if previous is not None:
                if previous.primary != record.primary or not isclose(
                    previous.target, record.target, rel_tol=0.0, abs_tol=1e-8
                ):
                    raise ValueError("immutable evaluated-pair outcome changed")
                continue
            self._records[key] = record

    def sample(
        self,
        count: int,
        rng: np.random.Generator,
        exclude: set[tuple[int, int]] | None = None,
        *,
        keys: Sequence[tuple[int, int]] | None = None,
    ) -> list[EvaluatedPair]:
        if count < 0:
            raise ValueError("history sample count must be nonnegative")
        excluded = exclude or set()
        available = [
            record
            for record in self.records(keys)
            if (record.source_index, record.reference_index) not in excluded
        ]
        count = min(count, len(available))
        if not count:
            return []
        selected: list[EvaluatedPair] = []
        quota = count // 3
        for primary in (-1, 0, 1):
            group = [record for record in available if record.primary == primary]
            if group:
                indices = rng.choice(len(group), min(quota, len(group)), replace=False)
                selected.extend(group[int(index)] for index in np.atleast_1d(indices))
        selected_keys = {
            (record.source_index, record.reference_index) for record in selected
        }
        remaining = [
            record
            for record in available
            if (record.source_index, record.reference_index) not in selected_keys
        ]
        needed = count - len(selected)
        if needed:
            indices = rng.choice(len(remaining), needed, replace=False)
            selected.extend(remaining[int(index)] for index in np.atleast_1d(indices))
        return selected

    def records(
        self, keys: Sequence[tuple[int, int]] | None = None
    ) -> list[EvaluatedPair]:
        if keys is None:
            return list(self._records.values())
        return [self._records[key] for key in keys]

    def state_dict(self) -> dict[str, object]:
        return {"records": [asdict(record) for record in self._records.values()]}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._records.clear()
        self.add([EvaluatedPair(**record) for record in state["records"]])
