from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_id(row: dict[str, str]) -> str:
    return str(
        row.get("source_id")
        or f"{row['dataset']}-{row['speaker_id']}-{row['utterance_id']}"
    )


def csv_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"expected a CSV boolean, found {value!r}")
    return value == "True"


def selector_rows(
    rows: list[dict[str, str]], config: dict[str, object], split: str | None = None
) -> list[dict[str, str]]:
    """Apply the configured target-emotion and optional intensity protocol."""
    labels = {str(value).casefold() for value in config["emotion_labels"]}
    intensities = {
        str(value).strip().casefold()
        for value in config.get("source_intensities", [])
        if str(value).strip()
    }
    selected = []
    for row in rows:
        if str(row.get("emotion", "")).casefold() not in labels:
            continue
        if split is not None and row.get("split") != split:
            continue
        intensity = str(row.get("intensity", "")).strip().casefold()
        if intensities and intensity not in intensities:
            continue
        selected.append(row)
    return selected


def test_rows(
    rows: list[dict[str, str]], config: dict[str, object], source_intensity: str | None = None
) -> list[dict[str, str]]:
    """Validate the complete configured test population before objective scoring."""
    selected = selector_rows(rows, config, split="test")
    expected = int(config["source_counts"]["test"])
    if source_intensity is not None:
        if config.get("dataset") != "ravdess" or source_intensity not in {"normal", "strong"}:
            raise ValueError("source-intensity is supported only for RAVDESS normal/strong")
        intensities = tuple(config.get("source_intensities", ()))
        if set(intensities) != {"normal", "strong"} or expected % 2:
            raise ValueError("RAVDESS objective subsets require both configured intensities")
        selected = [
            row for row in selected if row.get("intensity", "").casefold() == source_intensity
        ]
        expected //= 2
    if len(selected) != expected or len({source_id(row) for row in selected}) != expected:
        raise ValueError("test inventory differs from the configuration or has duplicate sources")
    return selected


def enhanced_inventory(
    enhanced_root: Path, sources: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Bind a complete enhancement export to the requested source population."""
    root = enhanced_root.resolve()
    selections = load_csv(root / "selections.csv")
    identifiers = [source_id(row) for row in sources]
    selected_ids = [row["source_id"] for row in selections]
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) != set(identifiers):
        raise ValueError("enhanced selections do not match the complete requested test population")
    result = {}
    for row in selections:
        relative = Path(row["output_audio_path"])
        audio = (root / relative).resolve()
        if relative.is_absolute() or not audio.is_relative_to(root) or not audio.is_file():
            raise ValueError(f"missing or invalid enhanced audio for {row['source_id']}")
        enhanced = csv_bool(row["enhanced"])
        reference = row["reference_index"]
        if enhanced != bool(reference) or bool(reference) != bool(row["reference_path"]):
            raise ValueError(f"inconsistent enhancement selection for {row['source_id']}")
        if reference and (not reference.isdecimal() or int(reference) < 0):
            raise ValueError(f"invalid selected reference for {row['source_id']}")
        result[row["source_id"]] = row
    return result


def read_waveform(path: Path) -> np.ndarray:
    import soundfile as sf

    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16_000 or values.shape[1] != 1:
        raise ValueError(f"objective evaluation requires mono 16-kHz audio: {path}")
    waveform = values[:, 0]
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid objective waveform: {path}")
    return waveform


class SourceFeatures:
    def __init__(self, root: Path, rows: list[dict[str, str]], capacity: int = 512) -> None:
        self.root = root.resolve()
        self.rows = rows
        self.capacity = capacity
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def __getitem__(self, index: int) -> torch.Tensor:
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value
        path = self.root / self.rows[index]["emotion_path"]
        values = np.load(path, allow_pickle=False)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape[1] != 1024 or not np.isfinite(values).all():
            raise ValueError(f"invalid source feature: {path}")
        value = torch.from_numpy(values.astype(np.float32, copy=False))
        self.cache[index] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value


def reference_inventory(
    rows: list[dict[str, str]], root: Path, labels: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor, list[Path]]:
    mapping = {label.casefold(): index for index, label in enumerate(labels)}
    features, emotions, paths = [], [], []
    for row in rows:
        label = row["emotion"].casefold()
        if label not in mapping:
            raise ValueError("reference bank contains an unconfigured emotion")
        path = root / row["reference_feature_path"]
        value = np.load(path, allow_pickle=False)
        if value.shape != (2064,) or not np.isfinite(value).all():
            raise ValueError(f"invalid reference feature: {path}")
        features.append(value.astype(np.float32, copy=False))
        emotions.append(mapping[label])
        paths.append(root / row["processed_audio_path"])
    return (
        torch.from_numpy(np.stack(features)),
        torch.tensor(emotions, dtype=torch.long),
        paths,
    )
