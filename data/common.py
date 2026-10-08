from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


FEATURE_SUFFIXES = {
    "hubert_path": "_hubert.txt",
    "emotion_path": "_emotion_diarization_embedding.npy",
    "speaker_path": "_speaker_embedding.npy",
    "energy_path": "_mel_energy_nonor.npy",
    "smoothed_energy_path": "_smoothed_energy_nonor.npy",
    "f0_path": "_nor_f0_clean.npy",
    "smoothed_f0_path": "_smoothed_f0_clean.npy",
    "vuv_path": "_vuv_clean.npy",
    "reference_feature_path": "_reference.npy",
}

COLUMNS = (
    "dataset",
    "split",
    "audio_path",
    "processed_audio_path",
    *FEATURE_SUFFIXES,
    "emotion2vec_path",
    "utterance_id",
    "content_id",
    "speaker_id",
    "gender",
    "language",
    "emotion",
    "intensity",
    "outer_fold",
    "statement_id",
    "repetition",
    "transcript",
)


def sidecar(audio: Path, suffix: str) -> Path:
    return Path(str(audio.with_suffix("")) + suffix)


def atomic_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def materialize_audio(source: Path, target: Path, sample_rate: int = 16000) -> None:
    if target.is_file():
        info = sf.info(target)
        if info.samplerate == sample_rate and info.channels == 1 and info.frames > 0:
            return
        raise ValueError(f"existing processed audio is invalid: {target}")
    signal, observed_rate = sf.read(source, dtype="float32", always_2d=True)
    signal = signal.mean(axis=1, dtype=np.float32)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"invalid source audio: {source}")
    if observed_rate != sample_rate:
        divisor = math.gcd(observed_rate, sample_rate)
        signal = resample_poly(
            signal, sample_rate // divisor, observed_rate // divisor
        ).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.wav")
    sf.write(temporary, signal, sample_rate, subtype="PCM_16")
    temporary.replace(target)


def feature_locations(audio: Path, output_root: Path) -> dict[str, str]:
    locations = {
        name: sidecar(audio, suffix).relative_to(output_root).as_posix()
        for name, suffix in FEATURE_SUFFIXES.items()
    }
    relative = audio.relative_to(output_root)
    locations["emotion2vec_path"] = (
        output_root / "emotion2vec" / relative.with_suffix(".npz")
    ).relative_to(output_root).as_posix()
    return locations


def process_rows(
    rows: list[dict[str, object]],
    raw_root: Path,
    output_root: Path,
    maestro=None,
    emotion2vec=None,
    metadata_only: bool = False,
    progress_every: int = 100,
) -> list[dict[str, object]]:
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    result = []
    emotion_jobs: list[tuple[Path, Path]] = []
    for index, original in enumerate(rows, start=1):
        row = dict(original)
        source = raw_root / str(row["audio_path"])
        name = str(row.pop("processed_name", source.name))
        audio = (
            output_root
            / "audio"
            / str(row["split"])
            / str(row["speaker_id"])
            / str(row["emotion"])
            / name
        )
        locations = feature_locations(audio, output_root)
        current = {
            **row,
            "processed_audio_path": audio.relative_to(output_root).as_posix(),
            **locations,
        }
        if not metadata_only:
            materialize_audio(source, audio)
            maestro.extract(audio)
            emotion_jobs.append(
                (audio, output_root / str(locations["emotion2vec_path"]))
            )
        result.append(current)
        if index % progress_every == 0 or index == len(rows):
            print(f"prepared {index:,}/{len(rows):,}", flush=True)
    if emotion_jobs:
        emotion2vec.extract_many(emotion_jobs, progress_every=progress_every)
    return result


def write_metadata(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        raise ValueError("metadata is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def write_split_metadata(rows: list[dict[str, object]], output_root: Path) -> None:
    for split in ("train", "dev", "test"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            raise ValueError(f"metadata has no {split} rows")
        write_metadata(selected, output_root / f"{split}.csv")


def write_split_definition(output_root: Path, definition: dict[str, object]) -> None:
    output = output_root / "split.json"
    value = json.dumps(definition, indent=2, sort_keys=True) + "\n"
    if output.is_file():
        if output.read_text(encoding="utf-8") != value:
            raise ValueError(f"output already contains another split: {output}")
        return
    atomic_text(output, value)


def write_maestro_lists(
    rows: Iterable[dict[str, object]],
    output_root: Path,
    emotions: set[str],
) -> None:
    root = output_root.resolve()
    selected = [row for row in rows if str(row["emotion"]) in emotions]
    for split in ("train", "dev"):
        paths = sorted(
            str(root / str(row["processed_audio_path"]))
            for row in selected
            if row["split"] == split
        )
        atomic_text(root / f"{split}.txt", "\n".join(paths) + "\n")
