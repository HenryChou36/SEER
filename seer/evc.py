"""Reusable EVC interface and persistent Maestro-EVC conversion backend."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import savgol_filter


F0_STD = 46.9128


class EVCBackend(Protocol):
    """The conversion operation required by SEER."""

    def convert(self, source: Path, reference: Path) -> np.ndarray: ...


class EVC:
    """Use one loaded backend for successive source--reference conversions."""

    def __init__(self, backend: EVCBackend) -> None:
        self.backend = backend

    def convert(self, source: Path, reference: Path) -> np.ndarray:
        return self.backend.convert(Path(source), Path(reference))

    def __call__(self, source: Path, reference: Path) -> np.ndarray:
        return self.convert(source, reference)

    def render(self, source: Path, reference: Path | None = None) -> np.ndarray:
        """Return a conversion, or the original waveform when no reference is used."""
        if reference is not None:
            return self.convert(source, reference)
        waveform, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        if sample_rate != 16_000:
            raise ValueError(f"SEER requires 16-kHz speech: {source}")
        waveform = waveform.mean(axis=1)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(f"invalid source audio: {source}")
        return waveform


def _sidecar(audio: Path, suffix: str) -> Path:
    return Path(str(audio.with_suffix("")) + suffix)


def _array(audio: Path, suffix: str) -> np.ndarray:
    path = _sidecar(audio, suffix)
    values = np.load(path, allow_pickle=False)
    if values.size == 0 or np.isinf(values).any() or np.isnan(values).all():
        raise ValueError(f"invalid MAESTRO feature: {path}")
    return values


def _codes(audio: Path) -> np.ndarray:
    path = _sidecar(audio, "_hubert.txt")
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.int64)
    if values.size == 0 or values.min() < 0 or values.max() >= 500:
        raise ValueError(f"invalid HuBERT codes: {path}")
    return values


def _repeat_codes(audio: Path) -> np.ndarray:
    info = sf.info(audio)
    if info.samplerate != 16_000 or info.channels != 1 or info.frames <= 0:
        raise ValueError(f"MAESTRO requires nonempty mono 16-kHz input: {audio}")
    values = _codes(audio)
    frames = min(info.frames // 320, len(values))
    if frames < 2:
        raise ValueError(f"too few aligned HuBERT frames: {audio}")
    return np.repeat(values[:frames], 2)


def _fit(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values).squeeze()
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"expected a nonempty vector, found {values.shape}")
    return values[:length] if len(values) >= length else np.pad(
        values, (0, length - len(values)), mode="edge"
    )


def _segments(codes: Sequence[int]) -> tuple[list[int], list[int], list[int], list[np.ndarray]]:
    values = np.asarray(codes, dtype=np.int64)
    unique = [int(values[0])]
    durations: list[int] = []
    indices = [0]
    count = 1
    for index, value in enumerate(values[1:], start=1):
        if int(value) == unique[-1]:
            count += 1
        else:
            durations.append(count)
            unique.append(int(value))
            indices.append(index)
            count = 1
    durations.append(count)
    smooth = savgol_filter(
        np.asarray(durations, dtype=np.float32), 5, 2, mode="nearest"
    ).astype(np.float32)
    return unique, durations, indices, [smooth]


def _inputs(source: Path, reference: Path) -> tuple[dict[str, np.ndarray], list[tuple]]:
    source_codes = _repeat_codes(source)
    source_vuv = _fit(_array(source, "_vuv_clean.npy"), len(source_codes)).astype(
        np.int64
    )
    reference_codes = _repeat_codes(reference)
    arrays = [
        _array(reference, "_smoothed_energy_nonor.npy"),
        _array(reference, "_mel_energy_nonor.npy"),
        _array(reference, "_smoothed_f0_clean.npy"),
        _array(reference, "_nor_f0_clean.npy"),
        _array(reference, "_vuv_clean.npy"),
    ]
    length = min(
        len(reference_codes), *(len(np.asarray(value).squeeze()) for value in arrays)
    )
    if length < 2:
        raise ValueError(f"too few aligned reference frames: {reference}")
    reference_codes = reference_codes[:length]
    smooth_energy, energy, smooth_f0, f0, vuv = [
        _fit(value, length) for value in arrays
    ]
    emotion = _array(reference, "_emotion_diarization_embedding.npy")
    speaker = _array(source, "_speaker_embedding.npy").squeeze()
    if emotion.ndim != 3 or emotion.shape[0] != 1 or emotion.shape[2] != 1024:
        raise ValueError(f"invalid temporal emotion embedding: {reference}")
    if speaker.shape != (192,):
        raise ValueError(f"invalid speaker embedding: {source}")
    return {
        "code": source_codes.astype(np.int64),
        "original_code": source_codes[::2].astype(np.int64),
        "speaker_embedding": speaker.astype(np.float32),
        "full_emotion_diarization_embedding": emotion[0].astype(np.float32),
        "smoothed_energy": smooth_energy.astype(np.float32),
        "energy": energy.astype(np.float32),
        "smoothed_f0": (smooth_f0 / F0_STD).astype(np.float32),
        "f0": (f0 / F0_STD).astype(np.float32),
        "vuv": vuv.astype(np.int64),
        "src_vuv": source_vuv,
    }, [_segments(reference_codes)]


class MaestroEVC:
    """Load MAESTRO once and convert many source/reference pairs."""

    def __init__(
        self,
        upstream_root: Path,
        checkpoint: Path,
        config: Path,
        device: str = "cuda",
    ) -> None:
        import torch

        upstream_root = upstream_root.resolve()
        sys.path.insert(0, str(upstream_root))
        from models import CodeGenerator
        from utils import AttrDict

        self.torch = torch
        self.device = torch.device(device)
        settings = AttrDict(json.loads(config.read_text(encoding="utf-8")))
        self.generator = CodeGenerator(settings, "inference").to(self.device)
        state = torch.load(checkpoint, map_location="cpu")
        self.generator.load_state_dict(state["generator"])
        self.generator.eval()
        self.generator.remove_weight_norm()

    def convert(self, source: Path, reference: Path) -> np.ndarray:
        features, unique_emotion = _inputs(source.resolve(), reference.resolve())
        tensors = {
            key: self.torch.from_numpy(value).unsqueeze(0).to(self.device)
            for key, value in features.items()
        }
        with self.torch.inference_mode():
            generated = self.generator(
                target_energy=None,
                target_f0=None,
                unique_emo=unique_emotion,
                **tensors,
            )[0]
        output = generated[0].squeeze().detach().float().cpu().numpy()
        if output.ndim != 1 or output.size == 0 or not np.isfinite(output).all():
            raise ValueError("MAESTRO produced invalid audio")
        peak = float(np.max(np.abs(output)))
        if peak > 1e-12:
            output = output / peak
        return np.clip(output, -1.0, 1.0).astype(np.float32)


def load_evc(
    backend: str = "maestro",
    *,
    upstream_root: Path,
    checkpoint: Path,
    config: Path,
    device: str = "cuda",
) -> EVC:
    """Load once per model specification and device, then reuse the same instance.

    Loading is explicit rather than happening when this module is imported.
    Checkpoints should have distinct paths; call ``clear_evc_cache`` to reload
    a checkpoint deliberately replaced at the same path.
    """
    backend = backend.lower()
    if backend != "maestro":
        raise ValueError(f"unsupported EVC backend: {backend}")
    return _load_evc(
        backend,
        Path(upstream_root).resolve(),
        Path(checkpoint).resolve(),
        Path(config).resolve(),
        str(device),
    )


@lru_cache(maxsize=None)
def _load_evc(
    backend: str,
    upstream_root: Path,
    checkpoint: Path,
    config: Path,
    device: str,
) -> EVC:
    return EVC(MaestroEVC(upstream_root, checkpoint, config, device=device))


def clear_evc_cache() -> None:
    """Release the wrapper cache; existing callers retain their loaded models."""
    _load_evc.cache_clear()
