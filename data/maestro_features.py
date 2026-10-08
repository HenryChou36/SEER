from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import atomic_array, atomic_text, sidecar


def statistics(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(7, dtype=np.float32)
    return np.asarray(
        [
            values.mean(),
            values.std(),
            values.min(),
            *np.quantile(values, (0.25, 0.50, 0.75)),
            values.max(),
        ],
        dtype=np.float32,
    )


def reference_feature(audio: Path) -> np.ndarray:
    emotion = np.load(
        sidecar(audio, "_emotion_diarization_embedding.npy"), allow_pickle=False
    )
    energy = np.load(sidecar(audio, "_smoothed_energy_nonor.npy"), allow_pickle=False)
    f0 = np.load(sidecar(audio, "_smoothed_f0_clean.npy"), allow_pickle=False)
    vuv = np.load(sidecar(audio, "_vuv_clean.npy"), allow_pickle=False)
    if emotion.ndim != 3 or emotion.shape[0] != 1 or emotion.shape[2] != 1024:
        raise ValueError(f"invalid MAESTRO emotion feature: {audio}")
    temporal = emotion[0].astype(np.float32)
    info = sf.info(audio)
    result = np.concatenate(
        (
            temporal.mean(axis=0),
            temporal.std(axis=0),
            statistics(energy),
            statistics(f0),
            np.asarray([np.asarray(vuv).astype(bool).mean(), info.duration]),
        )
    ).astype(np.float32)
    if result.shape != (2064,) or not np.isfinite(result).all():
        raise ValueError(f"invalid MAESTRO reference vector: {audio}")
    return result


class MaestroFeatures:
    """Resident extractors required by unmodified MAESTRO-EVC."""

    def __init__(
        self,
        upstream_root: Path,
        emotion_model: Path,
        speaker_model: Path,
        device: str = "cuda",
        cache_root: Path | None = None,
    ) -> None:
        if cache_root is None:
            raise ValueError("a feature-model cache directory is required")
        import joblib
        import torch
        import torchaudio
        from speechbrain.inference.diarization import Speech_Emotion_Diarization
        from speechbrain.inference.speaker import EncoderClassifier

        self.upstream_root = upstream_root.resolve()
        self.device = device
        self.torch = torch
        self.torchaudio = torchaudio
        cache = cache_root.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache / "torch_hub"))
        sys.path.insert(0, str(self.upstream_root))
        from extractors import f0 as upstream_f0
        from extractors import hubert as upstream_hubert

        self.f0 = upstream_f0
        self.hubert = upstream_hubert
        self.hubert_model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device).eval()
        self.kmeans = joblib.load(
            self.upstream_root / "hubert" / "hubert_base_ls960_L9_km500.bin"
        )
        self.emotion_model = Speech_Emotion_Diarization.from_hparams(
            source=str(emotion_model),
            savedir=str(cache / "emotion"),
            run_opts={"device": device},
        ).eval()
        self.speaker_model = EncoderClassifier.from_hparams(
            source=str(speaker_model),
            savedir=str(cache / "speaker"),
            run_opts={"device": device},
        ).eval()

    def extract_energy(self, audio: Path) -> None:
        import librosa
        from scipy.signal import savgol_filter

        first = sidecar(audio, "_mel_energy_nonor.npy")
        second = sidecar(audio, "_smoothed_energy_nonor.npy")
        if first.is_file() and second.is_file():
            return
        signal, sample_rate = librosa.load(audio, sr=16000, mono=True)
        frame = int(sample_rate * 0.01)
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_fft=frame,
            hop_length=frame,
            win_length=frame,
            n_mels=80,
            power=2.0,
            center=False,
        )
        energy = np.sqrt(np.mean(mel, axis=0) + 1e-9).astype(np.float32)
        energy[energy < 0.01] = 0.0
        window = 25
        if window > len(energy):
            window = max(3, (len(energy) // 2) * 2 + 1)
        smooth = (
            savgol_filter(energy, window, 2).astype(np.float32)
            if window >= 3 and window <= len(energy)
            else energy.copy()
        )
        atomic_array(first, energy)
        atomic_array(second, smooth)

    def extract_f0(self, audio: Path) -> None:
        import librosa

        paths = [
            sidecar(audio, "_nor_f0_clean.npy"),
            sidecar(audio, "_smoothed_f0_clean.npy"),
            sidecar(audio, "_vuv_clean.npy"),
        ]
        if all(path.is_file() for path in paths):
            return
        signal, _ = librosa.load(audio, sr=16000, mono=True)
        _, raw = self.f0.extract_f0_world(signal, 16000, method="dio", frame_period=10.0)
        value = self.f0.remove_outliers_iqr(raw)
        value = self.f0.remove_large_diff_neighbors(value, threshold_hz=50.0)
        value = self.f0.remove_isolated_f0_outliers(value, threshold_hz=50.0)
        value = self.f0.interpolate_short_zeros(value, max_gap=30)
        value, _, _ = self.f0.normalize_f0_mean(value)
        smooth = self.f0.savgol_smooth(value, window_size=25, polyorder=2)
        value = value.astype(np.float32)
        smooth = smooth.astype(np.float32)
        smooth[value == 0] = np.nan
        value[value == 0] = np.nan
        vuv = np.where(np.isnan(value), 0, 1).astype(np.uint8)
        for path, array in zip(paths, (value, smooth, vuv)):
            atomic_array(path, array)

    def extract_hubert(self, audio: Path) -> None:
        output = sidecar(audio, "_hubert.txt")
        if output.is_file():
            return
        with self.torch.inference_mode():
            signal = self.hubert.load_audio_mono_16k(str(audio), target_sr=16000)
            features = self.hubert.extract_hubert_layer_features(
                self.hubert_model, signal, layer=9, device=self.device
            )
            codes = self.hubert.kmeans_predict(self.kmeans, features)
        atomic_text(output, " ".join(map(str, codes.tolist())))

    def extract_emotion(self, audio: Path) -> None:
        output = sidecar(audio, "_emotion_diarization_embedding.npy")
        if output.is_file():
            return
        with self.torch.inference_mode():
            waveform = self.emotion_model.load_audio(str(audio))
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(self.device)
            lengths = self.torch.ones(waveform.shape[0], device=self.device)
            value = self.emotion_model.encode_batch(waveform, lengths)
        atomic_array(output, value.detach().cpu().numpy().astype(np.float32))

    def extract_speaker(self, audio: Path) -> None:
        output = sidecar(audio, "_speaker_embedding.npy")
        if output.is_file():
            return
        waveform, sample_rate = self.torchaudio.load(audio)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = self.torchaudio.functional.resample(
                waveform, sample_rate, 16000
            )
        waveform = waveform.to(self.device)
        with self.torch.inference_mode():
            value = self.speaker_model.encode_batch(
                waveform,
                wav_lens=self.torch.ones(waveform.shape[0], device=self.device),
            )
        value = value.detach().cpu().numpy().squeeze().astype(np.float32)
        if value.shape != (192,):
            raise ValueError(f"invalid MAESTRO speaker feature: {audio}")
        atomic_array(output, value)

    def extract(self, audio: Path) -> None:
        self.extract_energy(audio)
        self.extract_f0(audio)
        self.extract_hubert(audio)
        self.extract_emotion(audio)
        self.extract_speaker(audio)
        output = sidecar(audio, "_reference.npy")
        if not output.is_file():
            atomic_array(output, reference_feature(audio))
