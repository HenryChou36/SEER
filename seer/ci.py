"""CI24-PSHC acoustic simulation used by the SEER reward.

SPDX-License-Identifier: GPL-3.0-or-later
Adapted from ``egaudrain/vocoder`` (``vocode.m``,
``get_pshc.m``, ``GenCarrier2.m`` and associated envelope helpers).
The original code is Copyright LMA CNRS, CRNL CNRS, UMCG, Olivier Macherey
and Etienne Gaudrain and is distributed under GPL-3.0-or-later.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy import signal


CI24_EDGES_HZ = (
    188,
    313,
    438,
    563,
    688,
    813,
    938,
    1063,
    1188,
    1313,
    1563,
    1813,
    2063,
    2313,
    2688,
    3063,
    3563,
    4063,
    4688,
    5313,
    6063,
    6938,
    7938,
)


@dataclass(frozen=True)
class CIConfig:
    sample_rate: int = 16_000
    edges_hz: tuple[float, ...] = CI24_EDGES_HZ
    bandpass_order: int = 3
    envelope_lowpass_hz: float = 250.0
    envelope_order: int = 2
    selection_rate_hz: int = 900
    pshc_f0_hz: int = 1
    seed: int = 20_260_910
    match_input_rms: bool = True

    @property
    def available_channels(self) -> int:
        return len(self.edges_hz) - 1


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _phases(harmonics: np.ndarray, groups: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(groups)
    offsets = rng.rand(groups)
    group = (-harmonics) % groups
    return 2 * np.pi * (
        harmonics * permutation[group] / groups**2 + offsets[group]
    )


@lru_cache(maxsize=128)
def _carrier_period(
    lower_hz: float,
    upper_hz: float,
    sample_rate: int,
    f0_hz: int,
    seed: int,
) -> np.ndarray:
    if f0_hz != 1:
        raise ValueError("the validated SEER PSHC recipe uses a 1-Hz fundamental")
    harmonics = np.arange(
        int(np.floor(lower_hz + 0.5)), int(np.floor(upper_hz + 0.5)) + 1
    )
    if harmonics[0] <= 0 or harmonics[-1] >= sample_rate / 2:
        raise ValueError("carrier harmonics must remain inside the Nyquist interval")
    center_khz = np.sqrt(lower_hz * upper_hz) / 1000
    groups = int(
        np.floor(np.polyval([0.0459, -0.7964, 6.9097, 7.2244], center_khz) + 0.5)
    )
    phases = _phases(harmonics, groups, seed)
    spectrum = np.zeros(sample_rate // 2 + 1, dtype=complex)
    spectrum[harmonics] = (
        -0.5j * sample_rate * np.exp(1j * phases) / len(harmonics)
    )
    period = np.fft.irfft(spectrum, n=sample_rate)
    period = np.roll(period, -(int(np.floor(sample_rate / 4 + 0.5)) + 1))
    period = 0.9 * period / np.max(np.abs(period))
    period.setflags(write=False)
    return period


def _select(envelopes: np.ndarray, count: int, config: CIConfig) -> np.ndarray:
    reduced = signal.resample_poly(
        envelopes, config.selection_rate_hz, config.sample_rate, axis=0
    )
    rejected = np.argsort(reduced, axis=1, kind="stable")[:, :-count]
    selected = reduced.copy()
    np.put_along_axis(selected, rejected, 0, axis=1)
    full_times = np.arange(len(envelopes)) / config.sample_rate
    reduced_times = np.linspace(full_times[0], full_times[-1], len(selected))
    return np.column_stack(
        [np.interp(full_times, reduced_times, column) for column in selected.T]
    )


def simulate_many(
    audio: np.ndarray,
    selected_channels: Iterable[int] = (4, 8, 16),
    config: CIConfig = CIConfig(),
) -> dict[int, np.ndarray]:
    """Return one PSHC waveform per requested N-of-M condition.

    ``M`` is determined by ``config.edges_hz`` (22 active analysis bands in the
    CI24 allocation used here). All requested conditions reuse the same
    envelopes and carrier realization.
    """

    counts = tuple(int(value) for value in selected_channels)
    if not counts or len(counts) != len(set(counts)):
        raise ValueError("selected channel counts must be nonempty and unique")
    if any(value < 1 or value > config.available_channels for value in counts):
        raise ValueError("selected channel count exceeds the analysis bank")
    values = np.asarray(audio, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 100
        or not np.isfinite(values).all()
        or _rms(values) == 0
    ):
        raise ValueError("audio must be finite, nonzero mono speech")

    edges = np.asarray(config.edges_hz, dtype=np.float64)
    filters = [
        signal.butter(
            config.bandpass_order,
            [lower, upper],
            btype="bandpass",
            fs=config.sample_rate,
            output="sos",
        )
        for lower, upper in zip(edges[:-1], edges[1:])
    ]
    broad = signal.butter(
        config.bandpass_order,
        [edges[0], edges[-1]],
        btype="bandpass",
        fs=config.sample_rate,
        output="sos",
    )
    target_rms = _rms(signal.sosfiltfilt(broad, values, padlen=18))
    lowpass = signal.butter(
        config.envelope_order,
        config.envelope_lowpass_hz,
        fs=config.sample_rate,
        output="sos",
    )
    envelopes = np.column_stack(
        [
            np.maximum(
                signal.sosfiltfilt(
                    lowpass,
                    np.maximum(signal.sosfiltfilt(current, values, padlen=18), 0),
                    padlen=6,
                ),
                0,
            )
            for current in filters
        ]
    )
    active = {count: _select(envelopes, count, config) for count in counts}
    outputs = {count: np.zeros(values.size, dtype=np.float64) for count in counts}
    offsets = np.random.RandomState(config.seed)
    for index, current in enumerate(filters):
        period = _carrier_period(
            float(edges[index]),
            float(edges[index + 1]),
            config.sample_rate,
            config.pshc_f0_hz,
            config.seed,
        )
        repeats = int(np.ceil(values.size / config.sample_rate)) + 1
        offset = offsets.randint(0, repeats * config.sample_rate - values.size + 1)
        excitation = period[(np.arange(values.size) + offset) % config.sample_rate]
        for count in counts:
            outputs[count] += signal.sosfiltfilt(
                current, active[count][:, index] * excitation, padlen=18
            )

    for count, output in outputs.items():
        if not np.isfinite(output).all() or _rms(output) == 0:
            raise ValueError("CI simulation produced an invalid waveform")
        if config.match_input_rms:
            output = output / _rms(output) * target_rms
        outputs[count] = output.astype(np.float32)
    return outputs


def simulate(
    audio: np.ndarray,
    selected_channels: int = 8,
    config: CIConfig = CIConfig(),
) -> np.ndarray:
    """Simulate one N-of-M CI24-PSHC condition."""

    return simulate_many(audio, (selected_channels,), config)[selected_channels]
