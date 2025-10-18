"""Audio synthesis utilities for FTIR-to-sound conversion."""
from __future__ import annotations

import io
from typing import Iterable, List, Sequence, Tuple

import numpy as np

SPEED_OF_LIGHT = 2.99792458e10  # cm/s


def wavenumber_to_frequency_cm1(wavenumber: float) -> float:
    """Convert a wavenumber in cm⁻¹ to frequency in Hz."""

    return wavenumber * SPEED_OF_LIGHT


def map_wavenumber_to_audible(
    wavenumber: float,
    wn_range: Tuple[float, float] = (400.0, 4000.0),
    audible_range: Tuple[float, float] = (220.0, 1760.0),
) -> float:
    """Map an IR wavenumber to an audible frequency via linear scaling."""

    wn_min, wn_max = wn_range
    audio_min, audio_max = audible_range
    clamped = max(min(wavenumber, wn_max), wn_min)
    scale = (clamped - wn_min) / (wn_max - wn_min)
    return audio_min + scale * (audio_max - audio_min)


def generate_waveform(
    frequencies: Sequence[float],
    duration: float = 2.0,
    sample_rate: int = 44100,
    envelope: str = "hann",
) -> np.ndarray:
    """Generate a waveform by summing sine waves at ``frequencies``."""

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)
    for freq in frequencies:
        waveform += np.sin(2 * np.pi * freq * t)

    if envelope == "hann":
        window = np.hanning(len(t))
        waveform *= window

    # Normalise to prevent clipping
    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp
    return waveform.astype(np.float32)


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int = 44100) -> bytes:
    """Encode a waveform as WAV bytes."""

    from scipy.io import wavfile

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, waveform)
    return buffer.getvalue()


def groups_to_audio_frequencies(matches: Iterable) -> List[float]:
    """Convert functional group matches to audible frequencies."""

    freqs: List[float] = []
    for match in matches:
        if not getattr(match, "present", False):
            continue
        center = match.group.center_wavenumber
        audio_freq = map_wavenumber_to_audible(center)
        freqs.append(audio_freq)
    return freqs
