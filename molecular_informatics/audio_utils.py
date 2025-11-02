"""Audio synthesis utilities for FTIR-to-sound conversion."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

SPEED_OF_LIGHT = 2.99792458e10  # cm/s
AudioComponent = Tuple[float, float]


@dataclass(frozen=True)
class MedicinalSoundProfile:
    """Capture medicinal chemistry-inspired toggles for shaping the audio."""

    activity: Union[str, None] = None  # "active" or "inactive"
    activity_strength: Union[float, None] = None  # 0..1 scaling
    selectivity: Union[str, None] = None  # "selective"
    selectivity_strength: Union[float, None] = None
    toxicity: Union[str, None] = None  # "toxic"
    toxicity_strength: Union[float, None] = None
    bioavailability: Union[str, None] = None  # "bioavailable"
    bioavailability_strength: Union[float, None] = None

    def is_neutral(self) -> bool:
        """Return True when no medicinal sound modifiers are requested."""

        fields = (
            self.activity,
            self.activity_strength,
            self.selectivity,
            self.selectivity_strength,
            self.toxicity,
            self.toxicity_strength,
            self.bioavailability,
            self.bioavailability_strength,
        )
        return all(value in (None, 0.0) for value in fields)


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


def apply_medicinal_sound_effects(
    waveform: np.ndarray, profile: MedicinalSoundProfile, sample_rate: int
) -> np.ndarray:
    """Apply simple DSP effects themed around medicinal chemistry heuristics."""

    modified = waveform.astype(np.float32, copy=True)

    # Activity: emphasise or mute the signal.
    if profile.activity == "active":
        strength = float(profile.activity_strength or 1.0)
        drive = 1.0 + 1.6 * np.clip(strength, 0.0, 1.0)
        modified = np.tanh(drive * modified)
    elif profile.activity == "inactive":
        strength = float(profile.activity_strength or 1.0)
        attenuation = 1.0 - 0.95 * np.clip(strength, 0.0, 1.0)
        modified *= attenuation

    # Selectivity: a gentler tone courtesy of a small smoothing filter.
    if profile.selectivity == "selective":
        strength = float(profile.selectivity_strength or 1.0)
        span = max(1, int(2 + np.clip(strength, 0.0, 1.0) * 4))
        window_length = span * 2 + 1
        smoothing_kernel = np.hanning(window_length).astype(np.float32)
        smoothing_kernel /= smoothing_kernel.sum()
        modified = np.convolve(modified, smoothing_kernel, mode="same")

    # Toxicity: introduce harmonics by driving the signal into soft clipping.
    if profile.toxicity == "toxic":
        strength = float(profile.toxicity_strength or 1.0)
        drive = 1.0 + 3.2 * np.clip(strength, 0.0, 1.0)
        modified = np.tanh(drive * modified)

    # Bioavailability: longer moving average for a smoother feel.
    if profile.bioavailability == "bioavailable":
        strength = float(profile.bioavailability_strength or 1.0)
        window_seconds = 0.003 + 0.015 * np.clip(strength, 0.0, 1.0)
        max_window = min(int(sample_rate * window_seconds), len(modified))
        window = max(3, max_window)
        if window % 2 == 0:
            window += 1  # keep the window symmetric
        kernel = np.ones(window, dtype=np.float32)
        kernel /= kernel.sum()
        modified = np.convolve(modified, kernel, mode="same")

    return modified.astype(np.float32, copy=False)


def generate_waveform(
    frequencies: Sequence[Union[float, AudioComponent]],
    duration: float = 2.0,
    sample_rate: int = 44100,
    envelope: str = "hann",
    medicinal_profile: Union[MedicinalSoundProfile, None] = None,
) -> np.ndarray:
    """Generate a waveform by summing sine waves at ``frequencies``."""

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)
    for component in frequencies:
        if isinstance(component, tuple):
            freq, amplitude = component
        else:
            freq, amplitude = component, 1.0
        waveform += amplitude * np.sin(2 * np.pi * freq * t)

    if envelope == "hann":
        window = np.hanning(len(t))
        waveform *= window

    # Normalise to prevent clipping
    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp

    if medicinal_profile and not medicinal_profile.is_neutral():
        waveform = apply_medicinal_sound_effects(waveform, medicinal_profile, sample_rate)
        post_amp = np.max(np.abs(waveform))
        if post_amp > 1.0:
            waveform = waveform / post_amp

    return waveform.astype(np.float32, copy=False)


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int = 44100) -> bytes:
    """Encode a waveform as WAV bytes."""

    from scipy.io import wavfile

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, waveform)
    return buffer.getvalue()


def groups_to_audio_components(matches: Iterable) -> List[AudioComponent]:
    """Convert functional group matches to (frequency, amplitude) components."""

    components: List[AudioComponent] = []
    filtered = [m for m in matches if getattr(m, "present", False)]
    total_occurrences = sum(getattr(m, "match_count", 0) for m in filtered)
    if total_occurrences == 0:
        return components

    for match in filtered:
        center = match.group.center_wavenumber
        audio_freq = map_wavenumber_to_audible(center)
        weight = match.match_count / total_occurrences if total_occurrences else 0.0
        components.append((audio_freq, weight))
    return components
