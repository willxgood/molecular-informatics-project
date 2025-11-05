"""Audio effect processors for the molecular piano roll."""
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt, sosfiltfilt


def apply_gain(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a gain change in decibels to ``waveform``."""

    if not np.isfinite(gain_db) or gain_db == 0.0:
        return waveform
    factor = 10 ** (gain_db / 20.0)
    return (waveform * factor).astype(np.float32)


def apply_lowpass_filter(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter with the provided ``cutoff``."""

    if cutoff <= 0:
        raise ValueError("Cutoff frequency must be positive.")

    nyquist = 0.5 * sample_rate
    normalised = min(cutoff / nyquist, 0.999)
    sos = butter(order, normalised, btype="low", output="sos")

    # scipy sosfiltfilt requires enough samples; fall back gracefully for short clips
    section_length = max(len(section) for section in sos)
    padlen = 3 * (section_length - 1)
    if waveform.size <= padlen:
        filtered = sosfilt(sos, waveform)
    else:
        filtered = sosfiltfilt(sos, waveform)
    return filtered.astype(np.float32)


def apply_highpass_filter(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    cutoff: float,
    order: int = 2,
) -> np.ndarray:
    """Apply a Butterworth high-pass filter with the provided ``cutoff``."""

    if cutoff <= 0:
        raise ValueError("Cutoff frequency must be positive.")

    nyquist = 0.5 * sample_rate
    normalised = min(max(cutoff / nyquist, 1e-4), 0.999)
    sos = butter(order, normalised, btype="high", output="sos")

    section_length = max(len(section) for section in sos)
    padlen = 3 * (section_length - 1)
    if waveform.size <= padlen:
        filtered = sosfilt(sos, waveform)
    else:
        filtered = sosfiltfilt(sos, waveform)
    return filtered.astype(np.float32)


def apply_delay(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    delay_seconds: float,
    feedback: float = 0.3,
    mix: float = 0.25,
) -> np.ndarray:
    """Apply a feedback delay/echo effect to ``waveform``."""

    if delay_seconds <= 0:
        return waveform

    delay_samples = int(delay_seconds * sample_rate)
    if delay_samples <= 0:
        return waveform

    output = np.copy(waveform)
    delayed = np.zeros(len(waveform) + delay_samples, dtype=np.float32)
    delayed[: len(waveform)] = waveform

    for idx in range(delay_samples, len(delayed)):
        delayed[idx] += feedback * delayed[idx - delay_samples]

    delayed = delayed[: len(output)]
    return ((1 - mix) * output + mix * delayed).astype(np.float32)


def apply_distortion(waveform: np.ndarray, *, drive: float = 1.0) -> np.ndarray:
    """Apply a tanh-based waveshaper with configurable ``drive``."""

    drive = max(drive, 0.0)
    if drive == 0:
        return waveform
    shaped = np.tanh(waveform * (1.0 + drive * 9.0))
    max_amp = np.max(np.abs(shaped))
    if max_amp > 0:
        shaped = shaped / max_amp
    return shaped.astype(np.float32)


def apply_reverb(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    room_size: float = 0.4,
    decay: float = 0.5,
    wet: float = 0.25,
) -> np.ndarray:
    """Apply a simple convolution reverb using an exponentially decaying impulse."""

    wet = max(0.0, min(1.0, wet))
    if wet == 0.0:
        return waveform

    room_size = max(0.05, min(1.0, room_size))
    decay = max(0.05, min(0.99, decay))
    tail_seconds = 0.3 + room_size * 1.7
    tail_samples = int(tail_seconds * sample_rate)
    if tail_samples <= 1:
        return waveform

    impulse = np.zeros(tail_samples, dtype=np.float32)
    times = np.arange(tail_samples, dtype=np.float32)
    impulse = np.exp(-times / (decay * sample_rate * 0.6)).astype(np.float32)

    # Add a few early reflections to widen the stereo image (still mono-friendly).
    reflections = (
        (int(sample_rate * room_size * 0.12), 0.6),
        (int(sample_rate * room_size * 0.27), 0.4),
        (int(sample_rate * room_size * 0.43), 0.25),
    )
    for offset, gain in reflections:
        if 0 <= offset < tail_samples:
            impulse[offset] += gain

    convolved = fftconvolve(waveform, impulse)[: len(waveform)]
    return ((1 - wet) * waveform + wet * convolved).astype(np.float32)


def apply_presence_enhancer(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    amount: float = 0.3,
    air_cutoff: float = 2800.0,
    clean_cutoff: float = 120.0,
) -> np.ndarray:
    """Boost upper harmonics while taming rumble to add clarity."""

    amount = max(0.0, min(1.0, amount))
    if amount == 0.0:
        return waveform

    working = waveform.astype(np.float32)

    try:
        low_part = apply_lowpass_filter(
            working,
            sample_rate=sample_rate,
            cutoff=air_cutoff,
            order=4,
        )
    except ValueError:
        low_part = np.copy(working)

    high_band = working - low_part
    enhanced = working + amount * high_band

    if clean_cutoff:
        try:
            enhanced = apply_highpass_filter(
                enhanced,
                sample_rate=sample_rate,
                cutoff=clean_cutoff,
                order=2,
            )
        except ValueError:
            pass

    return normalise_audio(enhanced)


def normalise_audio(waveform: np.ndarray) -> np.ndarray:
    """Return ``waveform`` scaled to avoid clipping."""

    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp
    return waveform.astype(np.float32)


def apply_effect_chain(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    settings: Optional[Dict[str, Union[float, None]]] = None,
) -> np.ndarray:
    """Apply gain, filtering, delay, distortion, and reverb in sequence."""

    processed = waveform.astype(np.float32)
    settings = settings or {}

    gain_db = float(settings.get("gain_db", 0.0) or 0.0)
    if gain_db:
        processed = apply_gain(processed, gain_db)

    hp_cutoff = settings.get("highpass_cutoff")
    if hp_cutoff:
        processed = apply_highpass_filter(
            processed,
            sample_rate=sample_rate,
            cutoff=float(hp_cutoff),
        )

    cutoff = settings.get("lowpass_cutoff")
    if cutoff:
        processed = apply_lowpass_filter(
            processed,
            sample_rate=sample_rate,
            cutoff=float(cutoff),
        )

    delay_seconds = settings.get("delay_seconds")
    if delay_seconds:
        processed = apply_delay(
            processed,
            sample_rate=sample_rate,
            delay_seconds=float(delay_seconds),
            feedback=float(settings.get("delay_feedback", 0.3) or 0.3),
            mix=float(settings.get("delay_mix", 0.25) or 0.25),
        )

    drive = settings.get("drive")
    if drive:
        processed = apply_distortion(processed, drive=float(drive))

    reverb_wet = settings.get("reverb_wet")
    if reverb_wet:
        processed = apply_reverb(
            processed,
            sample_rate=sample_rate,
            wet=float(reverb_wet),
            room_size=float(settings.get("reverb_size", 0.4) or 0.4),
            decay=float(settings.get("reverb_decay", 0.6) or 0.6),
        )

    clarity_amount = settings.get("clarity_amount")
    if clarity_amount:
        processed = apply_presence_enhancer(
            processed,
            sample_rate=sample_rate,
            amount=float(clarity_amount),
        )

    return normalise_audio(processed)
