"""Audio synthesis utilities for FTIR-to-sound conversion and arrangement."""
from __future__ import annotations

import io
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

SPEED_OF_LIGHT = 2.99792458e10  # cm/s
AudioComponent = Tuple[float, float]


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


def _wrap_frequency_to_band(
    frequency: float,
    low: float = 110.0,
    high: float = 880.0,
) -> float:
    """Move ``frequency`` into ``[low, high]`` by octave shifts (powers of two)."""

    if frequency <= 0:
        return frequency
    wrapped = float(frequency)
    while wrapped < low:
        wrapped *= 2.0
    while wrapped > high:
        wrapped /= 2.0
    return wrapped


def _combine_nearby_components(
    components: Sequence[AudioComponent],
    *,
    tolerance_hz: float = 12.0,
) -> List[AudioComponent]:
    """Average components that sit within ``tolerance_hz`` of each other."""

    if not components:
        return []

    sorted_components = sorted(components, key=lambda item: item[0])
    merged: List[AudioComponent] = []
    current_freq, current_amp = sorted_components[0]

    for freq, amp in sorted_components[1:]:
        if abs(freq - current_freq) <= tolerance_hz:
            total_amp = current_amp + amp
            if total_amp > 0:
                current_freq = (current_freq * current_amp + freq * amp) / total_amp
            current_amp = total_amp
        else:
            merged.append((current_freq, current_amp))
            current_freq, current_amp = freq, amp

    merged.append((current_freq, current_amp))
    return merged


def _oscillator(
    t: np.ndarray,
    frequency: float,
    *,
    shape: str,
    detune: float,
    harmonics: Sequence[Tuple[int, float]],
    sub_osc: float,
    noise_mix: float,
) -> np.ndarray:
    """Generate an oscillator waveform for ``frequency`` over time vector ``t``."""

    # Interpret detune in semitones relative to the base frequency.
    detune_ratio = 2 ** (detune / 12.0)
    base_freq = max(0.0, frequency * detune_ratio)
    phase = 2 * np.pi * base_freq * t

    if shape == "square":
        wave = np.sign(np.sin(phase))
    elif shape == "saw":
        wave = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
    elif shape == "triangle":
        wave = 2.0 * np.abs(2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
    elif shape == "noise":
        wave = np.random.uniform(-1.0, 1.0, size=t.shape)
    else:  # default to sine
        wave = np.sin(phase)

    for multiple, amplitude in harmonics:
        if amplitude:
            wave += amplitude * np.sin(2 * np.pi * base_freq * multiple * t)

    if sub_osc:
        wave += sub_osc * np.sin(2 * np.pi * (base_freq / 2.0) * t)

    if noise_mix:
        wave = (1 - noise_mix) * wave + noise_mix * np.random.uniform(
            -1.0, 1.0, size=t.shape
        )

    return wave.astype(np.float32)


def _adsr_envelope(
    length: int,
    *,
    sample_rate: int,
    attack: float,
    decay: float,
    sustain_level: float,
    release: float,
) -> np.ndarray:
    """Construct an ADSR envelope with the provided parameters."""

    envelope = np.ones(length, dtype=np.float32)

    attack_samples = int(max(0.0, attack) * sample_rate)
    decay_samples = int(max(0.0, decay) * sample_rate)
    release_samples = int(max(0.0, release) * sample_rate)
    sustain_level = float(min(max(sustain_level, 0.0), 1.0))

    total_env = attack_samples + decay_samples + release_samples
    if total_env > length:
        scale = length / max(total_env, 1)
        attack_samples = int(round(attack_samples * scale))
        decay_samples = int(round(decay_samples * scale))
        release_samples = int(round(release_samples * scale))

    cursor = 0
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, endpoint=False)
        cursor += attack_samples

    if decay_samples > 0 and cursor < length:
        end = min(length, cursor + decay_samples)
        envelope[cursor:end] = np.linspace(1.0, sustain_level, end - cursor, endpoint=False)
        cursor = end

    sustain_samples = max(0, length - cursor - release_samples)
    if sustain_samples > 0 and cursor < length:
        envelope[cursor:cursor + sustain_samples] = sustain_level
        cursor += sustain_samples

    if release_samples > 0 and cursor < length:
        release_start = envelope[cursor - 1] if cursor > 0 else sustain_level
        envelope[cursor:] = np.linspace(release_start, 0.0, length - cursor, endpoint=True)
    elif cursor < length:
        envelope[cursor:] = sustain_level

    return envelope


def _apply_envelope(
    waveform: np.ndarray,
    *,
    envelope: str,
    sample_rate: int,
    envelope_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Apply one of the supported envelope shapes to ``waveform``."""

    if waveform.size == 0:
        return waveform

    envelope = envelope.lower()
    if envelope == "none":
        return waveform.astype(np.float32)

    if envelope == "hann":
        return (waveform * np.hanning(len(waveform))).astype(np.float32)

    if envelope == "adsr":
        params = envelope_params or {}
        adsr = _adsr_envelope(
            len(waveform),
            sample_rate=sample_rate,
            attack=float(params.get("attack", 0.05)),
            decay=float(params.get("decay", 0.1)),
            sustain_level=float(params.get("sustain_level", 0.7)),
            release=float(params.get("release", 0.2)),
        )
        return (waveform * adsr).astype(np.float32)

    return waveform.astype(np.float32)


def generate_waveform(
    frequencies: Sequence[Union[float, AudioComponent]],
    *,
    duration: float = 2.0,
    sample_rate: int = 44100,
    envelope: str = "hann",
    waveform_shape: str = "sine",
    detune: float = 0.0,
    harmonics: Optional[Sequence[Tuple[int, float]]] = None,
    sub_osc: float = 0.0,
    noise_mix: float = 0.0,
    envelope_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Generate a waveform by summing oscillators for ``frequencies``."""

    sample_count = int(sample_rate * duration)
    if sample_count <= 0:
        return np.zeros(0, dtype=np.float32)

    t = np.linspace(0, duration, sample_count, endpoint=False)
    waveform = np.zeros_like(t, dtype=np.float32)

    harmonics = harmonics or []
    for component in frequencies:
        if isinstance(component, tuple):
            freq, amplitude = component
        else:
            freq, amplitude = component, 1.0
        if freq <= 0 or amplitude == 0:
            continue
        osc = _oscillator(
            t,
            freq,
            shape=waveform_shape,
            detune=detune,
            harmonics=harmonics,
            sub_osc=sub_osc,
            noise_mix=noise_mix,
        )
        waveform += amplitude * osc

    waveform = _apply_envelope(
        waveform,
        envelope=envelope,
        sample_rate=sample_rate,
        envelope_params=envelope_params,
    )

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
        audio_freq = _wrap_frequency_to_band(audio_freq)
        weight = match.match_count / total_occurrences if total_occurrences else 0.0
        components.append((audio_freq, weight))

    smoothed = _combine_nearby_components(components)
    total_weight = sum(weight for _, weight in smoothed)
    if total_weight > 0:
        smoothed = [(freq, weight / total_weight) for freq, weight in smoothed]
    return smoothed
