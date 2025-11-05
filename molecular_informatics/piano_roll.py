"""Arrangement helpers for piano-roll style sequencing."""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import audio_utils
from .effects import apply_effect_chain, normalise_audio

AudioComponent = audio_utils.AudioComponent


@dataclass
class PianoRollEvent:
    """Representation of a single piano-roll note event."""

    start: float
    duration: float
    frequency: float
    amplitude: float
    label: Optional[str] = None


NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

NOTE_OFFSETS = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}
NOTE_NAME_PATTERN = re.compile(r"^([A-G])(#?)(-?\d+)$")


def frequency_to_midi(frequency: float) -> float:
    """Convert a frequency in Hz to a MIDI note number."""

    if frequency <= 0:
        raise ValueError("Frequency must be positive to convert to MIDI note.")
    return 69.0 + 12.0 * np.log2(frequency / 440.0)


def midi_to_note_name(midi_value: int) -> str:
    """Return the note name (with octave) for ``midi_value``."""

    octave, note_index = divmod(midi_value, 12)
    return f"{NOTE_NAMES[note_index]}{octave - 1}"


ALL_NOTE_NAMES = [
    midi_to_note_name(midi)
    for midi in range(36, 97)
]


def note_name_to_midi(note_name: str) -> Optional[int]:
    """Convert a note name like ``C4`` into a MIDI note number."""

    if not isinstance(note_name, str):
        return None
    match = NOTE_NAME_PATTERN.match(note_name.strip().upper())
    if not match:
        return None
    letter, accidental, octave_str = match.groups()
    pitch = letter + ("#" if accidental else "")
    offset = NOTE_OFFSETS.get(pitch)
    if offset is None:
        return None
    try:
        octave = int(octave_str)
    except ValueError:
        return None
    midi = (octave + 1) * 12 + offset
    if midi < 0 or midi > 127:
        return None
    return midi


def note_name_to_frequency(note_name: str) -> Optional[float]:
    """Return the frequency in Hz for a given note name."""

    midi = note_name_to_midi(note_name)
    if midi is None:
        return None
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def components_to_events(
    components: Sequence[AudioComponent],
    *,
    start: float,
    duration: float,
    label: Optional[str] = None,
) -> List[PianoRollEvent]:
    """Convert ``components`` into :class:`PianoRollEvent` objects."""

    events: List[PianoRollEvent] = []
    for frequency, amplitude in components:
        events.append(
            PianoRollEvent(
                start=start,
                duration=duration,
                frequency=frequency,
                amplitude=amplitude,
                label=label,
            )
        )
    return events


def loop_waveform(
    waveform: np.ndarray,
    *,
    loop_count: int,
    sample_rate: int,
    crossfade: float = 0.0,
) -> np.ndarray:
    """Repeat ``waveform`` ``loop_count`` times with optional crossfade."""

    loop_count = max(int(loop_count), 1)
    if loop_count == 1:
        return waveform.astype(np.float32)

    repeated = np.tile(waveform, loop_count).astype(np.float32)
    if crossfade <= 0:
        return repeated

    fade_samples = int(min(crossfade * sample_rate, len(waveform)))
    if fade_samples <= 0:
        return repeated

    segment_length = len(waveform)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = 1.0 - fade_out

    for idx in range(1, loop_count):
        boundary = idx * segment_length
        tail_start = boundary - fade_samples
        left = repeated[tail_start:boundary].copy()
        right = repeated[boundary:boundary + fade_samples].copy()
        repeated[tail_start:boundary] = left * fade_out + right * (1.0 - fade_out)
        repeated[boundary:boundary + fade_samples] = (
            left * fade_in + right * (1.0 - fade_in)
        )

    return normalise_audio(repeated)


def mix_timeline(
    segments: Sequence[Tuple[np.ndarray, float]],
    *,
    sample_rate: int,
) -> np.ndarray:
    """Mix ``segments`` where each entry is (waveform, start_time)."""

    if not segments:
        return np.zeros(0, dtype=np.float32)

    total_samples = 0
    for waveform, start in segments:
        if start < 0:
            raise ValueError("Segment start times must be non-negative.")
        total_samples = max(
            total_samples,
            int(round(start * sample_rate)) + len(waveform),
        )

    mix = np.zeros(total_samples, dtype=np.float32)
    for waveform, start in segments:
        start_index = int(round(start * sample_rate))
        end_index = start_index + len(waveform)
        mix[start_index:end_index] += waveform

    return normalise_audio(mix)


def _decode_wav_bytes(
    audio_bytes: bytes,
    *,
    target_sample_rate: int,
) -> np.ndarray:
    """Decode ``audio_bytes`` into a mono float32 waveform."""

    from scipy.io import wavfile
    from scipy.signal import resample

    buffer = io.BytesIO(audio_bytes)
    source_rate, data = wavfile.read(buffer)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    if source_rate != target_sample_rate:
        target_length = int(round(len(data) * target_sample_rate / source_rate))
        if target_length <= 0:
            return np.zeros(0, dtype=np.float32)
        data = resample(data, target_length).astype(np.float32)

    return data.astype(np.float32)


def build_track_waveform(
    track: Dict[str, Union[str, float, int, bool, bytes, List[AudioComponent]]],
    *,
    sample_rate: int,
) -> Tuple[np.ndarray, float, List[AudioComponent]]:
    """Return the rendered waveform, duration, and active components for ``track``."""

    source = track.get("source", "components")
    duration = float(track.get("duration", 1.0))

    if source == "upload" and track.get("audio_bytes"):
        waveform = _decode_wav_bytes(
            track["audio_bytes"],
            target_sample_rate=sample_rate,
        )
        if waveform.size == 0:
            return waveform, 0.0, []

        base_duration = len(waveform) / sample_rate
        target_duration = float(track.get("duration") or base_duration)
        if abs(target_duration - base_duration) > 1e-3 and target_duration > 0:
            from scipy.signal import resample

            desired_samples = int(round(target_duration * sample_rate))
            if desired_samples > 0:
                waveform = resample(waveform, desired_samples).astype(np.float32)
                base_duration = target_duration
        return waveform.astype(np.float32), base_duration, []

    components = track.get("components") or []
    note_mode = track.get("note_mode", "harmonic")
    components_override: Optional[List[AudioComponent]] = None

    if note_mode == "single":
        note_name = track.get("note_name")
        freq = note_name_to_frequency(note_name) if note_name else None
        if freq:
            components_override = [(float(freq), 1.0)]
    elif note_mode == "chord":
        chord_notes = track.get("chord_notes") or []
        freqs = [note_name_to_frequency(name) for name in chord_notes]
        freqs = [f for f in freqs if f]
        if freqs:
            weight = 1.0 / len(freqs)
            components_override = [(float(freq), weight) for freq in freqs]
    elif note_mode == "transpose":
        semitones = float(track.get("transpose_semitones", 0.0) or 0.0)
        if components:
            shift = 2 ** (semitones / 12.0)
            components_override = [
                (float(freq) * shift, float(weight))
                for freq, weight in components
            ]

    if components_override is not None:
        active_components = components_override
    else:
        active_components = [(float(freq), float(weight)) for freq, weight in components]

    if not active_components:
        return np.zeros(0, dtype=np.float32), 0.0, []

    harmonics: List[Tuple[int, float]] = []
    for harmonic in (2, 3):
        level = float(track.get(f"harmonic_{harmonic}", 0.0) or 0.0)
        if level:
            harmonics.append((harmonic, level))

    adsr_params = None
    if track.get("envelope", "hann") == "adsr":
        adsr_params = {
            "attack": float(track.get("adsr_attack", 0.05) or 0.0),
            "decay": float(track.get("adsr_decay", 0.1) or 0.0),
            "sustain_level": float(track.get("adsr_sustain_level", 0.7) or 0.0),
            "release": float(track.get("adsr_release", 0.2) or 0.0),
        }

    drone_mode = bool(track.get("drone_mode", False))

    waveform = audio_utils.generate_waveform(
        active_components,
        duration=duration,
        sample_rate=sample_rate,
        envelope="none" if drone_mode else track.get("envelope", "hann"),
        waveform_shape=track.get("waveform_shape", "sine"),
        detune=float(track.get("detune", 0.0) or 0.0),
        sub_osc=float(track.get("sub_osc", 0.0) or 0.0),
        noise_mix=float(track.get("noise_mix", 0.0) or 0.0),
        harmonics=harmonics,
        envelope_params=adsr_params,
    )
    return waveform, duration, active_components


def render_track_audio(
    track: Dict[str, Union[str, float, int, bool, bytes, List[AudioComponent]]],
    *,
    sample_rate: int,
    render_length: Optional[float] = None,
) -> Tuple[np.ndarray, float, int, List[AudioComponent]]:
    """Return the processed audio, clip duration, loop count, and active components."""

    base_waveform, duration, active_components = build_track_waveform(
        track,
        sample_rate=sample_rate,
    )
    if base_waveform.size == 0 or duration <= 0:
        return np.zeros(0, dtype=np.float32), 0.0, 0, []

    loop_mode = track.get("loop_mode", "fixed")
    start_time = float(track.get("start_time", 0.0) or 0.0)
    if loop_mode == "continuous" and render_length and render_length > 0:
        remaining = max(render_length - start_time, duration)
        loops_needed = max(1, int(math.ceil(remaining / duration)))
    else:
        loops_needed = max(1, int(track.get("loop_count", 1)))

    looped = loop_waveform(
        base_waveform,
        loop_count=loops_needed,
        sample_rate=sample_rate,
        crossfade=float(track.get("crossfade", 0.0) or 0.0),
    )

    processed = apply_effect_chain(
        looped,
        sample_rate=sample_rate,
        settings={
            "gain_db": track.get("gain_db"),
            "highpass_cutoff": track.get("highpass_cutoff"),
            "lowpass_cutoff": track.get("lowpass_cutoff"),
            "delay_seconds": track.get("delay_seconds"),
            "delay_feedback": track.get("delay_feedback"),
            "delay_mix": track.get("delay_mix"),
            "reverb_wet": track.get("reverb_wet"),
            "reverb_size": track.get("reverb_size"),
            "reverb_decay": track.get("reverb_decay"),
        },
    )

    volume = float(track.get("track_volume", 1.0) or 1.0)
    if volume != 1.0:
        processed = (processed * volume).astype(np.float32)

    return processed, duration, loops_needed, active_components


def build_arrangement(
    tracks: Sequence[Dict[str, Union[str, float, int, bool, bytes, List[AudioComponent]]]],
    *,
    sample_rate: int,
    render_length: Optional[float] = None,
) -> Dict[str, Union[np.ndarray, List[PianoRollEvent], List[Dict[str, str]]]]:
    """Render ``tracks`` into a mixed waveform and metadata payload."""

    if not tracks:
        return {"waveform": np.zeros(0, dtype=np.float32), "events": [], "summary": []}

    solo_active = any(track.get("is_solo", False) for track in tracks)
    segments: List[Tuple[np.ndarray, float]] = []
    events: List[PianoRollEvent] = []
    summary: List[Dict[str, str]] = []

    for track in tracks:
        if not track.get("is_enabled", True) or track.get("is_muted", False):
            continue
        if solo_active and not track.get("is_solo", False):
            continue

        processed, duration, loops_used, active_components = render_track_audio(
            track,
            sample_rate=sample_rate,
            render_length=render_length,
        )
        if processed.size == 0 or duration <= 0:
            continue

        start_time = float(track.get("start_time", 0.0) or 0.0)
        segments.append((processed, start_time))

        track_name = track.get("name") or "Track"
        components = active_components or []
        if components:
            for loop_index in range(loops_used):
                loop_start = start_time + loop_index * duration
                events.extend(
                    components_to_events(
                        components,
                        start=loop_start,
                        duration=duration,
                        label=track_name,
                    )
                )

        rendered_length = len(processed) / sample_rate
        summary.append(
            {
                "Track": track_name,
                "Start (s)": f"{start_time:.2f}",
                "Loops": "continuous" if track.get("loop_mode") == "continuous" else str(loops_used),
                "Length (s)": f"{rendered_length:.2f}",
                "Gain (dB)": f"{float(track.get('gain_db', 0.0) or 0.0):+.1f}",
            }
        )

    mixed = mix_timeline(segments, sample_rate=sample_rate)
    if render_length and render_length > 0:
        desired_samples = int(round(render_length * sample_rate))
        if desired_samples > mixed.size:
            padding = np.zeros(desired_samples - mixed.size, dtype=np.float32)
            mixed = np.concatenate([mixed, padding])
        elif desired_samples < mixed.size:
            mixed = mixed[:desired_samples]
    return {"waveform": mixed, "events": events, "summary": summary}
