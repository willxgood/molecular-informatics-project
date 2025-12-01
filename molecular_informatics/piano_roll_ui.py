"""Streamlit UI helpers for the piano roll arranger."""
from __future__ import annotations

import copy
import io
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from components.looping_player import render_looping_audio_player
from components.molecule_visualizer import render_molecule_visualizer
from rdkit import Chem

from . import audio_utils, chem_utils, piano_roll


def _tracks_state() -> List[Dict[str, Any]]:
    if "piano_roll_tracks" not in st.session_state:
        st.session_state["piano_roll_tracks"] = []
    return st.session_state["piano_roll_tracks"]


def _transport_state() -> Dict[str, Any]:
    if "piano_roll_transport" not in st.session_state:
        st.session_state["piano_roll_transport"] = {"playing": False}
    return st.session_state["piano_roll_transport"]


def _default_track_payload(
    *,
    name: str,
    duration: float,
    components: Optional[List[piano_roll.AudioComponent]] = None,
) -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "name": name,
        "start_time": 0.0,
        "duration": float(duration),
        "loop_count": 1,
        "crossfade": 0.0,
        "loop_mode": "continuous",
        "is_enabled": True,
        "is_muted": False,
        "is_solo": False,
        "bar_start": 1,
        "bar_end": 1,
        "tone_preset": "Custom",
        "quantize_scale": False,
        "quantize_root": "C",
        "quantize_mode": "major",
        "match_payload": None,
        "components": list(components or []),
        "source": "components",
        "audio_bytes": None,
        "waveform_shape": "sine",
        "detune": 0.0,
        "sub_osc": 0.0,
        "noise_mix": 0.0,
        "harmonic_2": 0.0,
        "harmonic_3": 0.0,
        "envelope": "hann",
        "adsr_attack": 0.05,
        "adsr_decay": 0.1,
        "adsr_sustain_level": 0.7,
        "adsr_release": 0.2,
        "drone_mode": False,
        "gain_db": 0.0,
        "highpass_cutoff": None,
        "lowpass_cutoff": None,
        "delay_seconds": None,
        "delay_feedback": 0.3,
        "delay_mix": 0.25,
        "reverb_wet": 0.0,
        "reverb_size": 0.4,
        "reverb_decay": 0.6,
        "track_volume": 1.0,
        "note_mode": "harmonic",
        "note_name": "C4",
        "transpose_semitones": 0.0,
        "chord_notes": [],
        "upload_sample_rate": None,
        "upload_name": None,
    }


def _add_molecule_track(
    *,
    name: str,
    components: List[piano_roll.AudioComponent],
    duration: float,
    smiles: str,
    matches: Optional[List[chem_utils.FunctionalGroupMatch]] = None,
):
    track = _default_track_payload(name=name, duration=duration, components=components)
    track["smiles"] = smiles
    if matches:
        track["match_payload"] = [
            {"wn": float(m.group.center_wavenumber), "count": int(m.match_count)}
            for m in matches
            if m.present
        ]
    _tracks_state().append(track)


def _read_uploaded_clip(file) -> Optional[Dict[str, Any]]:
    try:
        from scipy.io import wavfile
    except ImportError:  # pragma: no cover
        st.error("scipy must be installed to decode uploaded WAV files.")
        return None

    raw = file.getvalue()
    buffer = io.BytesIO(raw)
    sample_rate, data = wavfile.read(buffer)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    duration = len(data) / sample_rate if len(data) else 0.0
    return {
        "bytes": raw,
        "sample_rate": sample_rate,
        "duration": duration,
        "name": getattr(file, "name", "clip.wav"),
    }


def _synth_drum_loop(
    loop_bars: int,
    tempo_bpm: float,
    swing: float,
    sample_rate: int,
    pattern: str = "basic",
    *,
    kick_level: float = 1.0,
    snare_level: float = 1.0,
    hat_level: float = 1.0,
    humanize_ms: float = 0.0,
    master_gain: float = 1.0,
) -> Dict[str, Any]:
    """Generate a lightweight synthetic drum loop (kick/snare/hat) with swing applied."""

    loop_bars = max(1, int(loop_bars))
    seconds_per_beat = 60.0 / max(tempo_bpm, 1.0)
    beats_per_bar = 4.0
    loop_length = loop_bars * beats_per_bar * seconds_per_beat
    total_samples = int(loop_length * sample_rate)
    if total_samples <= 0:
        return {"bytes": b"", "sample_rate": sample_rate, "duration": 0.0, "name": "drums.wav"}

    waveform = np.zeros(total_samples, dtype=np.float32)

    def _add_pulse(start_time: float, shape: str):
        jitter = 0.0
        if humanize_ms > 0:
            jitter = np.random.uniform(-humanize_ms, humanize_ms) / 1000.0
        start_time += jitter
        if start_time < 0:
            start_time = 0.0
        start_idx = int(start_time * sample_rate)
        if start_idx >= total_samples:
            return
        if shape == "kick":
            length = int(0.3 * sample_rate)
            t = np.linspace(0, 0.3, length, endpoint=False)
            env = np.exp(-t * 9.0)
            sig = np.sin(2 * np.pi * 60.0 * t) * env * float(kick_level)
        elif shape == "snare":
            length = int(0.22 * sample_rate)
            t = np.linspace(0, 0.22, length, endpoint=False)
            env = np.exp(-t * 12.0)
            noise = np.random.uniform(-1.0, 1.0, size=length) * env
            sig = noise * 0.65 * float(snare_level)
        else:  # hat
            length = int(0.1 * sample_rate)
            t = np.linspace(0, 0.1, length, endpoint=False)
            env = np.exp(-t * 18.0)
            noise = np.random.uniform(-1.0, 1.0, size=length) * env
            sig = noise * 0.38 * float(hat_level)

        # Trim if start is near the end to avoid zero-length slices
        sig_len = len(sig)
        if start_idx + sig_len <= 0:
            return
        if start_idx < 0:
            sig = sig[-start_idx:]
            start_idx = 0
        end_idx = min(total_samples, start_idx + len(sig))
        if end_idx <= start_idx:
            return
        waveform[start_idx:end_idx] += sig[: end_idx - start_idx]

    def _swing_time(base_time: float, step_index: int, division: float) -> float:
        # Swing every second subdivision
        if swing <= 0:
            return base_time
        if step_index % 2 == 1:
            return base_time + swing * division * seconds_per_beat
        return base_time

    patterns = {
        "basic": {
            "kicks": [0.0, 2.0],
            "snares": [1.0, 3.0],
            "hats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        },
        "four_on_floor": {
            "kicks": [0.0, 1.0, 2.0, 3.0],
            "snares": [1.0, 3.0],
            "hats": [i * 0.25 for i in range(16)],
        },
        "halftime": {
            "kicks": [0.0],
            "snares": [2.0],
            "hats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        },
        "syncopated": {
            "kicks": [0.0, 1.5, 2.5],
            "snares": [1.0, 3.5],
            "hats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        },
        "two_step": {
            "kicks": [0.0, 1.75, 3.0],
            "snares": [2.0],
            "hats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        },
        "trap": {
            "kicks": [0.0, 1.5, 2.0, 3.25],
            "snares": [1.5, 3.0],
            "hats": [i * 0.25 for i in range(16)],
        },
    }

    def _swing_time(base_beats: float) -> float:
        base_time = base_beats * seconds_per_beat
        if swing <= 0:
            return base_time
        # Swing eighths (0.5 beat offsets): push every second 8th later
        if abs((base_beats * 2) - round(base_beats * 2)) < 1e-6:
            idx = int(round(base_beats * 2))
            if idx % 2 == 1:
                base_time += swing * 0.5 * seconds_per_beat
        return base_time

    pat = patterns.get(pattern, patterns["basic"])

    for bar in range(loop_bars):
        offset = bar * beats_per_bar
        for k in pat["kicks"]:
            _add_pulse(_swing_time(offset + k), "kick")
        for s in pat["snares"]:
            _add_pulse(_swing_time(offset + s), "snare")
        for h in pat["hats"]:
            _add_pulse(_swing_time(offset + h), "hat")

    waveform = np.clip(waveform, -1.0, 1.0) * float(master_gain) * 0.9
    wav_bytes = audio_utils.waveform_to_wav_bytes(waveform, sample_rate=sample_rate)
    return {
        "bytes": wav_bytes,
        "sample_rate": sample_rate,
        "duration": loop_length,
        "name": f"Drums_{pattern}_{loop_bars}bar_{int(tempo_bpm)}bpm.wav",
    }


def _add_uploaded_track(payload: Dict[str, Any]):
    track = _default_track_payload(
        name=payload.get("name", "Clip"),
        duration=max(payload.get("duration", 0.0), 0.1),
        components=[],
    )
    track.update(
        {
            "source": "upload",
            "audio_bytes": payload.get("bytes"),
            "upload_sample_rate": payload.get("sample_rate"),
            "upload_name": payload.get("name"),
        }
    )
    _tracks_state().append(track)


def _add_drum_track(payload: Dict[str, Any], settings: Dict[str, Any]):
    track = _default_track_payload(
        name=payload.get("name", "Drums"),
        duration=max(payload.get("duration", 0.0), 0.1),
        components=[],
    )
    track.update(
        {
            "source": "drum",
            "audio_bytes": payload.get("bytes"),
            "upload_sample_rate": payload.get("sample_rate"),
            "upload_name": payload.get("name", "Drums"),
            "drum_settings": settings,
        }
    )
    _tracks_state().append(track)


def _build_piano_roll_figure(
    events: List[piano_roll.PianoRollEvent],
    *,
    render_length: Optional[float],
    bar_length: Optional[float],
    grid_step: Optional[float] = None,
):
    if not events:
        return None

    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative
    except ImportError:  # pragma: no cover - optional dependency
        return None

    events_sorted = sorted(
        events,
        key=lambda ev: (piano_roll.frequency_to_midi(ev.frequency), ev.start),
    )

    palette = qualitative.Plotly
    color_map: Dict[str, str] = {}
    legend_present: Dict[str, bool] = {}
    note_order: Dict[str, int] = {}

    fig = go.Figure()
    for event in events_sorted:
        midi_val = piano_roll.frequency_to_midi(event.frequency)
        rounded = int(round(midi_val))
        note_name = piano_roll.midi_to_note_name(rounded)
        note_order[note_name] = rounded
        label = event.label or "Track"
        if label not in color_map:
            color_map[label] = palette[len(color_map) % len(palette)]

        opacity = float(min(0.95, max(0.25, event.amplitude * 1.4)))
        showlegend = not legend_present.get(label, False)
        legend_present[label] = True

        fig.add_trace(
            go.Bar(
                x=[event.duration],
                y=[note_name],
                base=event.start,
                orientation="h",
                name=label,
                marker=dict(color=color_map[label], opacity=opacity),
                customdata=[[label, event.frequency, event.amplitude]],
                hovertemplate=(
                    "Track: %{customdata[0]}<br>Note: %{y}<br>"
                    "Start: %{base:.2f}s<br>Duration: %{x:.2f}s<br>"
                    "Frequency: %{customdata[1]:.1f} Hz<br>Amplitude: %{customdata[2]:.2f}"
                ),
                legendgroup=label,
                showlegend=showlegend,
            )
        )

    ordered_notes = [note for note, _ in sorted(note_order.items(), key=lambda item: item[1])]

    shapes = []
    if render_length and bar_length and bar_length > 0:
        bar_line = 0.0
        while bar_line <= render_length:
            shapes.append(
                dict(
                    type="line",
                    x0=bar_line,
                    x1=bar_line,
                    y0=-0.5,
                    y1=len(ordered_notes) + 0.5,
                    line=dict(color="rgba(0,0,0,0.2)", width=1.5),
                    layer="below",
                )
            )
            bar_line += bar_length

        if grid_step and grid_step > 0:
            grid_line = 0.0
            while grid_line <= render_length:
                shapes.append(
                    dict(
                        type="line",
                        x0=grid_line,
                        x1=grid_line,
                        y0=-0.5,
                        y1=len(ordered_notes) + 0.5,
                        line=dict(color="rgba(0,0,0,0.08)", width=1),
                        layer="below",
                    )
                )
                grid_line += grid_step

    fig.update_layout(
        barmode="overlay",
        bargap=0.15,
        bargroupgap=0.05,
        xaxis_title="Time (s)",
        yaxis_title="Note",
        yaxis=dict(categoryorder="array", categoryarray=ordered_notes),
        template="plotly_white",
        height=max(320, 60 + 30 * len(ordered_notes)),
        legend_title="Tracks",
        margin=dict(l=60, r=30, t=40, b=60),
        shapes=shapes,
    )
    return fig


def _render_transport_controls(tracks: List[Dict[str, Any]]):
    transport = _transport_state()
    cols = st.columns([1, 1, 1, 1])
    if cols[0].button("Play arrangement", key="pr_transport_play"):
        transport["playing"] = True
    if cols[1].button("Stop", key="pr_transport_stop"):
        transport["playing"] = False
    if cols[2].button("Clear solos", key="pr_transport_clear_solos"):
        for track in tracks:
            track["is_solo"] = False
    if cols[3].button("Reset mutes", key="pr_transport_reset_mutes"):
        for track in tracks:
            track["is_muted"] = False


def _render_track_editor(
    track: Dict[str, Any],
    *,
    sample_rate: int,
    bar_count: int,
    bar_length: float,
    grid_step_bars: float,
    quantized_start: Optional[float] = None,
    tempo_bpm: Optional[float] = None,
    swing: Optional[float] = None,
) -> Dict[str, bool]:
    actions = {"remove": False, "preview": False, "duplicate": False, "duplicate_shifted": False, "regen_drum": False}
    header = f"{track.get('name', 'Track')}"
    tag = track.get("tag", "")
    if tag:
        header += f" · {tag}"
    if track.get("source") == "upload" and track.get("upload_name"):
        header += f" · {track['upload_name']}"

    with st.expander(header, expanded=False):
        top_cols = st.columns([1, 1, 1, 1, 1, 1, 1])
        track["is_enabled"] = top_cols[0].checkbox(
            "Active",
            value=track.get("is_enabled", True),
            key=f"track_active_{track['id']}",
        )
        track["is_muted"] = top_cols[1].checkbox(
            "Mute",
            value=track.get("is_muted", False),
            key=f"track_mute_{track['id']}",
        )
        track["is_solo"] = top_cols[2].checkbox(
            "Solo",
            value=track.get("is_solo", False),
            key=f"track_solo_{track['id']}",
        )
        if top_cols[3].button("Preview", key=f"track_preview_{track['id']}"):
            actions["preview"] = True
        if top_cols[4].button("Duplicate", key=f"track_duplicate_{track['id']}"):
            actions["duplicate"] = True
        if top_cols[5].button("Dup +1 bar", key=f"track_duplicate_shift_{track['id']}"):
            actions["duplicate_shifted"] = True
        if top_cols[6].button("Remove", key=f"track_remove_{track['id']}"):
            actions["remove"] = True

        track["name"] = st.text_input(
            "Track name",
            value=track.get("name", "Track"),
            key=f"track_name_{track['id']}",
        )
        track["tag"] = st.text_input(
            "Tag/notes (optional)",
            value=track.get("tag", ""),
            key=f"track_tag_{track['id']}",
            help="Add a quick label like 'pad' or 'lead'.",
        )
        if quantized_start is not None:
            st.caption(f"Quantized start @ swing/grid: {quantized_start:.2f}s")

        arrange_tab, synth_tab, effects_tab = st.tabs(["Arrange", "Synth", "Effects"])

        with arrange_tab:
            if bar_count > 1 and bar_length > 0:
                start_bar_pos = float(
                    st.slider(
                        "Start position (bars)",
                        min_value=0.0,
                        max_value=float(bar_count - 1),
                        value=float(track.get("start_time", 0.0)) / bar_length,
                        step=float(max(grid_step_bars, 0.01)),
                        key=f"track_start_bar_{track['id']}",
                        help="Set where this clip enters the loop; values are in bars.",
                    )
                )
            else:
                start_bar_pos = 0.0
            track["start_time"] = max(0.0, start_bar_pos * bar_length if bar_length else 0.0)
            track["duration"] = float(
                st.number_input(
                    "Clip duration (s)",
                    min_value=0.1,
                    step=0.1,
                    value=float(track.get("duration", 1.0)),
                    key=f"track_duration_{track['id']}",
                )
            )
            loop_mode_options = {
                "continuous": "Loop to arrangement length",
                "fixed": "Repeat a set number of times",
            }
            current_mode = track.get("loop_mode", "fixed")
            selected_mode = st.radio(
                "Looping",
                options=list(loop_mode_options.keys()),
                format_func=lambda key: loop_mode_options[key],
                index=list(loop_mode_options.keys()).index(current_mode)
                if current_mode in loop_mode_options
                else 1,
                key=f"track_loop_mode_{track['id']}",
                horizontal=True,
            )
            track["loop_mode"] = selected_mode
            if selected_mode == "fixed":
                track["loop_count"] = int(
                    st.number_input(
                        "Loop count",
                        min_value=1,
                        max_value=64,
                        step=1,
                        value=int(track.get("loop_count", 1)),
                        key=f"track_loops_{track['id']}",
                    )
                )
            max_crossfade = min(track["duration"] / 2.0, 2.0)
            track["crossfade"] = float(
                st.slider(
                    "Loop crossfade (s)",
                    min_value=0.0,
                    max_value=float(max_crossfade),
                    value=float(track.get("crossfade", 0.0)),
                    step=0.01,
                    key=f"track_crossfade_{track['id']}",
                )
            )
            if bar_count > 1:
                current_start = max(1, int(track.get("bar_start", 1) or 1))
                current_end = int(track.get("bar_end", bar_count) or bar_count)
                if current_end < current_start:
                    current_end = current_start
                # When new bars are added, expand the default range instead of leaving gaps.
                if current_start == 1 and current_end == 1 and bar_count > 1:
                    current_end = bar_count
                bar_range = st.slider(
                    "Active bars",
                    min_value=1,
                    max_value=bar_count,
                    value=(current_start, min(current_end, bar_count)),
                    key=f"track_bar_range_{track['id']}",
                    help="Only play this clip (and its effects) within the selected bars.",
                )
                track["bar_start"], track["bar_end"] = bar_range
            else:
                track["bar_start"], track["bar_end"] = 1, 1
            track["drone_mode"] = st.checkbox(
                "Drone sustain (disable fade)",
                value=track.get("drone_mode", False),
                key=f"track_drone_{track['id']}",
                help="Keeps the waveform at full level without the usual fade shape—ideal for drones.",
            )
            if track.get("source") == "upload" and track.get("upload_sample_rate"):
                st.caption(
                    f"Uploaded clip @ {int(track['upload_sample_rate'])} Hz"
                    f" · {track.get('duration', 0.0):.2f} s"
                )
            if track.get("source") == "drum" and track.get("drum_settings"):
                ds = track.get("drum_settings", {})
                st.caption(
                    f"Drum pattern: {ds.get('pattern', 'basic')} · {ds.get('loop_bars', 1)} bars · gain {ds.get('master_gain',1.0):.1f}"
                )
                col_d1, col_d2 = st.columns(2)
                pattern = col_d1.selectbox(
                    "Pattern",
                    options=["basic", "four_on_floor", "halftime", "syncopated", "two_step", "trap"],
                    index=["basic", "four_on_floor", "halftime", "syncopated", "two_step", "trap"].index(ds.get("pattern", "basic")),
                    key=f"drum_pat_{track['id']}",
                )
                loop_bars = col_d2.selectbox(
                    "Bars",
                    options=[1, 2, 4],
                    index=[1, 2, 4].index(int(ds.get("loop_bars", 2))),
                    key=f"drum_bars_{track['id']}",
                )
                mix_cols = st.columns(4)
                kick_level = mix_cols[0].slider("Kick", 0.2, 1.5, float(ds.get("kick_level", 1.0)), 0.05, key=f"drum_kick_{track['id']}")
                snare_level = mix_cols[1].slider("Snare", 0.2, 1.5, float(ds.get("snare_level", 1.0)), 0.05, key=f"drum_snare_{track['id']}")
                hat_level = mix_cols[2].slider("Hat", 0.2, 1.5, float(ds.get("hat_level", 1.0)), 0.05, key=f"drum_hat_{track['id']}")
                master_gain = mix_cols[3].slider("Gain", 0.5, 2.0, float(ds.get("master_gain", 1.3)), 0.05, key=f"drum_gain_{track['id']}")
                humanize_ms = st.slider(
                    "Humanize timing (ms)",
                    min_value=0.0,
                    max_value=25.0,
                    value=float(ds.get("humanize_ms", 5.0)),
                    step=1.0,
                    key=f"drum_humanize_{track['id']}",
                )
                changed = any(
                    [
                        pattern != ds.get("pattern"),
                        loop_bars != ds.get("loop_bars"),
                        abs(kick_level - float(ds.get("kick_level", 1.0))) > 1e-6,
                        abs(snare_level - float(ds.get("snare_level", 1.0))) > 1e-6,
                        abs(hat_level - float(ds.get("hat_level", 1.0))) > 1e-6,
                        abs(master_gain - float(ds.get("master_gain", 1.0))) > 1e-6,
                        abs(humanize_ms - float(ds.get("humanize_ms", 5.0))) > 1e-6,
                    ]
                )
                track["drum_settings"] = {
                    "loop_bars": loop_bars,
                    "pattern": pattern,
                    "kick_level": kick_level,
                    "snare_level": snare_level,
                    "hat_level": hat_level,
                    "humanize_ms": humanize_ms,
                    "master_gain": master_gain,
                    "tempo_bpm": tempo_bpm if tempo_bpm is not None else ds.get("tempo_bpm", 90),
                    "swing": swing if swing is not None else ds.get("swing", 0.0),
                }
                if changed or st.button("Regenerate drum (apply settings)", key=f"regen_drum_{track['id']}"):
                    actions["regen_drum"] = True

        with synth_tab:
            if track.get("source") == "upload":
                st.info("Uploaded clips bypass the synth oscillator controls.")
            else:
                preset = st.selectbox(
                    "Tone preset",
                    options=["Custom", "Soft pad", "Glass bell", "Crunchy lo-fi", "Drone bed"],
                    index=["Custom", "Soft pad", "Glass bell", "Crunchy lo-fi", "Drone bed"].index(track.get("tone_preset", "Custom")),
                    key=f"track_tone_preset_{track['id']}",
                    help="Quickly shape the oscillator and filters; choose Custom to dial in manually.",
                )
                track["tone_preset"] = preset

                macro_choice = st.selectbox(
                    "Macro quick-set",
                    options=["Manual", "Brighten", "Soften", "Spacey", "Motion"],
                    index=["Manual", "Brighten", "Soften", "Spacey", "Motion"].index(track.get("macro_choice", "Manual")),
                    key=f"macro_choice_{track['id']}",
                    help="Jump to a macro mix; sliders below reflect the selection.",
                )
                track["macro_choice"] = macro_choice
                if macro_choice != "Manual":
                    presets_map = {
                        "Brighten": (0.9, 0.6, 0.25, 0.55),
                        "Soften": (0.35, 0.45, 0.2, 0.3),
                        "Spacey": (0.5, 0.45, 0.8, 0.35),
                        "Motion": (0.55, 0.6, 0.35, 0.8),
                    }
                    macro_vals = presets_map.get(macro_choice)
                    if macro_vals:
                        track["macro_tone"], track["macro_texture"], track["macro_space"], track["macro_move"] = macro_vals

                macro_cols = st.columns(4)
                macro_labels = ["Tone", "Texture", "Space", "Movement"]
                macro_fields = ["macro_tone", "macro_texture", "macro_space", "macro_move"]
                macro_values = []
                for col, label, field in zip(macro_cols, macro_labels, macro_fields):
                    macro_values.append(
                        col.slider(
                            label,
                            min_value=0.0,
                            max_value=1.0,
                            step=0.05,
                            value=float(track.get(field, 0.5)),
                            key=f"{field}_{track['id']}_slider",
                            help="Macro controls tweak tone/filter/reverb/delay together.",
                        )
                    )
                (macro_tone, macro_texture, macro_space, macro_move) = macro_values
                track.update(
                    {
                        "macro_tone": macro_tone,
                        "macro_texture": macro_texture,
                        "macro_space": macro_space,
                        "macro_move": macro_move,
                    }
                )

                # Map macros/preset into synth parameters with gentle ranges.
                def _apply_macro_defaults():
                    base_env = {
                        "adsr_attack": 0.05,
                        "adsr_decay": 0.1,
                        "adsr_sustain_level": 0.7,
                        "adsr_release": 0.2,
                        "harmonic_2": 0.0,
                        "harmonic_3": 0.0,
                        "noise_mix": 0.0,
                        "sub_osc": 0.0,
                        "lowpass_cutoff": None,
                        "reverb_wet": 0.0,
                        "delay_mix": 0.25,
                        "delay_seconds": None,
                        "delay_feedback": 0.3,
                    }
                    track.update(base_env)

                if preset != "Custom":
                    _apply_macro_defaults()
                    if preset == "Soft pad":
                        track.update(
                            {
                                "waveform_shape": "sine",
                                "harmonic_2": 0.15,
                                "harmonic_3": 0.05,
                                "adsr_attack": 0.2,
                                "adsr_release": 0.6,
                                "reverb_wet": 0.25,
                                "reverb_size": 0.6,
                                "reverb_decay": 0.7,
                                "delay_mix": 0.1,
                                "lowpass_cutoff": 2200.0,
                            }
                        )
                    elif preset == "Glass bell":
                        track.update(
                            {
                                "waveform_shape": "sine",
                                "harmonic_2": 0.35,
                                "harmonic_3": 0.15,
                                "adsr_attack": 0.01,
                                "adsr_decay": 0.25,
                                "adsr_sustain_level": 0.4,
                                "adsr_release": 0.35,
                                "reverb_wet": 0.3,
                                "reverb_size": 0.4,
                                "reverb_decay": 0.5,
                                "delay_mix": 0.2,
                                "delay_seconds": 0.22,
                            }
                        )
                    elif preset == "Crunchy lo-fi":
                        track.update(
                            {
                                "waveform_shape": "square",
                                "harmonic_2": 0.0,
                                "harmonic_3": 0.2,
                                "noise_mix": 0.25,
                                "adsr_attack": 0.02,
                                "adsr_decay": 0.08,
                                "adsr_sustain_level": 0.6,
                                "adsr_release": 0.15,
                                "delay_mix": 0.15,
                                "delay_seconds": 0.28,
                                "lowpass_cutoff": 3200.0,
                            }
                        )
                    elif preset == "Drone bed":
                        track.update(
                            {
                                "waveform_shape": "sine",
                                "sub_osc": 0.35,
                                "harmonic_2": 0.05,
                                "harmonic_3": 0.05,
                                "adsr_attack": 0.4,
                                "adsr_decay": 0.4,
                                "adsr_sustain_level": 0.9,
                                "adsr_release": 1.2,
                                "drone_mode": True,
                                "reverb_wet": 0.35,
                                "reverb_size": 0.7,
                                "reverb_decay": 0.85,
                                "lowpass_cutoff": 1800.0,
                            }
                        )

                # Macro application: nudge key parameters while respecting presets.
                track["lowpass_cutoff"] = (
                    800.0 + macro_tone * 3200.0 if track.get("lowpass_cutoff") is None else track["lowpass_cutoff"]
                )
                track["harmonic_2"] = min(0.6, max(0.0, track.get("harmonic_2", 0.0) + (macro_texture - 0.5) * 0.4))
                track["noise_mix"] = min(0.6, max(0.0, track.get("noise_mix", 0.0) + (macro_texture - 0.5) * 0.3))
                track["reverb_wet"] = min(0.6, max(0.0, track.get("reverb_wet", 0.0) + macro_space * 0.3))
                if macro_move > 0.6:
                    track["delay_mix"] = min(0.5, track.get("delay_mix", 0.25) + (macro_move - 0.6) * 0.6)
                    track["delay_seconds"] = track.get("delay_seconds") or 0.25
                track["detune"] = (macro_move - 0.5) * 20.0

                track["waveform_shape"] = st.selectbox(
                    "Oscillator",
                    options=["sine", "square", "saw", "triangle", "noise"],
                    index=["sine", "square", "saw", "triangle", "noise"].index(
                        track.get("waveform_shape", "sine")
                    ),
                    key=f"track_waveform_{track['id']}",
                )
                track["detune"] = float(
                    st.slider(
                        "Detune (semitones)",
                        min_value=-12.0,
                        max_value=12.0,
                        value=float(track.get("detune", 0.0)),
                        step=0.1,
                        key=f"track_detune_{track['id']}",
                    )
                )
                track["sub_osc"] = float(
                    st.slider(
                        "Sub oscillator blend",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(track.get("sub_osc", 0.0)),
                        step=0.05,
                        key=f"track_subosc_{track['id']}",
                    )
                )
                track["noise_mix"] = float(
                    st.slider(
                        "Noise mix",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(track.get("noise_mix", 0.0)),
                        step=0.05,
                        key=f"track_noisemix_{track['id']}",
                    )
                )
                track["harmonic_2"] = float(
                    st.slider(
                        "2nd harmonic",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(track.get("harmonic_2", 0.0)),
                        step=0.05,
                        key=f"track_h2_{track['id']}",
                    )
                )
                track["harmonic_3"] = float(
                    st.slider(
                        "3rd harmonic",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(track.get("harmonic_3", 0.0)),
                        step=0.05,
                        key=f"track_h3_{track['id']}",
                    )
                )

                envelope = st.selectbox(
                    "Envelope",
                    options=["hann", "adsr", "none"],
                    index=["hann", "adsr", "none"].index(track.get("envelope", "hann")),
                    key=f"track_envelope_{track['id']}",
                )
                track["envelope"] = envelope
                if envelope == "adsr":
                    track["adsr_attack"] = float(
                        st.slider(
                            "Attack (s)",
                            min_value=0.0,
                            max_value=2.0,
                            value=float(track.get("adsr_attack", 0.05)),
                            step=0.01,
                            key=f"track_adsr_attack_{track['id']}",
                        )
                    )
                    track["adsr_decay"] = float(
                        st.slider(
                            "Decay (s)",
                            min_value=0.0,
                            max_value=2.0,
                            value=float(track.get("adsr_decay", 0.1)),
                            step=0.01,
                            key=f"track_adsr_decay_{track['id']}",
                        )
                    )
                    track["adsr_sustain_level"] = float(
                        st.slider(
                            "Sustain level",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(track.get("adsr_sustain_level", 0.7)),
                            step=0.05,
                            key=f"track_adsr_sustain_{track['id']}",
                        )
                    )
                    track["adsr_release"] = float(
                        st.slider(
                            "Release (s)",
                            min_value=0.0,
                            max_value=3.0,
                            value=float(track.get("adsr_release", 0.2)),
                            step=0.01,
                            key=f"track_adsr_release_{track['id']}",
                        )
                )

                pitch_mode_map = {
                    "harmonic": "Use molecular harmonics",
                    "single": "Single note",
                    "chord": "Chord builder",
                    "transpose": "Transpose (semitones)",
                }
                pitch_mode = st.selectbox(
                    "Pitch source",
                    options=list(pitch_mode_map.keys()),
                    format_func=lambda key: pitch_mode_map[key],
                    index=list(pitch_mode_map.keys()).index(track.get("note_mode", "harmonic")),
                    key=f"track_pitch_mode_{track['id']}",
                )
                track["note_mode"] = pitch_mode

                if pitch_mode == "single":
                    track["note_name"] = st.selectbox(
                        "Target note",
                        options=piano_roll.ALL_NOTE_NAMES,
                        index=piano_roll.ALL_NOTE_NAMES.index(track.get("note_name", "C4"))
                        if track.get("note_name", "C4") in piano_roll.ALL_NOTE_NAMES
                        else piano_roll.ALL_NOTE_NAMES.index("C4"),
                        key=f"track_single_note_{track['id']}",
                        help="Replace molecule harmonics with a single sustained note.",
                    )
                elif pitch_mode == "chord":
                    default_notes = track.get("chord_notes") or []
                    track["chord_notes"] = st.multiselect(
                        "Chord notes",
                        options=piano_roll.ALL_NOTE_NAMES,
                        default=[note for note in default_notes if note in piano_roll.ALL_NOTE_NAMES],
                        key=f"track_chord_notes_{track['id']}",
                        help="Select one or more notes to form a custom chord drone.",
                    )
                elif pitch_mode == "transpose":
                    track["transpose_semitones"] = float(
                        st.slider(
                            "Transpose (semitones)",
                            min_value=-24.0,
                            max_value=24.0,
                            value=float(track.get("transpose_semitones", 0.0)),
                            step=0.5,
                            key=f"track_transpose_{track['id']}",
                            help="Shift every harmonic up or down in semitones.",
                        )
                    )

        with effects_tab:
            if track.get("source") != "upload":
                quantize = st.checkbox(
                    "Quantize to scale",
                    value=track.get("quantize_scale", False),
                    key=f"track_quantize_scale_{track['id']}",
                    help="Snap molecular frequencies to a musical scale for smoother chords.",
                )
                track["quantize_scale"] = quantize
                if quantize:
                    cols = st.columns([1, 1])
                    track["quantize_root"] = cols[0].selectbox(
                        "Scale root",
                        options=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
                        index=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"].index(
                            track.get("quantize_root", "C")
                        ),
                        key=f"track_quantize_root_{track['id']}",
                    )
                    track["quantize_mode"] = cols[1].selectbox(
                        "Scale",
                        options=["major", "minor", "dorian", "mixolydian", "lydian", "phrygian", "locrian"],
                        index=["major", "minor", "dorian", "mixolydian", "lydian", "phrygian", "locrian"].index(
                            track.get("quantize_mode", "major")
                        ),
                        key=f"track_quantize_mode_{track['id']}",
                    )

            track["gain_db"] = float(
                st.slider(
                    "Gain (dB)",
                    min_value=-18.0,
                    max_value=18.0,
                    value=float(track.get("gain_db", 0.0)),
                    step=0.5,
                    key=f"track_gain_{track['id']}",
                )
            )

            track["track_volume"] = float(
                st.slider(
                    "Volume (linear)",
                    min_value=0.0,
                    max_value=2.0,
                    value=float(track.get("track_volume", 1.0)),
                    step=0.05,
                    key=f"track_volume_{track['id']}",
                    help="Overall level multiplier applied after effects (1.0 = unity).",
                )
            )

            enable_highpass = st.checkbox(
                "Enable high-pass filter",
                value=track.get("highpass_cutoff") is not None,
                key=f"track_hp_enable_{track['id']}",
            )
            if enable_highpass:
                track["highpass_cutoff"] = float(
                    st.slider(
                        "High-pass cutoff (Hz)",
                        min_value=20.0,
                        max_value=1000.0,
                        value=float(track.get("highpass_cutoff", 120.0) or 120.0),
                        step=10.0,
                        key=f"track_hp_cutoff_{track['id']}",
                    )
                )
            else:
                track["highpass_cutoff"] = None

            enable_lowpass = st.checkbox(
                "Enable low-pass filter",
                value=track.get("lowpass_cutoff") is not None,
                key=f"track_lp_enable_{track['id']}",
            )
            if enable_lowpass:
                default_lp = float(track.get("lowpass_cutoff") or min(sample_rate // 2, 2000))
                track["lowpass_cutoff"] = float(
                    st.slider(
                        "Low-pass cutoff (Hz)",
                        min_value=200.0,
                        max_value=float(sample_rate // 2),
                        value=default_lp,
                        step=50.0,
                        key=f"track_lp_cutoff_{track['id']}",
                    )
                )
            else:
                track["lowpass_cutoff"] = None

            enable_delay = st.checkbox(
                "Enable delay",
                value=track.get("delay_seconds") is not None,
                key=f"track_delay_enable_{track['id']}",
            )
            if enable_delay:
                track["delay_seconds"] = float(
                    st.slider(
                        "Delay time (s)",
                        min_value=0.05,
                        max_value=1.5,
                        value=float(track.get("delay_seconds", 0.3) or 0.3),
                        step=0.05,
                        key=f"track_delay_time_{track['id']}",
                    )
                )
                track["delay_feedback"] = float(
                    st.slider(
                        "Delay feedback",
                        min_value=0.0,
                        max_value=0.95,
                        value=float(track.get("delay_feedback", 0.3)),
                        step=0.05,
                        key=f"track_delay_feedback_{track['id']}",
                    )
                )
                track["delay_mix"] = float(
                    st.slider(
                        "Delay mix",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(track.get("delay_mix", 0.25)),
                        step=0.05,
                        key=f"track_delay_mix_{track['id']}",
                    )
                )
            else:
                track["delay_seconds"] = None

            enable_reverb = st.checkbox(
                "Enable reverb",
                value=track.get("reverb_wet", 0.0) > 0,
                key=f"track_reverb_enable_{track['id']}",
            )
            if enable_reverb:
                track["reverb_wet"] = float(
                    st.slider(
                        "Reverb mix",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(track.get("reverb_wet", 0.25)),
                        step=0.05,
                        key=f"track_reverb_mix_{track['id']}",
                    )
                )
                track["reverb_size"] = float(
                    st.slider(
                        "Room size",
                        min_value=0.1,
                        max_value=1.0,
                        value=float(track.get("reverb_size", 0.4)),
                        step=0.05,
                        key=f"track_reverb_size_{track['id']}",
                    )
                )
                track["reverb_decay"] = float(
                    st.slider(
                        "Decay",
                        min_value=0.1,
                        max_value=0.99,
                        value=float(track.get("reverb_decay", 0.6)),
                        step=0.05,
                        key=f"track_reverb_decay_{track['id']}",
                    )
                )
            else:
                track["reverb_wet"] = 0.0

    return actions


def _collect_molecule_visuals(
    tracks: List[Dict[str, Any]],
    current: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect unique molecule visuals from tracks plus the current molecule."""

    visuals: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add_entry(smiles: str, preset_matches: Optional[List[chem_utils.FunctionalGroupMatch]] = None):
        if not smiles or smiles in seen:
            return
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        matches = preset_matches or chem_utils.find_functional_groups(mol)
        visuals.append({"smiles": smiles, "mol": mol, "matches": matches})
        seen.add(smiles)

    if current and current.get("smiles"):
        add_entry(str(current["smiles"]), current.get("matches"))

    for track in tracks:
        smiles = track.get("smiles")
        if isinstance(smiles, str):
            add_entry(smiles)

    return visuals


def render_piano_roll_section(
    *,
    info,
    components: List[piano_roll.AudioComponent],
    duration: float,
    sample_rate: int,
    matches: Optional[List[chem_utils.FunctionalGroupMatch]] = None,
    mapping_config: Optional[Dict[str, object]] = None,
):
    st.subheader("Piano roll arranger")

    tracks = _tracks_state()

    settings_state = st.session_state.setdefault(
        "piano_roll_settings",
        {
            "bar_count": 1,
            "render_length": duration,
            "grid_step_bars": 0.25,
            "tempo_bpm": 90,
            "swing": 0.0,
        },
    )

    base_clip_length = duration
    if tracks:
        base_clip_length = float(tracks[0].get("duration", duration) or duration)
    base_clip_length = max(base_clip_length, 0.1)

    tempo_bpm = st.slider(
        "Tempo (BPM)",
        min_value=40,
        max_value=180,
        value=int(settings_state.get("tempo_bpm", 90)),
        step=1,
        help="Sets bar length for grid snapping and timing cues.",
        key="piano_roll_tempo",
    )
    settings_state["tempo_bpm"] = tempo_bpm
    seconds_per_beat = 60.0 / tempo_bpm
    beats_per_bar = 4.0
    # Keep bar length aligned to tempo but never shorter than the base clip duration
    base_clip_length = max(beats_per_bar * seconds_per_beat, base_clip_length)

    bar_count = st.slider(
        "Arrangement bars (4/4)",
        min_value=1,
        max_value=64,
        value=int(settings_state.get("bar_count", 1)),
        step=1,
        help="Each bar is 4 beats. Tracks in continuous mode will loop to fill this time.",
        key="piano_roll_bar_count",
    )
    arrangement_length = base_clip_length * bar_count
    settings_state["bar_count"] = bar_count
    settings_state["render_length"] = arrangement_length
    settings_state["bar_length"] = base_clip_length
    grid_step_bars = st.select_slider(
        "Grid resolution (bars)",
        options=[1.0, 0.5, 0.25],
        value=float(settings_state.get("grid_step_bars", 0.25)),
        format_func=lambda v: f"{v} bar" if v == 1.0 else f"{int(1/v)} per bar",
        key="piano_roll_grid_step",
        help="Snap start positions to a DAW-like grid.",
    )
    settings_state["grid_step_bars"] = float(grid_step_bars)
    st.caption(
        f"Current length: {arrangement_length:.2f}s"
        f" (bar {base_clip_length:.2f}s @ {tempo_bpm} BPM × {bar_count} bars)."
        " Tracks default to continuous looping so you hear the clip repeat across bars."
    )

    swing = st.slider(
        "Swing (rhythmic shuffle)",
        min_value=0.0,
        max_value=0.2,
        step=0.01,
        value=float(settings_state.get("swing", 0.0)),
        help="Pushes every second grid step later for a swung feel.",
        key="piano_roll_swing",
    )
    settings_state["swing"] = swing
    st.caption(
        f"Swing {swing*100:.0f}% applied to grid; starts quantize to {float(grid_step_bars)} bar steps with swing timing."
    )

    add_cols = st.columns([2, 1, 2, 1])
    with add_cols[0]:
        st.markdown(
            "Layer multiple molecular clips, uploaded WAV files, and synth tweaks"
            " to sculpt richer arrangements."
        )
    with add_cols[1]:
        if st.button("Add molecule clip", key="pr_add_molecule"):
            _add_molecule_track(
                name=info.smiles,
                components=components,
                duration=duration,
                smiles=info.smiles,
                matches=matches,
            )
    with add_cols[2]:
        drum_cols = st.columns([1, 1])
        loop_len = drum_cols[0].selectbox(
            "Drum length (bars)",
            options=[1, 2, 4],
            index=[1, 2, 4].index(2),
            key="drum_loop_bars",
        )
        pattern = drum_cols[1].selectbox(
            "Pattern",
            options=["basic", "four_on_floor", "halftime", "syncopated", "two_step", "trap"],
            format_func=lambda p: {
                "basic": "Backbeat",
                "four_on_floor": "Four on the floor",
                "halftime": "Halftime",
                "syncopated": "Syncopated",
                "two_step": "2-step",
                "trap": "Trap-ish",
            }.get(p, p),
            key="drum_loop_pattern",
        )
        mix_cols = st.columns(4)
        kick_level = mix_cols[0].slider("Kick", 0.2, 1.5, 1.0, 0.05, key="drum_kick_level")
        snare_level = mix_cols[1].slider("Snare", 0.2, 1.5, 1.0, 0.05, key="drum_snare_level")
        hat_level = mix_cols[2].slider("Hat", 0.2, 1.5, 1.0, 0.05, key="drum_hat_level")
        drum_gain = mix_cols[3].slider("Drum gain", 0.5, 2.0, 1.3, 0.05, key="drum_gain")
        humanize_ms = st.slider(
            "Humanize timing (ms)",
            min_value=0.0,
            max_value=25.0,
            value=5.0,
            step=1.0,
            key="drum_humanize",
            help="Adds tiny timing jitter to avoid robotic loops.",
        )
        if st.button("Add drum loop", key="pr_add_drum_loop"):
            drum_payload = _synth_drum_loop(
                loop_len,
                settings_state.get("tempo_bpm", 90),
                settings_state.get("swing", 0.0),
                sample_rate,
                pattern,
                kick_level=kick_level,
                snare_level=snare_level,
                hat_level=hat_level,
                humanize_ms=humanize_ms,
                master_gain=drum_gain,
            )
            if drum_payload.get("bytes"):
                drum_settings = {
                    "loop_bars": loop_len,
                    "pattern": pattern,
                    "kick_level": kick_level,
                    "snare_level": snare_level,
                    "hat_level": hat_level,
                    "humanize_ms": humanize_ms,
                    "master_gain": drum_gain,
                    "tempo_bpm": settings_state.get("tempo_bpm", 90),
                    "swing": settings_state.get("swing", 0.0),
                }
                _add_drum_track(drum_payload, drum_settings)
    with add_cols[3]:
        uploaded_file = st.file_uploader(
            "Upload WAV clip",
            type=["wav"],
            key="piano_roll_upload_file",
            label_visibility="collapsed",
        )

    if uploaded_file is not None:
        clip_payload = _read_uploaded_clip(uploaded_file)
        if clip_payload is not None:
            st.caption(
                f"`{clip_payload['name']}` · {clip_payload['duration']:.2f} s"
                f" @ {clip_payload['sample_rate']} Hz"
            )
            if st.button("Add uploaded clip", key="pr_confirm_upload"):
                _add_uploaded_track(clip_payload)
                st.session_state["piano_roll_upload_file"] = None

    if not tracks:
        st.info(
            "No clips in the piano roll yet. Add a molecule or upload a clip to"
            " start building the arrangement."
        )
        return

    def _remap_track_components(track: Dict[str, Any]):
        if track.get("source") != "components":
            return
        payload = track.get("match_payload")
        if not payload:
            return
        total = sum(item.get("count", 0) for item in payload) or 1
        audible_range = mapping_config.get("audible_range", (100.0, 4000.0)) if mapping_config else (100.0, 4000.0)
        wrap = bool(mapping_config.get("wrap", False)) if mapping_config else False
        wrap_band = mapping_config.get("wrap_band", (110.0, 880.0)) if mapping_config else (110.0, 880.0)
        remapped: List[piano_roll.AudioComponent] = []
        for item in payload:
            wn = float(item.get("wn", 0.0))
            count = float(item.get("count", 0.0))
            freq = audio_utils.map_wavenumber_to_audible(wn, audible_range=audible_range)
            if wrap:
                freq = audio_utils._wrap_frequency_to_band(freq, low=wrap_band[0], high=wrap_band[1])  # type: ignore[attr-defined]
            remapped.append((freq, count / total))
        track["components"] = remapped

    for track in tracks:
        _remap_track_components(track)

    grid_step_seconds = base_clip_length * float(grid_step_bars)

    def _apply_groove(start: float) -> float:
        if grid_step_seconds <= 0:
            return max(0.0, start)
        quantized = round(start / grid_step_seconds) * grid_step_seconds
        step_idx = int(round(quantized / grid_step_seconds))
        swing_amount = float(settings_state.get("swing", 0.0) or 0.0)
        if swing_amount > 0 and step_idx % 2 == 1:
            quantized += swing_amount * grid_step_seconds
        quantized = max(0.0, quantized)
        if arrangement_length:
            quantized = min(quantized, max(0.0, arrangement_length - 0.01))
        return quantized

    quantized_starts: Dict[str, float] = {}
    for t in tracks:
        quantized_starts[t["id"]] = _apply_groove(float(t.get("start_time", 0.0) or 0.0))

    _render_transport_controls(tracks)

    pending_preview: Optional[str] = None
    removal_ids: List[str] = []
    duplicate_payloads: List[Dict[str, Any]] = []
    duplicate_shift_payloads: List[Dict[str, Any]] = []

    for track in list(tracks):
        # Ensure bar gating defaults exist
        track.setdefault("bar_start", 1)
        track.setdefault("bar_end", bar_count)
        actions = _render_track_editor(
            track,
            sample_rate=sample_rate,
            bar_count=bar_count,
            bar_length=base_clip_length,
            grid_step_bars=float(grid_step_bars),
            quantized_start=quantized_starts.get(track["id"]),
            tempo_bpm=tempo_bpm,
            swing=swing,
        )
        if actions["remove"]:
            removal_ids.append(track["id"])
        if actions["preview"]:
            pending_preview = track["id"]
        if actions.get("duplicate"):
            clone = copy.deepcopy(track)
            clone["id"] = uuid.uuid4().hex
            clone["name"] = f"{track.get('name', 'Track')} copy"
            clone["is_muted"] = False
            clone["is_solo"] = False
            duplicate_payloads.append(clone)
        if actions.get("duplicate_shifted"):
            clone = copy.deepcopy(track)
            clone["id"] = uuid.uuid4().hex
            clone["name"] = f"{track.get('name', 'Track')} copy"
            clone["is_muted"] = False
            clone["is_solo"] = False
            clone["start_time"] = max(0.0, float(track.get("start_time", 0.0) or 0.0) + base_clip_length)
            duplicate_shift_payloads.append(clone)
        if actions.get("regen_drum"):
            drum_settings = track.get("drum_settings", {})
            loop_bars = int(drum_settings.get("loop_bars", 2) or 2)
            pattern = drum_settings.get("pattern", "basic")
            kick_level = float(drum_settings.get("kick_level", 1.0) or 1.0)
            snare_level = float(drum_settings.get("snare_level", 1.0) or 1.0)
            hat_level = float(drum_settings.get("hat_level", 1.0) or 1.0)
            humanize_ms = float(drum_settings.get("humanize_ms", 5.0) or 0.0)
            drum_payload = _synth_drum_loop(
                loop_bars,
                tempo_bpm if tempo_bpm is not None else drum_settings.get("tempo_bpm", 90),
                swing if swing is not None else drum_settings.get("swing", 0.0),
                sample_rate,
                pattern,
                kick_level=kick_level,
                snare_level=snare_level,
                hat_level=hat_level,
                humanize_ms=humanize_ms,
                master_gain=float(drum_settings.get("master_gain", 1.0)),
            )
            if drum_payload.get("bytes"):
                track["audio_bytes"] = drum_payload.get("bytes")
                track["upload_sample_rate"] = drum_payload.get("sample_rate")
                track["duration"] = drum_payload.get("duration", track.get("duration", 0.1))
                track["drum_settings"] = {
                    "loop_bars": loop_bars,
                    "pattern": pattern,
                    "kick_level": kick_level,
                    "snare_level": snare_level,
                    "hat_level": hat_level,
                    "humanize_ms": humanize_ms,
                    "master_gain": float(drum_settings.get("master_gain", 1.0)),
                    "tempo_bpm": tempo_bpm if tempo_bpm is not None else drum_settings.get("tempo_bpm", 90),
                    "swing": swing if swing is not None else drum_settings.get("swing", 0.0),
                }
                st.rerun()

    if removal_ids:
        st.session_state["piano_roll_tracks"] = [
            t for t in tracks if t["id"] not in removal_ids
        ]
        tracks = _tracks_state()

    if duplicate_payloads:
        _tracks_state().extend(duplicate_payloads)
        tracks = _tracks_state()
    if duplicate_shift_payloads:
        _tracks_state().extend(duplicate_shift_payloads)
        tracks = _tracks_state()

    tracks_for_render = copy.deepcopy(tracks)
    for t in tracks_for_render:
        t["start_time"] = _apply_groove(float(t.get("start_time", 0.0) or 0.0))

    arrangement = piano_roll.build_arrangement(
        tracks_for_render,
        sample_rate=sample_rate,
        render_length=arrangement_length,
        bar_length=base_clip_length,
    )
    waveform = arrangement.get("waveform")
    events = arrangement.get("events") or []
    summary = arrangement.get("summary") or []

    transport = _transport_state()
    if isinstance(waveform, np.ndarray) and waveform.size:
        audio_bytes = audio_utils.waveform_to_wav_bytes(waveform, sample_rate=sample_rate)

        if transport.get("playing") and "pr_loop_player_autoplay" not in st.session_state:
            st.session_state["pr_loop_player_autoplay"] = True

        loop_col, autoplay_col = st.columns([1, 1])
        loop_enabled = loop_col.toggle(
            "Loop arrangement",
            value=st.session_state.get("pr_loop_player_loop", True),
            key="pr_loop_player_loop",
            help="Keep the arrangement running while you tweak tracks and effects.",
        )
        autoplay_enabled = autoplay_col.toggle(
            "Auto-play on change",
            value=st.session_state.get(
                "pr_loop_player_autoplay", transport.get("playing", False)
            ),
            key="pr_loop_player_autoplay",
            help="Restart playback after edits for near real-time feedback.",
        )

        audio_element_id = render_looping_audio_player(
            audio_bytes=audio_bytes,
            waveform=np.array([], dtype=np.float32),  # skip waveform viz to avoid overlap
            sample_rate=sample_rate,
            loop=loop_enabled,
            autoplay=autoplay_enabled,
            height=120,
        )
        # Collect visuals for all molecules present in the session (current + tracks)
        current_context = {"smiles": info.smiles, "matches": matches}
        visuals = _collect_molecule_visuals(tracks, current_context)
        if visuals:
            st.markdown("**FTIR vibration visualisers**")
            vis_cols = st.columns(min(3, len(visuals)))
            for idx, visual in enumerate(visuals):
                col = vis_cols[idx % len(vis_cols)]
                with col:
                    render_molecule_visualizer(
                        mol=visual["mol"],
                        matches=visual["matches"],
                        audio_element_id=audio_element_id,
                        audible_range=mapping_config.get("audible_range", (100.0, 4000.0)) if mapping_config else (100.0, 4000.0),
                        wrap=bool(mapping_config.get("wrap", False)) if mapping_config else False,
                        wrap_band=mapping_config.get("wrap_band", (110.0, 880.0)) if mapping_config else (110.0, 880.0),
                        width=520,
                        height=360,
                    )
        st.download_button(
            "Download arrangement",
            data=audio_bytes,
            file_name="arrangement.wav",
            mime="audio/wav",
            key="pr_download_arrangement",
        )
    else:
        st.warning("No audio rendered – adjust track settings or enable playback.")

    if summary:
        st.dataframe(pd.DataFrame(summary), hide_index=True)

    figure = _build_piano_roll_figure(
        events,
        render_length=arrangement_length,
        bar_length=base_clip_length,
        grid_step=float(grid_step_seconds) if grid_step_seconds else None,
    )
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info(
            "Install `plotly` to visualise clips on the piano roll grid, or use"
            " the summary table for timing information."
        )

    if pending_preview:
        track = next((t for t in tracks if t["id"] == pending_preview), None)
        if track is not None:
            preview_waveform, _, _, _ = piano_roll.render_track_audio(
                track,
                sample_rate=sample_rate,
                render_length=arrangement_length,
            )
            if preview_waveform.size:
                preview_bytes = audio_utils.waveform_to_wav_bytes(
                    preview_waveform, sample_rate=sample_rate
                )
                st.audio(preview_bytes, format="audio/wav")
            else:
                st.warning("Track produced no audio with the current settings.")
