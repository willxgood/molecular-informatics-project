"""Streamlit UI helpers for the piano roll arranger."""
from __future__ import annotations

import io
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from components.looping_player import render_looping_audio_player
from . import audio_utils, piano_roll


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
):
    track = _default_track_payload(name=name, duration=duration, components=components)
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


def _build_piano_roll_figure(
    events: List[piano_roll.PianoRollEvent],
    *,
    render_length: Optional[float],
    bar_length: Optional[float],
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
                    line=dict(color="rgba(0,0,0,0.15)", width=1),
                    layer="below",
                )
            )
            bar_line += bar_length

    fig.update_layout(
        barmode="overlay",
        bargap=0.15,
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
) -> Dict[str, bool]:
    actions = {"remove": False, "preview": False}
    header = f"{track.get('name', 'Track')}"
    if track.get("source") == "upload" and track.get("upload_name"):
        header += f" · {track['upload_name']}"

    with st.expander(header, expanded=False):
        top_cols = st.columns([1, 1, 1, 1, 1])
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
        if top_cols[4].button("Remove", key=f"track_remove_{track['id']}"):
            actions["remove"] = True

        track["name"] = st.text_input(
            "Track name",
            value=track.get("name", "Track"),
            key=f"track_name_{track['id']}",
        )

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

        with synth_tab:
            if track.get("source") == "upload":
                st.info("Uploaded clips bypass the synth oscillator controls.")
            else:
                preset = st.selectbox(
                    "Tone preset",
                    options=["Custom", "Soft pad", "Lo-fi"],
                    index=["Custom", "Soft pad", "Lo-fi"].index(track.get("tone_preset", "Custom")),
                    key=f"track_tone_preset_{track['id']}",
                    help="Quickly soften or texture the sound; choose Custom to dial in manually.",
                )
                track["tone_preset"] = preset

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


def render_piano_roll_section(
    *,
    info,
    components: List[piano_roll.AudioComponent],
    duration: float,
    sample_rate: int,
):
    st.subheader("Piano roll arranger")

    tracks = _tracks_state()

    settings_state = st.session_state.setdefault(
        "piano_roll_settings",
        {"bar_count": 1, "render_length": duration, "grid_step_bars": 0.25},
    )

    base_clip_length = duration
    if tracks:
        base_clip_length = float(tracks[0].get("duration", duration) or duration)
    base_clip_length = max(base_clip_length, 0.1)

    bar_count = st.slider(
        "Arrangement bars (multiples of clip length)",
        min_value=1,
        max_value=64,
        value=int(settings_state.get("bar_count", 1)),
        step=1,
        help="Each bar repeats the base clip length. Tracks in continuous mode will loop to fill this time.",
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
        f" (clip {base_clip_length:.2f}s × {bar_count} bars)."
        " Tracks default to continuous looping so you hear the clip repeat across bars."
    )

    add_cols = st.columns([2, 1, 1])
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
            )
    with add_cols[2]:
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

    _render_transport_controls(tracks)

    pending_preview: Optional[str] = None
    removal_ids: List[str] = []

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
        )
        if actions["remove"]:
            removal_ids.append(track["id"])
        if actions["preview"]:
            pending_preview = track["id"]

    if removal_ids:
        st.session_state["piano_roll_tracks"] = [
            t for t in tracks if t["id"] not in removal_ids
        ]
        tracks = _tracks_state()

    arrangement = piano_roll.build_arrangement(
        tracks,
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

        render_looping_audio_player(
            audio_bytes=audio_bytes,
            waveform=waveform,
            sample_rate=sample_rate,
            loop=loop_enabled,
            autoplay=autoplay_enabled,
            height=260,
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
