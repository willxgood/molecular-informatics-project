"""UI helpers for reaction visualisations in the Streamlit app."""
from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from . import audio_utils, chem_utils


def render_molecule_summary(
    title: str, queries: List[str], infos: List[chem_utils.MoleculeInfo]
):
    """Display a simple table summarising resolved molecules."""

    rows = []
    for query, info in zip(queries, infos):
        rows.append(
            {
                "Input": query,
                "Canonical SMILES": info.smiles,
                "Formula": info.formula,
            }
        )

    st.subheader(title)
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_reaction_changes(
    deltas: List[chem_utils.FunctionalGroupDelta],
    duration: float,
    sample_rate: int,
):
    """Visualise the net change in functional groups and render the delta audio."""

    nonzero = [delta for delta in deltas if delta.delta != 0]
    if not nonzero:
        st.subheader("Functional group changes")
        st.info(
            "No net functional group change detected between reactants and products."
        )
        return

    total_change = sum(abs(delta.delta) for delta in nonzero)
    rows = []
    components = []
    for delta in nonzero:
        base_freq = audio_utils.map_wavenumber_to_audible(
            delta.group.center_wavenumber
        )
        freq = base_freq * 0.9 if delta.delta < 0 else base_freq
        weight = abs(delta.delta) / total_change if total_change else 0.0
        trend = "Increase" if delta.delta > 0 else "Decrease"
        rows.append(
            {
                "Functional group": delta.group.name,
                "Reactants": delta.reactant_count,
                "Products": delta.product_count,
                "Change": delta.delta,
                "Trend": trend,
                "Contribution (%)": f"{weight * 100:.1f}",
                "Audio frequency (Hz)": f"{freq:.1f}",
            }
        )
        components.append((freq, weight))

    st.subheader("Functional group changes")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    waveform = audio_utils.generate_waveform(
        components, duration=duration, sample_rate=sample_rate
    )
    audio_bytes = audio_utils.waveform_to_wav_bytes(waveform, sample_rate=sample_rate)
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        "Download reaction delta WAV",
        data=audio_bytes,
        file_name="reaction_delta.wav",
        mime="audio/wav",
        key="download_reaction_delta",
    )
    st.caption(
        "Positive changes keep their mapped frequency while disappearing groups"
        " are shifted slightly lower to distinguish their contribution."
    )
