"""Streamlit application for turning molecules into FTIR-inspired audio."""
from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from molecular_informatics import audio_utils, chem_utils


def render_sidebar():
    st.sidebar.title("Molecule input")
    st.sidebar.write(
        "Enter a SMILES string, PubChem CID, or a chemical name."
    )
    query = st.sidebar.text_input("Molecule", "ethanol")
    duration = st.sidebar.slider("Audio duration (s)", 1.0, 10.0, 4.0, 0.5)
    sample_rate = st.sidebar.select_slider(
        "Sample rate", options=[22050, 32000, 44100], value=44100
    )
    return query, duration, sample_rate


def render_molecule_info(info: chem_utils.MoleculeInfo):
    st.subheader("Molecule overview")
    cols = st.columns([1, 2])
    with cols[0]:
        st.image(chem_utils.get_molecule_image(info.mol, size=300), caption=info.smiles)
    with cols[1]:
        st.markdown(
            f"**Canonical SMILES:** `{info.smiles}`\n\n"
            f"**Formula:** {info.formula}"
        )


def render_ftir_table(matches: List[chem_utils.FunctionalGroupMatch]):
    st.subheader("Predicted FTIR features")
    summary = chem_utils.summarise_groups(matches)
    df = pd.DataFrame(summary)
    present_mask = df["Detected"].eq("Yes")
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Range (cm⁻¹)": st.column_config.TextColumn("Range (cm⁻¹)"),
            "Center (cm⁻¹)": st.column_config.TextColumn("Center (cm⁻¹)"),
        },
    )
    st.caption(
        "Matches are determined using SMARTS substructure searches."
        " Extend `molecular_informatics/ftir_data.py` to support additional"
        " functional groups."
    )
    st.json(json.loads(df[present_mask].to_json(orient="records")))


def render_audio_section(
    matches: List[chem_utils.FunctionalGroupMatch],
    duration: float,
    sample_rate: int,
):
    st.subheader("Molecular soundscape")
    freqs = audio_utils.groups_to_audio_frequencies(matches)
    if not freqs:
        st.info("No functional groups detected for audio synthesis.")
        return

    st.markdown(
        "The following audible frequencies are linearly mapped from the"
        " FTIR center wavenumbers."
    )
    freq_table = pd.DataFrame(
        {
            "Functional group": [m.group.name for m in matches if m.present],
            "Center (cm⁻¹)": [m.group.center_wavenumber for m in matches if m.present],
            "Audio frequency (Hz)": [f"{freq:.1f}" for freq in freqs],
        }
    )
    st.dataframe(freq_table, hide_index=True)

    waveform = audio_utils.generate_waveform(freqs, duration=duration, sample_rate=sample_rate)
    audio_bytes = audio_utils.waveform_to_wav_bytes(waveform, sample_rate=sample_rate)
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        "Download WAV",
        data=audio_bytes,
        file_name="molecule.wav",
        mime="audio/wav",
    )
    st.caption(
        "Add or remove functional groups to reshape the sonic palette."
        " Try adjusting the duration to hear sustained harmonics."
    )


def main():
    st.set_page_config(page_title="Molecular FTIR Sound Generator", layout="wide")
    st.title("Molecular FTIR Sound Generator")
    st.write(
        "This app converts molecular structures into a musical interpretation"
        " of their predicted FTIR absorptions."
    )

    query, duration, sample_rate = render_sidebar()

    if not query:
        st.stop()

    try:
        info = chem_utils.resolve_molecule(query)
    except chem_utils.MoleculeResolutionError as exc:
        st.error(str(exc))
        st.stop()

    render_molecule_info(info)

    matches = chem_utils.find_functional_groups(info.mol)
    render_ftir_table(matches)
    render_audio_section(matches, duration, sample_rate)


if __name__ == "__main__":
    main()
