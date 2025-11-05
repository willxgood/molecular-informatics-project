"""Streamlit application for turning molecules into FTIR-inspired audio."""
from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
import streamlit as st

from molecular_informatics import audio_utils, chem_utils, piano_roll_ui

try:
    from streamlit_ketcher import st_ketcher
except ImportError:  # pragma: no cover - optional dependency
    st_ketcher = None  # type: ignore[assignment]

def render_sidebar():
    st.sidebar.title("Configuration")
    duration = st.sidebar.slider("Audio duration (s)", 1.0, 10.0, 4.0, 0.5)
    sample_rate = st.sidebar.select_slider(
        "Sample rate", options=[22050, 32000, 44100], value=44100
    )
    return {
        "duration": duration,
        "sample_rate": sample_rate,
    }


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


def render_ftir_table(
    matches: List[chem_utils.FunctionalGroupMatch],
    heading: str = "Predicted FTIR features",
):
    st.subheader(heading)
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
    heading: str = "Molecular soundscape",
):
    st.subheader(heading)
    present_matches = [m for m in matches if m.present]
    components = audio_utils.groups_to_audio_components(present_matches)
    if not components:
        st.info("No functional groups detected for audio synthesis.")
        return None

    st.markdown(
        "The following audible frequencies are linearly mapped from the"
        " FTIR center wavenumbers."
    )
    freq_table = pd.DataFrame(
        {
            "Functional group": [m.group.name for m in present_matches],
            "Center (cm⁻¹)": [m.group.center_wavenumber for m in present_matches],
            "Occurrences": [m.match_count for m in present_matches],
            "Contribution (%)": [f"{weight * 100:.1f}" for _, weight in components],
            "Audio frequency (Hz)": [f"{freq:.1f}" for freq, _ in components],
        }
    )
    st.dataframe(freq_table, hide_index=True)

    waveform = audio_utils.generate_waveform(
        components, duration=duration, sample_rate=sample_rate
    )
    audio_bytes = audio_utils.waveform_to_wav_bytes(waveform, sample_rate=sample_rate)
    st.audio(audio_bytes, format="audio/wav")
    download_key = f"download_{heading.replace(' ', '_').lower()}"
    st.download_button(
        "Download WAV",
        data=audio_bytes,
        file_name="molecule.wav",
        mime="audio/wav",
        key=download_key,
    )
    st.caption(
        "Add or remove functional groups to reshape the sonic palette."
        " Try adjusting the duration to hear sustained harmonics."
    )
    return {
        "components": components,
        "waveform": waveform,
        "audio_bytes": audio_bytes,
        "matches": present_matches,
    }

def render_ketcher_editor(initial_smiles: str, *, key: str):
    """Render the Ketcher drawing widget when available."""
    if st_ketcher is None:
        st.info(
            "Install `streamlit-ketcher` to enable the interactive drawing tool."
            " Falling back to manual SMILES entry."
        )
        return None

    try:
        return st_ketcher(value=initial_smiles, key=key)
    except Exception as exc:  # pragma: no cover - defensive
        st.warning(f"Unable to render Ketcher editor: {exc}")
        return None


def main():
    st.set_page_config(page_title="Molecular FTIR Sound Generator", layout="wide")
    st.title("Molecular FTIR Sound Generator")
    st.write(
        "This app converts molecular structures into a musical interpretation"
        " of their predicted FTIR absorptions."
    )

    inputs = render_sidebar()
    duration = inputs["duration"]
    sample_rate = inputs["sample_rate"]

    if "smiles_input" not in st.session_state:
        st.session_state["smiles_input"] = "CCO"
    if "smiles_input_pending" in st.session_state:
        st.session_state["smiles_input"] = st.session_state.pop("smiles_input_pending")

    st.subheader("Molecule input")
    st.text_input("SMILES string", key="smiles_input")
    st.caption("Paste a SMILES string or draw the molecule below to generate audio.")

    drawn_smiles = render_ketcher_editor(
        st.session_state["smiles_input"], key="ketcher_editor"
    )
    if isinstance(drawn_smiles, str):
        updated_smiles = drawn_smiles.strip()
        if updated_smiles and updated_smiles != st.session_state["smiles_input"]:
            st.session_state["smiles_input_pending"] = updated_smiles
            st.rerun()

    query = st.session_state["smiles_input"].strip()
    if not query:
        st.info("Enter a SMILES string or use the drawing tool to begin.")
        st.stop()

    try:
        info = chem_utils.resolve_molecule(query)
    except chem_utils.MoleculeResolutionError as exc:
        st.error(str(exc))
        st.stop()

    render_molecule_info(info)

    matches = chem_utils.find_functional_groups(info.mol)
    render_ftir_table(matches)
    audio_context = render_audio_section(matches, duration, sample_rate)
    if audio_context is not None:
        piano_roll_ui.render_piano_roll_section(
            info=info,
            components=audio_context["components"],
            duration=duration,
            sample_rate=sample_rate,
        )


if __name__ == "__main__":
    main()
