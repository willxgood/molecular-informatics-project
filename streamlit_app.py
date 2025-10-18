"""Streamlit application for turning molecules into FTIR-inspired audio."""
from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
import streamlit as st

from molecular_informatics import audio_utils, chem_utils, reaction_ui


def parse_multiline(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_molecules(queries: List[str], label: str) -> List[chem_utils.MoleculeInfo]:
    infos: List[chem_utils.MoleculeInfo] = []
    for query in queries:
        try:
            infos.append(chem_utils.resolve_molecule(query))
        except chem_utils.MoleculeResolutionError as exc:
            st.error(f"{label} '{query}': {exc}")
            st.stop()
    if not infos:
        st.error(f"Please provide at least one {label.lower()}.")
        st.stop()
    return infos
def render_sidebar() -> Dict[str, object]:
    st.sidebar.title("Configuration")
    mode = st.sidebar.radio("Mode", ("Single molecule", "Reaction"))
    duration = st.sidebar.slider("Audio duration (s)", 1.0, 10.0, 4.0, 0.5)
    sample_rate = st.sidebar.select_slider(
        "Sample rate", options=[22050, 32000, 44100], value=44100
    )

    if mode == "Single molecule":
        st.sidebar.write("Enter a SMILES string.")
        query = st.sidebar.text_input("Molecule", "CCO")
        return {
            "mode": "single",
            "query": query,
            "duration": duration,
            "sample_rate": sample_rate,
        }

    st.sidebar.write("Provide SMILES strings for reactants and products (one per line).")
    reactants_text = st.sidebar.text_area(
        "Reactants", "CO\nCC(=O)O", height=96
    )
    products_text = st.sidebar.text_area("Products", "COC(=O)C\nO", height=96)
    return {
        "mode": "reaction",
        "reactants_text": reactants_text,
        "products_text": products_text,
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
        return

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

    if inputs["mode"] == "single":
        query = inputs["query"]
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
        return

    reactant_queries = parse_multiline(inputs["reactants_text"])
    product_queries = parse_multiline(inputs["products_text"])

    reactant_infos = resolve_molecules(reactant_queries, "Reactant")
    product_infos = resolve_molecules(product_queries, "Product")

    reaction_ui.render_molecule_summary("Reactants", reactant_queries, reactant_infos)
    reaction_ui.render_molecule_summary("Products", product_queries, product_infos)

    reactant_match_sets = [
        chem_utils.find_functional_groups(info.mol) for info in reactant_infos
    ]
    product_match_sets = [
        chem_utils.find_functional_groups(info.mol) for info in product_infos
    ]
    reactant_matches = chem_utils.aggregate_group_matches(reactant_match_sets)
    product_matches = chem_utils.aggregate_group_matches(product_match_sets)
    deltas = chem_utils.compute_group_deltas(reactant_matches, product_matches)

    render_ftir_table(reactant_matches, heading="Reactant FTIR features")
    render_audio_section(
        reactant_matches, duration, sample_rate, heading="Reactant soundscape"
    )

    render_ftir_table(product_matches, heading="Product FTIR features")
    render_audio_section(
        product_matches, duration, sample_rate, heading="Product soundscape"
    )

    reaction_ui.render_reaction_changes(deltas, duration, sample_rate)


if __name__ == "__main__":
    main()
