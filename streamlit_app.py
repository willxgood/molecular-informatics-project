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
    mapping_mode = st.sidebar.selectbox(
        "Audio mapping",
        options=["Wide (unwrapped, 100–4000 Hz)", "Musical (wrapped midrange)"],
        help="Choose a wider, spacing-preserving band or a wrapped midrange for smoother tone.",
    )
    if mapping_mode.startswith("Wide"):
        mapping_config = {
            "audible_range": (100.0, 4000.0),
            "wrap": False,
            "wrap_band": (110.0, 880.0),
            "label": "Wide",
        }
    else:
        mapping_config = {
            "audible_range": (220.0, 1760.0),
            "wrap": True,
            "wrap_band": (110.0, 880.0),
            "label": "Musical",
        }
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "mapping_config": mapping_config,
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
    mapping_config: Optional[Dict[str, object]] = None,
):
    st.subheader(heading)
    mapping_config = mapping_config or {
        "audible_range": (100.0, 4000.0),
        "wrap": False,
        "wrap_band": (110.0, 880.0),
        "label": "Wide",
    }
    present_matches = [m for m in matches if m.present]
    components = audio_utils.groups_to_audio_components(
        present_matches,
        audible_range=mapping_config["audible_range"],  # type: ignore[arg-type]
        wrap=bool(mapping_config.get("wrap", False)),
        wrap_band=mapping_config.get("wrap_band", (110.0, 880.0)),  # type: ignore[arg-type]
    )
    if not components:
        st.info("No functional groups detected for audio synthesis.")
        return None

    st.markdown(
        "The following audible frequencies are linearly mapped from the"
        " FTIR center wavenumbers."
    )
    # Compute per-match frequencies and contributions so the table columns stay aligned
    total_occurrences = sum(m.match_count for m in present_matches) or 1
    def _map_freq(wn: float) -> float:
        freq = audio_utils.map_wavenumber_to_audible(
            wn, audible_range=mapping_config["audible_range"]  # type: ignore[arg-type]
        )
        if mapping_config.get("wrap", False):
            freq = audio_utils._wrap_frequency_to_band(  # type: ignore[attr-defined]
                freq, low=mapping_config.get("wrap_band", (110.0, 880.0))[0], high=mapping_config.get("wrap_band", (110.0, 880.0))[1]
            )
        return freq

    display_frequencies = [_map_freq(m.group.center_wavenumber) for m in present_matches]
    contributions = [
        f"{(m.match_count / total_occurrences) * 100:.1f}" for m in present_matches
    ]
    freq_table = pd.DataFrame(
        {
            "Functional group": [m.group.name for m in present_matches],
            "Center (cm⁻¹)": [m.group.center_wavenumber for m in present_matches],
            "Occurrences": [m.match_count for m in present_matches],
            "Contribution (%)": contributions,
            "Audio frequency (Hz)": [f"{freq:.1f}" for freq in display_frequencies],
        }
    )
    st.dataframe(freq_table, hide_index=True)
    with st.expander("How are FTIR wavenumbers mapped to audio?"):
        aud_min, aud_max = mapping_config["audible_range"]  # type: ignore[index]
        wrap_note = (
            f"wrapped into {mapping_config.get('wrap_band', (110.0, 880.0))[0]:.0f}–"
            f"{mapping_config.get('wrap_band', (110.0, 880.0))[1]:.0f} Hz for a smoother band."
            if mapping_config.get("wrap", False)
            else "unwrapped to preserve spacing."
        )
        st.markdown(
            f"- Current mode: {mapping_config.get('label', 'Wide')} — map 400–4000 cm⁻¹ linearly to"
            f" {aud_min:.0f}–{aud_max:.0f} Hz, {wrap_note}\n"
            "- A physical conversion (`ν = wavenumber × c`) would land in the 10¹³–10¹⁴ Hz infrared range, far above hearing; raw use would alias. The mapping is intentionally musical/educational."
        )

    waveform = audio_utils.generate_waveform(
        components, duration=duration, sample_rate=sample_rate
    )
    audio_bytes = audio_utils.waveform_to_wav_bytes(waveform, sample_rate=sample_rate)
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
        " Playback is consolidated in the piano roll – add the molecule clip there to hear it."
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
    mapping_config = inputs["mapping_config"]

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
    audio_context = render_audio_section(
        matches,
        duration,
        sample_rate,
        mapping_config=mapping_config,
    )
    if audio_context is not None:
        piano_roll_ui.render_piano_roll_section(
            info=info,
            components=audio_context["components"],
            duration=duration,
            sample_rate=sample_rate,
            matches=audio_context["matches"],
            mapping_config=mapping_config,
        )


if __name__ == "__main__":
    main()
