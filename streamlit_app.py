"""Streamlit application for turning molecules into FTIR-inspired audio."""
from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd
import streamlit as st

from molecular_informatics import audio_utils, chem_utils

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
    molecule_info: Optional[chem_utils.MoleculeInfo] = None,
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

    medicinal_profile: Optional[audio_utils.MedicinalSoundProfile] = None

    with st.expander("Medicinal chemistry sound design", expanded=False):
        auto_enabled = st.checkbox(
            "Auto apply PubChem sound design",
            value=False,
            help="When enabled, tries to infer activity, selectivity, toxicity, and bioavailability from PubChem annotations.",
        )

        manual_controls_needed = True
        if auto_enabled:
            if chem_utils.requests is None:
                st.warning("Install the 'requests' package to enable PubChem integration.")
            elif molecule_info is None:
                st.info("Resolve a molecule first to enable automatic sound design.")
            else:
                cid_resolved = molecule_info.pubchem_cid is not None
                if not cid_resolved:
                    with st.spinner("Resolving PubChem CID from SMILES..."):
                        cid = chem_utils.lookup_pubchem_cid(molecule_info.smiles)
                    if cid is not None:
                        molecule_info.pubchem_cid = cid
                        cid_resolved = True
                    else:
                        st.info(
                            "Unable to resolve a PubChem CID from the supplied SMILES. Try a recognised compound name or toggle the controls manually."
                        )
                if cid_resolved:
                    with st.spinner("Fetching PubChem medicinal annotations..."):
                        summary = chem_utils.fetch_pubchem_medicinal_summary(molecule_info.pubchem_cid)
                    if summary and summary.has_data():
                        def fmt_intensity(value: Optional[float]) -> str:
                            if value is None:
                                return "—"
                            return f"{value * 100:.0f}%"

                        def fmt_evidence(value: Optional[str]) -> str:
                            return value if value else "—"

                        medicinal_profile = audio_utils.MedicinalSoundProfile(
                            activity=summary.activity,
                            activity_strength=summary.activity_strength,
                            selectivity=summary.selectivity,
                            selectivity_strength=summary.selectivity_strength,
                            toxicity=summary.toxicity,
                            toxicity_strength=summary.toxicity_strength,
                            bioavailability=summary.bioavailability,
                            bioavailability_strength=summary.bioavailability_strength,
                        )
                        effects_table = []
                        if summary.activity:
                            effects_table.append(
                                {
                                    "Cue": "Activity",
                                    "Status": summary.activity,
                                    "Intensity (%)": fmt_intensity(summary.activity_strength),
                                    "Evidence": fmt_evidence(summary.activity_evidence),
                                }
                            )
                        if summary.selectivity:
                            effects_table.append(
                                {
                                    "Cue": "Selectivity",
                                    "Status": summary.selectivity,
                                    "Intensity (%)": fmt_intensity(summary.selectivity_strength),
                                    "Evidence": fmt_evidence(summary.selectivity_evidence),
                                }
                            )
                        if summary.toxicity:
                            effects_table.append(
                                {
                                    "Cue": "Toxicity",
                                    "Status": summary.toxicity,
                                    "Intensity (%)": fmt_intensity(summary.toxicity_strength),
                                    "Evidence": fmt_evidence(summary.toxicity_evidence),
                                }
                            )
                        if summary.bioavailability:
                            effects_table.append(
                                {
                                    "Cue": "Bioavailability",
                                    "Status": summary.bioavailability,
                                    "Intensity (%)": fmt_intensity(summary.bioavailability_strength),
                                    "Evidence": fmt_evidence(summary.bioavailability_evidence),
                                }
                            )
                        if effects_table:
                            manual_controls_needed = False
                            st.markdown("**PubChem sound design cues**")
                            st.dataframe(pd.DataFrame(effects_table), hide_index=True)
                        else:
                            st.info(
                                "PubChem returned pharmacological data, but no clear activity, selectivity, toxicity, or bioavailability cues were detected."
                            )
                    else:
                        st.info(
                            "PubChem returned no medicinal annotations for this compound."
                        )

        if manual_controls_needed:
            activity_choice = st.selectbox(
                "Activity profile",
                options=["Off", "Active", "Inactive"],
                index=0,
                help="Active compounds sound stronger, inactive ones are muted.",
            )
            selectivity_enabled = st.toggle(
                "Selective polish",
                value=False,
                help="Selective profiles get a cleaner rendering.",
            )
            toxicity_enabled = st.toggle(
                "Toxic distortion",
                value=False,
                help="Introduce a distorted edge for toxic liabilities.",
            )
            bioavailability_enabled = st.toggle(
                "Bioavailable smoothing",
                value=False,
                help="Smooth out the waveform to represent bioavailable compounds.",
            )
            manual_effects = []
            activity_value: Optional[str] = None
            activity_strength: Optional[float] = None
            if activity_choice == "Active":
                activity_value = "active"
                activity_strength = 1.0
                manual_effects.append(
                    {
                        "Cue": "Activity",
                        "Status": "active",
                        "Intensity (%)": "100%",
                        "Evidence": "Manual selection",
                    }
                )
            elif activity_choice == "Inactive":
                activity_value = "inactive"
                activity_strength = 1.0
                manual_effects.append(
                    {
                        "Cue": "Activity",
                        "Status": "inactive",
                        "Intensity (%)": "100%",
                        "Evidence": "Manual selection",
                    }
                )
            selectivity_value = "selective" if selectivity_enabled else None
            selectivity_strength = 1.0 if selectivity_enabled else None
            if selectivity_enabled:
                manual_effects.append(
                    {
                        "Cue": "Selectivity",
                        "Status": "selective",
                        "Intensity (%)": "100%",
                        "Evidence": "Manual selection",
                    }
                )
            toxicity_value = "toxic" if toxicity_enabled else None
            toxicity_strength = 1.0 if toxicity_enabled else None
            if toxicity_enabled:
                manual_effects.append(
                    {
                        "Cue": "Toxicity",
                        "Status": "toxic",
                        "Intensity (%)": "100%",
                        "Evidence": "Manual selection",
                    }
                )
            bioavailability_value = "bioavailable" if bioavailability_enabled else None
            bioavailability_strength = 1.0 if bioavailability_enabled else None
            if bioavailability_enabled:
                manual_effects.append(
                    {
                        "Cue": "Bioavailability",
                        "Status": "bioavailable",
                        "Intensity (%)": "100%",
                        "Evidence": "Manual selection",
                    }
                )
            medicinal_profile = audio_utils.MedicinalSoundProfile(
                activity=activity_value,
                activity_strength=activity_strength,
                selectivity=selectivity_value,
                selectivity_strength=selectivity_strength,
                toxicity=toxicity_value,
                toxicity_strength=toxicity_strength,
                bioavailability=bioavailability_value,
                bioavailability_strength=bioavailability_strength,
            )
            if manual_effects:
                st.markdown("**Manual sound design cues**")
                st.dataframe(pd.DataFrame(manual_effects), hide_index=True)

    profile_arg = None
    if medicinal_profile and not medicinal_profile.is_neutral():
        profile_arg = medicinal_profile

    waveform = audio_utils.generate_waveform(
        components,
        duration=duration,
        sample_rate=sample_rate,
        medicinal_profile=profile_arg,
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
    render_audio_section(matches, duration, sample_rate, molecule_info=info)


if __name__ == "__main__":
    main()
