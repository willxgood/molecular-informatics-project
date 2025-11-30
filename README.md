# Molecular FTIR Sound Generator

Turn molecular structures into FTIR-inspired audio, explore functional groups, and arrange clips in a simple piano roll. Built with Streamlit + RDKit.

## Features
- Resolve molecules from SMILES, common names, or PubChem CID (if `requests` + network available).
- Detect functional groups via SMARTS patterns and map their FTIR centers into the audible range.
- Generate and download molecular “soundscapes”; layer clips in a piano-roll arranger with synth presets and effects.
- Optional Ketcher widget for drawing molecules; presets and history for quick starts.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Optional extras
- `streamlit-ketcher` to draw molecules instead of typing SMILES.
- `plotly` to render the piano-roll grid visualization.

Install with:
```bash
pip install streamlit-ketcher plotly
```

## Project layout
- `streamlit_app.py` — main Streamlit UI.
- `molecular_informatics/` — core logic (functional groups, audio mapping, arranger UI, effects).
- `components/` — shared UI components (audio player, molecule visualizer).
- `requirements.txt` — base dependencies.

## Notes for publishing
- License: MIT license included in `LICENSE`.
- No secrets are stored in the repo; PubChem lookups happen client-side via `requests` when network is available.
- Tested on Python 3.10+; RDKit wheels come from `rdkit-pypi` for ease of install.

## Contributing
Pull requests are welcome. Please open an issue if you find a bug or want a new functional group/preset.
