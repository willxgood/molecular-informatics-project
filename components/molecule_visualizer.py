"""Molecular visualizer that pulses functional groups in sync with audio playback."""
from __future__ import annotations

import json
from typing import Iterable, List, Optional, Tuple

import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdDepictor

from molecular_informatics import audio_utils, chem_utils

# Distinct colors for groups; cycled if more are needed.
PALETTE = [
    "#60a5fa",
    "#a78bfa",
    "#f472b6",
    "#f59e0b",
    "#34d399",
    "#22d3ee",
    "#fb7185",
    "#c084fc",
]


def _normalise_coords(mol: Chem.Mol) -> Tuple[List[dict], List[dict]]:
    """Return atom and bond coordinate payloads normalised to [0, 1]."""

    work = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(work)
    conf = work.GetConformer()
    xs, ys = [], []
    for idx in range(work.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        xs.append(pos.x)
        ys.append(pos.y)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)

    atoms = []
    for idx in range(work.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        atoms.append(
            {
                "id": idx,
                "x": (pos.x - min_x) / span_x,
                "y": (pos.y - min_y) / span_y,
            }
        )

    bonds = []
    for bond in work.GetBonds():
        bonds.append(
            {
                "a": bond.GetBeginAtomIdx(),
                "b": bond.GetEndAtomIdx(),
            }
        )
    return atoms, bonds


def _group_payload(
    matches: Iterable[chem_utils.FunctionalGroupMatch],
    *,
    audible_range: Tuple[float, float] = (100.0, 4000.0),
    wrap: bool = False,
    wrap_band: Tuple[float, float] = (110.0, 880.0),
) -> List[dict]:
    """Prepare per-group payload with atom indices and mapped frequency."""

    payload: List[dict] = []
    palette_len = len(PALETTE)
    for idx, match in enumerate(matches):
        if not match.present:
            continue
        freq = audio_utils.map_wavenumber_to_audible(
            match.group.center_wavenumber, audible_range=audible_range
        )
        if wrap:
            freq = audio_utils._wrap_frequency_to_band(  # type: ignore[attr-defined]
                freq, low=wrap_band[0], high=wrap_band[1]
            )
        wavenumber = float(match.group.center_wavenumber)
        color = PALETTE[idx % palette_len]
        # Flatten atom hits for drawing (may be empty if not returned)
        atom_ids = sorted({atom for hit in match.atom_matches for atom in hit}) if match.atom_matches else []
        payload.append(
            {
                "name": match.group.name,
                "atoms": atom_ids,
                "color": color,
                "frequency": float(freq),
                "wavenumber": wavenumber,
            }
        )
    return payload


def render_molecule_visualizer(
    *,
    mol: Chem.Mol,
    matches: Iterable[chem_utils.FunctionalGroupMatch],
    audio_element_id: Optional[str] = None,
    audible_range: Tuple[float, float] = (100.0, 4000.0),
    wrap: bool = False,
    wrap_band: Tuple[float, float] = (110.0, 880.0),
    width: int = 420,
    height: int = 360,
):
    """Render a 2D molecule with pulsing functional groups synced to audio playback."""

    atoms, bonds = _normalise_coords(mol)
    groups = _group_payload(
        matches,
        audible_range=audible_range,
        wrap=wrap,
        wrap_band=wrap_band,
    )
    if not atoms:
        return

    data = json.dumps({"atoms": atoms, "bonds": bonds, "groups": groups})
    audio_selector = f"document.getElementById('{audio_element_id}')" if audio_element_id else "null"

    html = f"""
    <div style="width:100%; max-width:{width}px; margin:auto;">
      <div style="color:#e5e7eb; font-weight:600; margin-bottom:6px;">FTIR vibration visualiser</div>
      <canvas id="mol-canvas" width="{width}" height="{height}" style="width:100%; background:linear-gradient(135deg,#0f172a 0%,#0b1224 100%); border:1px solid #1f2937; border-radius:10px;"></canvas>
    </div>
    <script>
      (function() {{
        const payload = {data};
        const audioEl = {audio_selector};
        const canvas = document.getElementById("mol-canvas");
        const ctx = canvas.getContext("2d");
        const padding = 36;
        const atomRadius = 6;

        function toCanvas(p) {{
          return {{
            x: padding + p.x * (canvas.width - 2*padding),
            y: padding + (1 - p.y) * (canvas.height - 2*padding),
          }};
        }}

        function drawFrame(timeMs) {{
          const t = (audioEl && audioEl.currentTime) ? audioEl.currentTime : timeMs / 1000;
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          // bonds
          ctx.strokeStyle = "#475569";
          ctx.lineWidth = 2;
          for (const bond of payload.bonds) {{
            const a = toCanvas(payload.atoms[bond.a]);
            const b = toCanvas(payload.atoms[bond.b]);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }}

          // group pulses
          for (const [idx, group] of payload.groups.entries()) {{
            const rate = Math.min(8, Math.max(1.5, (group.frequency || 2) / 120));
            const phase = Math.sin(2 * Math.PI * rate * t);
            const pulse = 0.4 + 0.4 * (0.5 + 0.5 * phase);
            ctx.fillStyle = group.color + "55";
            for (const atomIdx of group.atoms) {{
              const p = toCanvas(payload.atoms[atomIdx]);
              ctx.beginPath();
              ctx.arc(p.x, p.y, atomRadius * (1.6 + pulse), 0, Math.PI * 2);
              ctx.fill();
            }}
          }}

          // atoms
          for (const atom of payload.atoms) {{
            const p = toCanvas(atom);
            ctx.beginPath();
            ctx.fillStyle = "#cbd5f5";
            ctx.arc(p.x, p.y, atomRadius, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#0f172a";
            ctx.lineWidth = 1.2;
            ctx.stroke();
          }}

          // labels
          ctx.font = "12px Inter, -apple-system, BlinkMacSystemFont, sans-serif";
          ctx.fillStyle = "#e5e7eb";
          let y = canvas.height - 14;
          for (const group of payload.groups) {{
            ctx.fillStyle = group.color;
            ctx.fillRect(16, y - 9, 10, 10);
            ctx.fillStyle = "#e5e7eb";
            ctx.fillText(`${{group.name}} (${{
              group.wavenumber.toFixed(0)
            }} cm⁻¹ → ${{group.frequency.toFixed(0)}} Hz)`, 32, y);
            y -= 16;
          }}
          requestAnimationFrame(drawFrame);
        }}
        requestAnimationFrame(drawFrame);
      }})();
    </script>
    """
    st.components.v1.html(html, height=height + 40)
