"""Looping audio player with waveform visual for Streamlit."""
from __future__ import annotations

import base64
import json
import uuid
from typing import Optional, Sequence

import numpy as np
import streamlit as st


def _normalise_waveform(waveform: np.ndarray, max_points: int = 1500) -> Sequence[float]:
    """Downsample and normalise a waveform for lightweight canvas rendering."""

    if waveform.size == 0:
        return [0.0]

    step = max(1, int(len(waveform) / max_points))
    trimmed = waveform[::step]
    peak = float(np.max(np.abs(trimmed))) if trimmed.size else 1.0
    peak = peak if peak > 0 else 1.0
    return (trimmed / peak).astype(float).tolist()


def render_looping_audio_player(
    *,
    audio_bytes: bytes,
    waveform: np.ndarray,
    sample_rate: int,
    loop: bool = True,
    autoplay: bool = True,
    height: int = 240,
) -> Optional[str]:
    """Embed a custom looping player with a lightweight waveform visual.

    Returns the audio element ID so other components can sync to playback.
    """

    if not isinstance(audio_bytes, (bytes, bytearray)):
        return None

    audio_b64 = base64.b64encode(audio_bytes).decode()
    uid = uuid.uuid4().hex
    norm_wave = _normalise_waveform(waveform)

    loop_attr = "loop" if loop else ""
    autoplay_attr = "autoplay" if autoplay else ""
    wave_json = json.dumps(norm_wave)

    html = f"""
    <div style="width:100%; border:1px solid #e5e7eb; border-radius:8px; padding:12px; background:linear-gradient(135deg,#0f172a 0%,#111827 100%);">
      <audio id="audio-{uid}" controls {loop_attr} {autoplay_attr} style="width:100%; margin-bottom:8px;">
        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
      <canvas id="wave-{uid}" style="width:100%; height:160px; display:block; background:rgba(255,255,255,0.04); border-radius:6px;"></canvas>
      <script>
        (function() {{
          const data = {wave_json};
          const audio = document.getElementById("audio-{uid}");
          const canvas = document.getElementById("wave-{uid}");
          const ctx = canvas.getContext("2d");
          const desiredHeight = 160;

          function resize() {{
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = desiredHeight * dpr;
            canvas.style.width = rect.width + "px";
            canvas.style.height = desiredHeight + "px";
            ctx.resetTransform();
            ctx.scale(dpr, dpr);
            drawWaveform();
          }}

          function drawWaveform(progress) {{
            const rect = canvas.getBoundingClientRect();
            const width = rect.width;
            const height = desiredHeight;
            const mid = height / 2;
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = "#0b1222";
            ctx.fillRect(0, 0, width, height);

            const step = data.length / width || 1;
            ctx.strokeStyle = "#60a5fa";
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let x = 0; x < width; x++) {{
              const idx = Math.min(data.length - 1, Math.floor(x * step));
              const v = data[idx] || 0;
              const y = mid - v * (height / 2 - 8);
              if (x === 0) {{ ctx.moveTo(x, y); }} else {{ ctx.lineTo(x, y); }}
            }}
            ctx.stroke();

            if (progress !== undefined) {{
              const px = width * progress;
              ctx.fillStyle = "rgba(96, 165, 250, 0.18)";
              ctx.fillRect(0, 0, px, height);
              ctx.fillStyle = "#2563eb";
              ctx.fillRect(px, 0, 2, height);
            }}
          }}

          let raf = null;
          function tick() {{
            if (!audio.duration) {{
              raf = requestAnimationFrame(tick);
              return;
            }}
            const progress = Math.min(1, audio.currentTime / audio.duration);
            drawWaveform(progress);
            raf = requestAnimationFrame(tick);
          }}

          audio.addEventListener("play", () => {{
            if (raf) cancelAnimationFrame(raf);
            tick();
          }});
          audio.addEventListener("pause", () => raf && cancelAnimationFrame(raf));
          audio.addEventListener("ended", () => raf && cancelAnimationFrame(raf));
          window.addEventListener("resize", resize);
          resize();

          if ({str(bool(autoplay)).lower()}) {{
            audio.addEventListener("loadedmetadata", () => {{
              audio.play().catch(() => {{}});
            }});
          }}
        }})();
      </script>
    </div>
    """
    st.components.v1.html(html, height=height)
    return f"audio-{uid}"
