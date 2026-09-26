"""Streamlit demo: upload an image -> depth map from Depth Anything V2 (S/B/L).
Run:  streamlit run app/streamlit_app.py
"""
import io
import time

import matplotlib
import numpy as np
import streamlit as st
import torch
from PIL import Image

from dav2.models import V2_KEYS, load_model, pick_device, sync

matplotlib.use("Agg")
st.set_page_config(page_title="Depth Anything V2 demo", layout="wide")
LABELS = {"v2-small": "Small (24.8M)", "v2-base": "Base (97.5M)", "v2-large": "Large (335M)"}


@st.cache_resource(show_spinner="Loading model (first run downloads weights)...")
def get_model(key: str):
    m = load_model(key)
    dummy = Image.fromarray(np.zeros((256, 256, 3), np.uint8))
    m.predict(dummy)  # warm-up so the first user request is not timed with lazy init
    return m


def colorize(disp: np.ndarray) -> np.ndarray:
    d = (disp - disp.min()) / max(float(disp.max() - disp.min()), 1e-8)
    return (matplotlib.colormaps["magma"](d)[..., :3] * 255).astype(np.uint8)  # bright = near


def run(key: str, img: Image.Image):
    m = get_model(key)
    sync(m.device); t0 = time.perf_counter()
    disp = m.predict(img)
    sync(m.device)
    return m, disp, (time.perf_counter() - t0) * 1000


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); return buf.getvalue()


st.title("Depth Anything V2 - monocular depth from one photo")
st.caption("Pretrained checkpoints, zero-shot, no training. Output is *relative* depth (bright = near), not metres.")

with st.sidebar:
    key = st.selectbox("Model size", V2_KEYS, format_func=LABELS.get, index=0)
    compare = st.checkbox("Compare all three sizes", value=False)
    st.write(f"Device: `{pick_device()}`")

up = st.file_uploader("Upload a .jpg or .png", type=["jpg", "jpeg", "png"])
if up is None:
    st.info("Upload an image to begin.")
    st.stop()

img = Image.open(up).convert("RGB")
keys = V2_KEYS if compare else [key]
cols = st.columns(1 + len(keys))
cols[0].image(img, caption=f"Input ({img.size[0]}x{img.size[1]})", use_container_width=True)
for col, k in zip(cols[1:], keys):
    m, disp, ms = run(k, img)
    col.image(colorize(disp), caption=f"{LABELS[k]} - {ms:.0f} ms", use_container_width=True)
    col.download_button("Download colour PNG", png_bytes(colorize(disp)), f"depth_{k}.png", "image/png", key=f"c{k}")
    d16 = ((disp - disp.min()) / max(float(disp.max() - disp.min()), 1e-8) * 65535).astype(np.uint16)
    b = io.BytesIO(); Image.fromarray(d16).save(b, format="PNG")
    col.download_button("Download 16-bit PNG", b.getvalue(), f"depth16_{k}.png", "image/png", key=f"r{k}")
    col.caption(f"{m.params_m:.1f}M params | device: {m.device}")
