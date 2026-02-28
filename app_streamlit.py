from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from conase_geo.predict import GeoPredictor

st.set_page_config(page_title="Dialect Location Guesser", layout="centered")
st.title("Dialect Location Guesser")
st.caption("Toy game: upload a short speech clip and get top-k likely location labels.")

checkpoint_path = st.text_input("Checkpoint path", value="checkpoints/best.pt")
top_k = st.slider("Top-k predictions", min_value=1, max_value=10, value=5)
token_times_input = st.text_area(
    "Optional token_times_json (only needed for timing-only models)",
    value="",
    help="Paste JSON list of token timestamps if you want timing-only or timing+audio to use token timing.",
)


@st.cache_resource(show_spinner=False)
def load_predictor(path: str) -> GeoPredictor:
    return GeoPredictor(path)


uploaded = st.file_uploader("Upload speech audio", type=["wav", "mp3", "m4a", "flac", "ogg"])
if st.button("Guess Location"):
    if uploaded is None:
        st.error("Upload an audio file first.")
    elif not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
    else:
        suffix = Path(uploaded.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded.read())
            temp_path = Path(temp.name)

        predictor = load_predictor(checkpoint_path)
        preds = predictor.predict(
            audio_path=temp_path,
            top_k=top_k,
            token_times_json=token_times_input.strip() or None,
        )
        if not preds:
            st.warning("No predictions produced.")
        else:
            best_label, best_prob = preds[0]
            st.subheader(f"Guess: {best_label} ({best_prob:.2%})")
            st.write("Top predictions:")
            for rank, (label, prob) in enumerate(preds, start=1):
                st.write(f"{rank}. {label}: {prob:.2%}")
