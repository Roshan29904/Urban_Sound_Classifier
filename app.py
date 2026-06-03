import os
import tempfile
 
import numpy as np
import streamlit as st
 
from config import CLASS_NAMES
from model_loader import load_assets
from preprocess import preprocess_audio
 
 
st.set_page_config(page_title="Urban Sound Classifier", page_icon="🔊", layout="centered")
st.title("🔊 Urban Sound Classifier")
 
 
@st.cache_resource
def initialize_model():
    return load_assets()
 
 
model, _mean, _std = initialize_model()
 
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
 
if uploaded_file is not None:
    audio_bytes = uploaded_file.getvalue()
 
    # Lowercase extension so st.audio format param is always valid (e.g. .WAV → wav)
    ext = os.path.splitext(uploaded_file.name)[-1][1:].lower()
    st.audio(audio_bytes, format=f"audio/{ext}")
 
    if st.button("Classify audio"):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp:
                temp.write(audio_bytes)
                temp_path = temp.name
 
            with st.spinner("Classifying audio..."):
                x_input = preprocess_audio(temp_path)
                raw_preds = model.predict(x_input, verbose=0)[0]
 
            # Model's final layer is softmax — outputs already sum to 1.0, use directly.
            preds = np.asarray(raw_preds, dtype=np.float64)
            top_indices = preds.argsort()[::-1]  # all 10, sorted from best to worst
 
            best_idx = top_indices[0]
            st.success(
                f"Top prediction: **{CLASS_NAMES[best_idx]}** "
                f"({preds[best_idx] * 100:.2f}%)"
            )
 
            # Show ALL 10 classes so confidences add up to 100%
            st.write("### All class probabilities")
            for idx in top_indices:
                label = CLASS_NAMES[idx]
                confidence = float(preds[idx])
                st.write(f"**{label}** — {confidence * 100:.2f}%")
                st.progress(confidence)
 
        except Exception as error:
            st.error(f"Prediction failed: {error}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)