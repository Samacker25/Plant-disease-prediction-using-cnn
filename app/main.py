import os
import json
from PIL import Image

import numpy as np
import streamlit as st
from huggingface_hub import snapshot_download

from keras import Sequential
from keras.layers import TFSMLayer

# --------------------------------------------------
# Paths
# --------------------------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Load model from Hugging Face (Keras 3 safe)
# --------------------------------------------------
@st.cache_resource
def load_model():
    with st.spinner("📥 Loading model from Hugging Face..."):
        repo_dir = snapshot_download(
            repo_id="Samacker25/plant-disease-prediction",
            repo_type="dataset",
            allow_patterns=["plant_disease_prediction_model/**"],
            local_dir_use_symlinks=False
        )

        model_dir = os.path.join(repo_dir, "plant_disease_prediction_model")

        model = Sequential([
            TFSMLayer(
                model_dir,
                call_endpoint="serving_default"
            )
        ])

        return model


model = load_model()

# --------------------------------------------------
# Load class indices
# --------------------------------------------------
with open(os.path.join(working_dir, "class_indices.json")) as f:
    class_indices = json.load(f)

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    image = Image.open(image_file).convert("RGB")
    image = image.resize(target_size)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict_image_class(model, image_file, class_indices):
    img = load_and_preprocess_image(image_file)
    preds = model.predict(img)

    class_id = str(np.argmax(preds[0]))
    return class_indices[class_id]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🌿 Plant Disease Classifier")

uploaded_image = st.file_uploader(
    "Upload an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image.resize((150, 150)))

    with col2:
        if st.button("Classify"):
            with st.spinner("🔍 Predicting..."):
                result = predict_image_class(
                    model,
                    uploaded_image,
                    class_indices
                )
                st.success(f"Prediction: {result}")

# --------------------------------------------------
# Background & footer
# --------------------------------------------------
def set_bg_from_url(url, opacity=1):
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <footer style="margin-top:6rem;text-align:center;">
            <p>
                Made by Soumen Kundu |
                <a href="https://www.linkedin.com/in/Samacker25">LinkedIn</a> |
                <a href="https://github.com/Samacker25">GitHub</a>
            </p>
        </footer>
        """,
        unsafe_allow_html=True
    )


set_bg_from_url(
    "https://t4.ftcdn.net/jpg/02/83/36/47/240_F_283364706_EyAbCnZVtqp0qFBNJv06ld8p1rGW7PWB.jpg",
    opacity=0.7
)
