import os
import json
from PIL import Image

import numpy as np
import streamlit as st
from huggingface_hub import snapshot_download
import tensorflow as tf
import tarfile

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
            allow_patterns=["plant_disease_prediction_model.tar.gz"]
        )

        tar_path = os.path.join(repo_dir, "plant_disease_prediction_model.tar.gz")

        if not os.path.exists(tar_path):
            raise FileNotFoundError(
                f"Model archive not found after download: {tar_path}"
            )
        
        extract_dir = os.path.join(repo_dir, "model")

        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)

        # Auto-detect SavedModel folder
        for root, dirs, files in os.walk(extract_dir):
            if "saved_model.pb" in files:
                model_path = root
                break
        else:
            raise FileNotFoundError("saved_model.pb not found after extraction")

        model = tf.keras.models.load_model(model_path)
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
    infer = model.signatures["serving_default"]

    outputs = infer(tf.constant(img))
    preds = list(outputs.values())[0].numpy()

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
                    model=model,
                    image_file=uploaded_image,
                    class_indices=class_indices
                )
                st.success(f"Prediction: {result}")

# st.write("Model type:", type(model))


# --------------------------------------------------
# Background & footer
# --------------------------------------------------
def set_bg_from_url(url, opacity=1):
    
    footer = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">
    <footer>
        <div style='visibility: visible;margin-top:7rem;justify-content:center;display:flex;'>
            <p style="font-size:1.1rem;">
                Made by Soumen Kundu
                &nbsp;
                <a href="https://www.linkedin.com/in/Samacker25">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="white" class="bi bi-linkedin" viewBox="0 0 16 16">
                        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                    </svg>          
                </a>
                &nbsp;
                <a href="https://github.com/Samacker25">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="white" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                </a>
            </p>
        </div>
    </footer>
"""
    st.markdown(footer, unsafe_allow_html=True)

    # Set background image using HTML and CSS
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



# Set background image from URL
set_bg_from_url("https://t4.ftcdn.net/jpg/02/83/36/47/240_F_283364706_EyAbCnZVtqp0qFBNJv06ld8p1rGW7PWB.jpg",
    opacity=0.7)
