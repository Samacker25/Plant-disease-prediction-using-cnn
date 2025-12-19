## 🌿 Plant Disease Prediction Using CNN

A deep learning–based web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN).
The application is built with TensorFlow/Keras and deployed using Streamlit, with the trained model hosted on Hugging Face for lightweight deployment.

## 🚀 Features

🌱 Classifies plant leaf images into healthy or diseased categories

🧠 CNN model trained on labeled plant disease image datasets

🖼️ Image preprocessing and normalization

⚡ Real-time predictions via a Streamlit web interface

☁️ Model hosted on Hugging Face Dataset Hub

🐳 Supports Docker-based deployment for production

💻 Works locally, on Streamlit Cloud, and Hugging Face Spaces

## 🧠 Model Overview

Architecture: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Input Size: 224 × 224 × 3

Output: Multi-class plant disease prediction

Model Format: TensorFlow SavedModel (compressed as .tar.gz)

Inference: serving_default signature (Keras 3 compatible)

## 🗂️ Project Structure

plant-disease-prediction-using-cnn/
│
├── app/
│   ├── main.py                # Streamlit application
│   ├── class_indices.json     # Class label mapping
│
├── trained_model/
│   └── plant_disease_prediction_model/
│
├── requirements.txt
├── Dockerfile
├── runtime.txt
├── README.md
└── .gitignore

## 📦 Model Hosting (Hugging Face)

The trained model is stored on Hugging Face to keep the GitHub repository lightweight.

## Model Dataset URL
 https://huggingface.co/datasets/Samacker25/plant-disease-prediction

The application automatically downloads and loads the model at runtime using:

from huggingface_hub import snapshot_download

🖥️ Running Locally
1️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit app
streamlit run app/main.py


Open browser at:
👉 http://localhost:8501

🐳 Docker Deployment (Recommended)
Build Docker image
docker build -t plant-disease-app .

Run container
docker run -p 7860:7860 plant-disease-app


Open browser at:
 http://localhost:7860

## ☁️ Deployment Options

✅ Streamlit Cloud (Python 3.10)

✅ Hugging Face Spaces (Docker SDK)

✅ Local / On-Prem Docker

✅ Cloud VM (AWS / Azure / GCP)

🧪 Example Prediction Flow

Upload a plant leaf image (.jpg, .jpeg, .png)

Image is resized and normalized

CNN model performs inference

Predicted disease label is displayed

## 🛠️ Tech Stack

Python 3.10

TensorFlow 2.15

Keras

Streamlit

NumPy

Pillow

Hugging Face Hub

Docker

## ⚠️ Notes on Compatibility

TensorFlow 2.15.0 requires Python ≤ 3.10

Hugging Face Spaces default Python is 3.13, so Docker SDK is required

Keras 3 does not support legacy SavedModel loading via load_model() → inference is done using model.signatures["serving_default"]

## 👤 Author

Soumen Kundu
🎓 MCA Graduate | Aspiring ML / MLOps Engineer

🔗 GitHub: https://github.com/Samacker25

🔗 LinkedIn: https://www.linkedin.com/in/Samacker25