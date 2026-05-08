import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

# --- Configuration & Techie Styling ---
st.set_page_config(page_title="Music Genre Classifier", layout="wide")

# Injecting Custom CSS for the "Techie" Vibe
st.markdown("""
    <style>
    /* Force monospace terminal-style fonts across the app */
    html, body, [class*="css"]  {
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Style the prediction result boxes */
    div.stInfo {
        background-color: rgba(0, 255, 100, 0.05); /* Slight neon green tint */
        border: 1px solid #00ff64;
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Loads the encoder, scaler, and trained models into memory once."""
    le = joblib.load('label_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    models = {
        "K-Nearest Neighbors": joblib.load('knn.joblib'),
        "Support Vector Machine": joblib.load('svc.joblib'),
        "Random Forest": joblib.load('rf.joblib'),
        "Gradient Boosting": joblib.load('gb.joblib')
    }
    return le, scaler, models

# Load artifacts
try:
    label_encoder, scaler, models = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found! Please run `train_pipeline.py` first.")
    st.stop()

# --- Feature Extraction ---
def extract_features(audio_path):
    """Extracts features from the uploaded audio file to match training data."""
    y, sr = librosa.load(audio_path, duration=30, sr=None)
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    features = np.hstack([
        np.mean(chroma_stft), np.var(chroma_stft),
        np.mean(spectral_centroid), np.var(spectral_centroid),
        np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
        *[np.mean(mfcc[i]) for i in range(20)], *[np.var(mfcc[i]) for i in range(20)]
    ])
    
    return features.reshape(1, -1) 

# --- User Interface ---
st.title("Music Genre Classifier")
st.markdown("*- this is a research approach to train machine learning models for a prediction task. Base ML models are not recommended for this task as they have a very low accuracy. This model is for demo purpose only, Furthermore, if you require better prediction accuracy, consider DNNs*")
st.markdown("---")

# Model Definitions
st.markdown("### Initialized Models:")
st.markdown("""
* **K-Nearest Neighbors (KNN):** Classifies audio by finding the most mathematically similar songs in the training data.
* **Support Vector Machine (SVM):** Draws complex, high-dimensional boundaries to separate different musical profiles.
* **Random Forest:** Uses a massive committee of randomized decision trees to vote on the most likely genre.
* **Gradient Boosting:** Builds decision trees sequentially, with each new tree actively correcting the errors of the previous ones.
""")
st.markdown("---")

# --- Main Area for File Upload ---
st.markdown("### Execute Analysis")
uploaded_file = st.file_uploader("Upload Audio Target (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Initialize Processing Sequence"):
        with st.spinner("Extracting parameters and running models..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(uploaded_file.read())
                temp_audio_path = temp_audio.name
            
            try:
                # 1. Extract & Scale Features
                raw_features = extract_features(temp_audio_path)
                scaled_features = scaler.transform(raw_features)
                
                # Hardcoded accuracies from your previous training logs
                model_accuracies = {
                    "K-Nearest Neighbors": "63.5%",
                    "Support Vector Machine": "68.5%",
                    "Random Forest": "65.5%",
                    "Gradient Boosting": "69.0%"
                }
                
                st.markdown("### Prediction Results:")
                # Create a responsive 4-column layout for the results
                cols = st.columns(4)
                
                # Loop through all models and print results side-by-side
                for idx, (name, model) in enumerate(models.items()):
                    prediction_num = model.predict(scaled_features)
                    predicted_genre = label_encoder.inverse_transform(prediction_num)[0]
                    
                    with cols[idx]:
                        st.info(f"**{name}**\n\n🎯 **{predicted_genre.upper()}**\n\n*Acc: {model_accuracies[name]}*")
                
            except Exception as e:
                st.error(f"Execution Error: {e}")
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
