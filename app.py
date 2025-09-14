import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import base64

# -------- User Config --------
SAMPLE_RATE = 22050
DURATION = 30
IMG_SHAPE = (128, 128)
MODEL_PATH = "best_model.keras"  # Your trained CNN model path
CLASSES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Path to uploaded background image
BACKGROUND_IMAGE_PATH = "C:/Users/Praabhass/Downloads/archive/wavy_background_3.jpg"

# -------- Set Background --------
def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image(BACKGROUND_IMAGE_PATH)

# -------- Utility Functions --------
def load_audio(path, sr=SAMPLE_RATE, duration=DURATION):
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration)
    except:
        y = np.zeros(sr * duration)
    expected_len = sr * duration
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    else:
        y = y[:expected_len]
    return y

def mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=128, hop_length=512, fmin=20, fmax=SAMPLE_RATE//2):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax)
    log_mel = librosa.power_to_db(mel)
    return log_mel

def resize_spectrogram(spec, size=IMG_SHAPE):
    spec = np.nan_to_num(spec)
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)
    spec_img = (spec_norm * 255).astype(np.uint8)
    resized = cv2.resize(spec_img, (size[1], size[0]))
    return resized.astype(np.float32) / 255.0

# -------- Load Model --------
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

# -------- Streamlit UI --------
st.title("🎵 Music Genre Classification")
st.write("Upload an audio file (WAV/MP3) and click **Proceed** to predict its genre.")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

if uploaded_file:
    # Load audio
    y = load_audio(uploaded_file)

    # Play audio
    st.subheader("🎧 Play Uploaded Audio")
    st.audio(uploaded_file, format='audio/wav')

    # Display waveform
    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=SAMPLE_RATE)
    st.pyplot(fig)

    # Proceed button
    if st.button("Proceed"):
        # Generate mel-spectrogram
        spec = mel_spectrogram(y)
        spec_img = resize_spectrogram(spec)
        input_img = np.expand_dims(spec_img, axis=(0,-1))

        # Predict genre
        pred_probs = model.predict(input_img)[0]
        pred_class = np.argmax(pred_probs)

        st.subheader("Predicted Genre")
        st.write(f"**{CLASSES[pred_class]}**")

        # Display prediction probabilities
        st.subheader("Prediction Probabilities")
        prob_dict = {CLASSES[i]: float(pred_probs[i]) for i in range(len(CLASSES))}
        st.bar_chart(prob_dict)

        # Display spectrogram
        st.subheader("Mel-Spectrogram")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(spec, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        st.pyplot(fig2)

