🎵 Music Genre Classification
📌 Overview

This project predicts the genre of music tracks using deep learning.
It is trained on the GTZAN dataset and uses MFCC features with a CNN model.

🚀 Features

Classifies audio into 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).

Streamlit app for uploading audio and getting predictions.

Displays waveform, spectrogram, and confidence scores.

⚙️ Installation
pip install -r requirements.txt

▶️ Usage

Run the Streamlit app:

streamlit run src/app.py


Upload a .wav or .mp3 file to get the predicted genre.

📌 Tools Used

Python, Librosa, TensorFlow/Keras, Matplotlib, Streamlit
