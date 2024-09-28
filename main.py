import os
import json
import requests
import tempfile
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Set up page config
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?export=download&id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"
    
    with st.spinner("Downloading model... This may take a while."):
        try:
            response = requests.get(model_url)
            response.raise_for_status()
            
            # Save the model to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            # Load the model using TensorFlow's load_model function
            model = tf.keras.models.load_model(tmp_file_path)
            
            # Remove the temporary file
            os.unlink(tmp_file_path)
            
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            return None

# Load the model
model = load_model()

if model is None:
    st.stop()  # Stop the app if model loading failed
# Load class indices
class_indices_path = "class_indices.json"  # Make sure this file is in your repository
try:
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
except FileNotFoundError:
    st.error(f"Class indices file not found at {class_indices_path}. Please ensure the file exists in the correct location.")
    st.stop()

# Use class_indices directly, no need to reverse it
class_names = class_indices

def predict_disease(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return class_names[str(predicted_class)], confidence  # Convert predicted_class to string
